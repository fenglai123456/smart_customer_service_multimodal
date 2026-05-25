from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
LIBRISPEECH = ROOT / "datasets" / "audio" / "librispeech" / "LibriSpeech"
MODEL_PATH = ROOT / "models" / "librispeech_lstm.pt"
REPORT_PATH = ROOT / "reports" / "audio_lstm_metrics.json"


class AudioLSTM(nn.Module):
    def __init__(self, n_mfcc: int, hidden_size: int, num_classes: int, num_layers: int = 2):
        super().__init__()
        dropout = 0.2 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(n_mfcc, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.classifier(self.dropout(hidden[-1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an LSTM acoustic baseline on LibriSpeech.")
    parser.add_argument("--subset", default="dev-clean", choices=["dev-clean", "train-clean-100"])
    parser.add_argument("--max_samples", type=int, default=240)
    parser.add_argument("--top_k_speakers", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=96)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=8e-4)
    args = parser.parse_args()

    items = collect_items(LIBRISPEECH / args.subset, args.max_samples, args.top_k_speakers)
    label_to_id = {label: idx for idx, label in enumerate(sorted({item[1] for item in items}))}
    features, labels = [], []
    for audio_path, label in items:
        features.append(extract_mfcc(audio_path))
        labels.append(label_to_id[label])

    x = np.stack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    train_x, test_x, train_y, test_y = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_x), torch.tensor(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(TensorDataset(torch.tensor(test_x), torch.tensor(test_y)), batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioLSTM(
        n_mfcc=20,
        hidden_size=args.hidden_size,
        num_classes=len(label_to_id),
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1
        print(f"epoch={epoch + 1} loss={running_loss / max(1, batches):.4f}")

    truth, pred = evaluate(model, test_loader, device)
    id_to_label = {value: key for key, value in label_to_id.items()}
    truth_labels = [id_to_label[item] for item in truth]
    pred_labels = [id_to_label[item] for item in pred]
    metrics = {
        "model": "MFCC + LSTM",
        "task": "LibriSpeech acoustic speaker-proxy classification",
        "note": "LibriSpeech has speech/transcript data but no customer-service intent labels; this trains the requested LSTM acoustic branch.",
        "subset": args.subset,
        "samples": len(items),
        "epochs": args.epochs,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "learning_rate": args.learning_rate,
        "accuracy": round(float(accuracy_score(truth_labels, pred_labels)), 4),
        "classification_report": classification_report(truth_labels, pred_labels, output_dict=True, zero_division=0),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_id": label_to_id,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    REPORT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def collect_items(subset_dir: Path, max_samples: int, top_k_speakers: int) -> list[tuple[Path, str]]:
    flacs = sorted(subset_dir.rglob("*.flac"))
    speaker_counts = Counter(path.parts[-3] for path in flacs)
    speakers = [speaker for speaker, _ in speaker_counts.most_common(top_k_speakers)]
    by_speaker = defaultdict(list)
    for path in flacs:
        speaker = path.parts[-3]
        if speaker in speakers:
            by_speaker[speaker].append((path, speaker))

    per_speaker = max(1, max_samples // len(speakers))
    selected = []
    for speaker in speakers:
        selected.extend(by_speaker[speaker][:per_speaker])
    return selected


def extract_mfcc(path: Path, sr: int = 16000, n_mfcc: int = 20, max_frames: int = 120) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, duration=3.0)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
    if len(mfcc) < max_frames:
        pad = np.zeros((max_frames - len(mfcc), n_mfcc), dtype=np.float32)
        mfcc = np.vstack([mfcc, pad])
    return mfcc[:max_frames]


def evaluate(model, loader, device):
    model.eval()
    truth, pred = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device)).cpu()
            truth.extend(batch_y.numpy().tolist())
            pred.extend(logits.argmax(dim=1).numpy().tolist())
    return truth, pred


if __name__ == "__main__":
    main()
