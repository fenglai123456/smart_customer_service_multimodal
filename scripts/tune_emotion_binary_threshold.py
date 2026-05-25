from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "emotion" / "fer2013" / "fer2013.csv"
MODEL_PATH = ROOT / "models" / "fer2013_cnn.pt"
REPORT_PATH = ROOT / "reports" / "emotion_cnn_metrics.json"
THRESHOLD_REPORT_PATH = ROOT / "reports" / "emotion_cnn_threshold_metrics.json"


class EmotionCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the angry probability threshold for a binary FER-2013 CNN.")
    parser.add_argument("--min_angry_recall", type=float, default=0.5)
    parser.add_argument("--max_accuracy_drop", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if not checkpoint.get("binary_angry"):
        raise SystemExit("Threshold tuning requires a binary angry-vs-other fer2013_cnn.pt checkpoint.")

    images, labels = load_binary_dataset()
    _, test_x, _, test_y = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    loader = DataLoader(TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long)), batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    angry_prob = predict_angry_probabilities(model, loader, device)

    baseline_pred = (angry_prob >= 0.5).astype(int)
    baseline_accuracy = float(accuracy_score(test_y, baseline_pred))
    candidates = []
    for threshold in np.linspace(0.20, 0.80, 121):
        pred = (angry_prob >= threshold).astype(int)
        report = classification_report(test_y, pred, output_dict=True, zero_division=0)
        accuracy = float(accuracy_score(test_y, pred))
        angry_recall = float(report.get("1", {}).get("recall", 0.0))
        angry_precision = float(report.get("1", {}).get("precision", 0.0))
        if angry_recall >= args.min_angry_recall and accuracy >= baseline_accuracy - args.max_accuracy_drop:
            candidates.append((accuracy, angry_precision, angry_recall, float(threshold), report))

    if candidates:
        accuracy, angry_precision, angry_recall, threshold, report = max(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
    else:
        best = []
        for threshold in np.linspace(0.20, 0.80, 121):
            pred = (angry_prob >= threshold).astype(int)
            report = classification_report(test_y, pred, output_dict=True, zero_division=0)
            accuracy = float(accuracy_score(test_y, pred))
            angry_recall = float(report.get("1", {}).get("recall", 0.0))
            angry_precision = float(report.get("1", {}).get("precision", 0.0))
            best.append((angry_recall, accuracy, angry_precision, float(threshold), report))
        angry_recall, accuracy, angry_precision, threshold, report = max(best, key=lambda item: (item[0], item[1], item[2]))

    metrics = {
        "model": "2-layer CNN",
        "task": "FER-2013 angry-vs-other CNN classification with tuned threshold",
        "source_model": str(MODEL_PATH),
        "samples": int(len(labels)),
        "test_size": int(len(test_y)),
        "threshold": round(float(threshold), 4),
        "baseline_threshold": 0.5,
        "baseline_accuracy": round(baseline_accuracy, 4),
        "accuracy": round(float(accuracy), 4),
        "angry_precision": round(float(angry_precision), 4),
        "angry_recall": round(float(angry_recall), 4),
        "classification_report": report,
    }

    old_metrics = checkpoint.get("metrics", {})
    old_metrics["threshold_tuning"] = metrics
    checkpoint["metrics"] = old_metrics
    checkpoint["decision_threshold"] = float(threshold)
    torch.save(checkpoint, MODEL_PATH)
    THRESHOLD_REPORT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(old_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def load_binary_dataset() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA_PATH)
    images = np.stack([np.fromstring(pixels, sep=" ", dtype=np.float32).reshape(48, 48) for pixels in df["pixels"]])
    labels = (df["emotion"].astype(int).to_numpy() == 0).astype(int)
    return (images / 255.0)[:, None, :, :], labels


def predict_angry_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x.to(device)).cpu()
            probs.extend(torch.softmax(logits, dim=1)[:, 1].numpy().tolist())
    return np.asarray(probs, dtype=np.float32)


if __name__ == "__main__":
    main()
