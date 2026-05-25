from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "emotion" / "fer2013" / "fer2013.csv"
MODEL_PATH = ROOT / "models" / "fer2013_cnn.pt"
REPORT_PATH = ROOT / "reports" / "emotion_cnn_metrics.json"


class EmotionCNN(nn.Module):
    def __init__(self, num_classes: int = 7):
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
    parser = argparse.ArgumentParser(description="Train a CNN on FER-2013.")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=8e-4)
    parser.add_argument("--binary_angry", action="store_true", help="Train a customer-service angry-vs-other CNN.")
    parser.add_argument("--class_weight", action="store_true", help="Use balanced class weights.")
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH, nrows=args.max_samples or None)
    images = np.stack([np.fromstring(pixels, sep=" ", dtype=np.float32).reshape(48, 48) for pixels in df["pixels"]])
    labels = df["emotion"].astype(int).to_numpy()
    if args.binary_angry:
        labels = (labels == 0).astype(int)
    images = (images / 255.0)[:, None, :, :]

    train_x, test_x, train_y, test_y = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long)),
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_tensor = torch.ones(int(labels.max()) + 1, dtype=torch.float32)
    if args.class_weight:
        classes = np.unique(train_y)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_y)
        for class_id, weight in zip(classes, class_weights):
            weight_tensor[int(class_id)] = float(weight)
    model = EmotionCNN(num_classes=int(labels.max()) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1
        print(f"epoch={epoch + 1} loss={running_loss / max(1, batches):.4f}")

    y_true, y_pred = evaluate(model, test_loader, device)
    metrics = {
        "model": "2-layer CNN",
        "task": "FER-2013 angry-vs-other CNN classification" if args.binary_angry else "FER-2013 facial expression classification",
        "mode": "sampled" if args.max_samples else "full",
        "samples": int(len(df)),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "binary_angry": args.binary_angry,
        "class_weight": args.class_weight,
        "architecture": "enhanced_cnn_v2",
        "num_classes": int(labels.max()) + 1,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "enhanced_cnn_v2",
            "num_classes": int(labels.max()) + 1,
            "binary_angry": args.binary_angry,
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    REPORT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def evaluate(model, loader, device):
    model.eval()
    truth, pred = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)).cpu()
            truth.extend(y.numpy().tolist())
            pred.extend(logits.argmax(dim=1).numpy().tolist())
    return truth, pred


if __name__ == "__main__":
    main()
