from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "demo_assets" / "customer_screenshots"
MODEL_PATH = ROOT / "models" / "demo_screenshot_intent_cnn.pt"
REPORT_PATH = ROOT / "reports" / "demo_screenshot_intent_metrics.json"
LABELS = ["fault", "order", "refund", "howto", "normal"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DemoScreenshotDataset(Dataset):
    def __init__(self, items: list[tuple[Path, str]], label_to_id: dict[str, int], image_size: int):
        self.items = items
        self.label_to_id = label_to_id
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        image = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.tensor(arr), torch.tensor(self.label_to_id[label], dtype=torch.long)


class DemoScreenshotCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)),
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 192),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train demo-only screenshot intent CNN for course presentation.")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()

    train_items = collect_items("train")
    test_items = collect_items("test")
    if not train_items or not test_items:
        raise SystemExit("Demo screenshots are missing. Run scripts/generate_demo_customer_screenshots.py first.")

    label_to_id = {label: index for index, label in enumerate(LABELS)}
    train_loader = DataLoader(
        DemoScreenshotDataset(train_items, label_to_id, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        DemoScreenshotDataset(test_items, label_to_id, args.image_size),
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DemoScreenshotCNN(num_classes=len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        truth, pred = evaluate(model, test_loader, device)
        accuracy = float(accuracy_score(truth, pred))
        epoch_loss = total_loss / max(1, batches)
        history.append({"epoch": epoch + 1, "loss": round(epoch_loss, 6), "accuracy": round(accuracy, 6)})
        print(f"epoch={epoch + 1} loss={epoch_loss:.4f} test_accuracy={accuracy:.4f}")

    truth, pred = evaluate(model, test_loader, device)
    metrics = {
        "model": "DemoScreenshotCNN",
        "task": "demo-only five-class customer screenshot intent classification",
        "strict_sources": False,
        "usage": "course_demo_only",
        "labels": LABELS,
        "train_size": len(train_items),
        "test_size": len(test_items),
        "label_counts": {
            "train": dict(sorted(Counter(label for _, label in train_items).items())),
            "test": dict(sorted(Counter(label for _, label in test_items).items())),
        },
        "epochs": args.epochs,
        "image_size": args.image_size,
        "device": str(device),
        "accuracy": round(float(accuracy_score(truth, pred)), 4),
        "history": history,
        "classification_report": classification_report(truth, pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": LABELS, "matrix": confusion_matrix(truth, pred, labels=LABELS).tolist()},
        "note": "Demo model is intentionally separate from strict public-data model and must not be used as strict compliance evidence.",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_id": label_to_id,
            "image_size": args.image_size,
            "architecture": "demo_screenshot_cnn_v1",
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def collect_items(split: str) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for label in LABELS:
        folder = DATA_DIR / split / label
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((path, label))
    return items


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[str], list[str]]:
    model.eval()
    truth: list[str] = []
    pred: list[str] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device)).cpu()
            truth.extend(LABELS[int(index)] for index in batch_y.numpy())
            pred.extend(LABELS[int(index)] for index in logits.argmax(dim=1).numpy())
    return truth, pred


if __name__ == "__main__":
    main()
