from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
COCO_VAL = ROOT / "datasets" / "images" / "coco" / "val2017"
CUSTOMER_SCREENSHOT_DIR = ROOT / "datasets" / "images" / "customer_screenshots"
MODEL_PATH = ROOT / "models" / "customer_image_binary_cnn.pt"
REPORT_PATH = ROOT / "reports" / "customer_image_binary_metrics.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class BinaryCNN(nn.Module):
    def __init__(self, image_size: int):
        super().__init__()
        pooled = image_size // 4
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * pooled * pooled, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train binary CNN: synthetic customer screenshots vs natural COCO images.")
    parser.add_argument("--samples_per_class", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--strict_sources",
        action="store_true",
        help="Use only COCO + real/public customer screenshots; synthetic screenshots are forbidden.",
    )
    args = parser.parse_args()

    x, y, source_note = build_dataset(args.samples_per_class, args.image_size, strict_sources=args.strict_sources)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
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
    model = BinaryCNN(args.image_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    truth, pred = evaluate(model, test_loader, device)
    target_names = ["natural_image", "customer_screenshot"]
    metrics = {
        "model": "2-layer CNN",
        "task": "customer screenshot vs natural image classification",
        "strict_sources": bool(args.strict_sources),
        "source_contract": "COCO + customer screenshot public subset" if args.strict_sources else "COCO + synthetic screenshots",
        "source_note": source_note,
        "samples": int(len(y)),
        "epochs": args.epochs,
        "accuracy": round(float(accuracy_score(truth, pred)), 4),
        "classification_report": classification_report(
            truth,
            pred,
            labels=[0, 1],
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metrics": metrics}, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_dataset(samples_per_class: int, image_size: int, strict_sources: bool) -> tuple[np.ndarray, np.ndarray, str]:
    natural_paths = sorted(COCO_VAL.glob("*.jpg"))[:samples_per_class]
    natural = [load_image(path, image_size) for path in natural_paths]
    if strict_sources:
        screenshot_paths = collect_customer_screenshots(samples_per_class)
        if not screenshot_paths:
            raise SystemExit(
                "Strict image training requires real/public customer screenshots under "
                f"{CUSTOMER_SCREENSHOT_DIR}. Synthetic screenshots are not allowed."
            )
        screenshots = [load_image(path, image_size) for path in screenshot_paths]
        source_note = "real/public customer screenshots from datasets/images/customer_screenshots"
    else:
        screenshots = [make_screenshot_like(i, image_size) for i in range(samples_per_class)]
        source_note = "synthetic screenshots; demo only, not strict-compliant"
    if not natural:
        raise SystemExit(f"COCO images are missing: {COCO_VAL}")
    x = np.stack(natural + screenshots).astype(np.float32)
    y = np.array([0] * len(natural) + [1] * len(screenshots), dtype=np.int64)
    return x, y, source_note


def collect_customer_screenshots(limit: int) -> list[Path]:
    paths = [
        path
        for path in sorted(CUSTOMER_SCREENSHOT_DIR.rglob("*"))
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()
    ]
    return paths[:limit]


def load_image(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0


def make_screenshot_like(index: int, image_size: int) -> np.ndarray:
    rng = np.random.default_rng(index)
    image = Image.new("RGB", (image_size, image_size), (245, 248, 252))
    draw = ImageDraw.Draw(image)
    palette = [(37, 99, 235), (220, 38, 38), (15, 118, 110), (31, 41, 55)]
    draw.rectangle((0, 0, image_size, 9), fill=(37, 99, 235))
    for row in range(5):
        top = 14 + row * 9
        left = 5 + int(rng.integers(0, 5))
        width = int(rng.integers(image_size // 3, image_size - 8))
        color = palette[int(rng.integers(0, len(palette)))]
        draw.rectangle((left, top, min(image_size - 5, left + width), top + 4), fill=color)
    if index % 3 == 0:
        draw.rectangle((8, image_size - 20, image_size - 8, image_size - 7), outline=(220, 38, 38), width=2)
        draw.text((12, image_size - 19), "ERR", fill=(220, 38, 38), font=ImageFont.load_default())
    arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return arr


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
