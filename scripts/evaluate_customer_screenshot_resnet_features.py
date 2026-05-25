from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torchvision import models

from train_customer_screenshot_intent_cnn import (
    IMAGE_EXTENSIONS,
    STRICT_LABELS,
    source_key,
    split_by_source_group,
)


ROOT = Path(__file__).resolve().parents[1]
CUSTOMER_SCREENSHOT_DIR = ROOT / "datasets" / "images" / "customer_screenshots"
REPORT_PATH = ROOT / "reports" / "customer_screenshot_resnet_feature_metrics.json"
FEATURE_PATH = ROOT / "models" / "customer_screenshot_resnet_features.npz"


def main() -> None:
    items = collect_items()
    train_items, test_items, split_details = split_by_source_group(items, STRICT_LABELS, 0.2, 42)
    label_to_id = {label: index for index, label in enumerate(STRICT_LABELS)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = build_extractor().to(device).eval()
    train_x = extract_features(extractor, train_items, device)
    test_x = extract_features(extractor, test_items, device)
    train_y = np.asarray([label_to_id[label] for _, label, _ in train_items], dtype=np.int64)
    test_y = np.asarray([label_to_id[label] for _, label, _ in test_items], dtype=np.int64)

    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=42)
    classifier.fit(train_x_scaled, train_y)
    pred_y = classifier.predict(test_x_scaled)

    truth = [STRICT_LABELS[index] for index in test_y]
    pred = [STRICT_LABELS[index] for index in pred_y]
    label_counts = Counter(label for _, label, _ in items)
    metrics = {
        "model": "ImageNet ResNet18 frozen features + LogisticRegression",
        "task": "five-class customer-service screenshot intent classification",
        "strict_sources": True,
        "source_contract": "customer screenshot public subset; grouped by original public source page id",
        "labels": STRICT_LABELS,
        "samples": int(len(items)),
        "label_counts": dict(sorted(label_counts.items())),
        "train_size": int(len(train_items)),
        "test_size": int(len(test_items)),
        "split": split_details,
        "device": str(device),
        "accuracy": round(float(accuracy_score(truth, pred)), 4),
        "classification_report": classification_report(truth, pred, labels=STRICT_LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": {
            "labels": STRICT_LABELS,
            "matrix": confusion_matrix(truth, pred, labels=STRICT_LABELS).tolist(),
        },
        "note": (
            "This fast upper-bound check uses frozen public ImageNet features and does not replace the CNN route. "
            "It helps decide whether more training can solve the current data problem."
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FEATURE_PATH,
        train_x=train_x,
        test_x=test_x,
        train_y=train_y,
        test_y=test_y,
        labels=np.asarray(STRICT_LABELS),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def collect_items() -> list[tuple[Path, str, str]]:
    items: list[tuple[Path, str, str]] = []
    for label in STRICT_LABELS:
        folder = CUSTOMER_SCREENSHOT_DIR / label
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((path, label, source_key(path)))
    return items


def build_extractor() -> torch.nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    return model


def extract_features(
    extractor: torch.nn.Module,
    items: list[tuple[Path, str, str]],
    device: torch.device,
    batch_size: int = 48,
) -> np.ndarray:
    features: list[np.ndarray] = []
    batch: list[torch.Tensor] = []
    with torch.no_grad():
        for path, _, _ in items:
            batch.append(preprocess(path))
            if len(batch) >= batch_size:
                features.append(extractor(torch.stack(batch).to(device)).cpu().numpy())
                batch = []
        if batch:
            features.append(extractor(torch.stack(batch).to(device)).cpu().numpy())
    return np.vstack(features).astype(np.float32)


def preprocess(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    arr = (arr - mean) / std
    return torch.tensor(arr, dtype=torch.float32)


if __name__ == "__main__":
    main()
