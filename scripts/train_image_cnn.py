from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models


ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "datasets" / "images" / "coco"
ANN_PATH = COCO_DIR / "annotations" / "instances_val2017.json"
IMAGE_DIR = COCO_DIR / "val2017"
MODEL_PATH = ROOT / "models" / "coco_cnn.pt"
REPORT_PATH = ROOT / "reports" / "image_cnn_metrics.json"


CocoItem = dict[str, object]


class CocoDataset(Dataset):
    def __init__(self, items, label_to_id, image_size: int, normalize: bool):
        self.items = items
        self.label_to_id = label_to_id
        self.image_size = image_size
        self.normalize = normalize

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        path = Path(item["path"])
        label = str(item["label"])
        image = Image.open(path).convert("RGB")
        bbox = item.get("bbox")
        if bbox:
            image = crop_coco_bbox(image, bbox)
        image = image.resize((self.image_size, self.image_size))
        arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        if self.normalize:
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            arr = (arr - mean) / std
        return torch.tensor(arr), torch.tensor(self.label_to_id[label], dtype=torch.long)


class ImageCNN(nn.Module):
    def __init__(self, num_classes: int, image_size: int):
        super().__init__()
        pooled = image_size // 4
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * pooled * pooled, 256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small CNN on COCO val2017 top categories.")
    parser.add_argument("--max_samples", type=int, default=400)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=8e-4)
    parser.add_argument(
        "--training_unit",
        choices=["full_image", "object_crop"],
        default="full_image",
        help="Train on whole COCO images or on official COCO object bounding-box crops.",
    )
    parser.add_argument(
        "--min_bbox_area",
        type=float,
        default=1024.0,
        help="Minimum annotation area used when --training_unit object_crop is selected.",
    )
    parser.add_argument(
        "--architecture",
        choices=["small_cnn", "resnet18_transfer"],
        default="small_cnn",
        help="Use a local CNN or an ImageNet-pretrained ResNet18 transfer model.",
    )
    parser.add_argument("--freeze_backbone", action="store_true", help="Train only the ResNet classifier head.")
    args = parser.parse_args()

    items = build_items(args.top_k, args.max_samples, args.training_unit, args.min_bbox_area)
    labels = sorted({str(item["label"]) for item in items})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    train_items, test_items, split_details = split_items(
        items,
        test_size=0.2,
        seed=42,
        group_by_image=args.training_unit == "object_crop",
    )
    train_loader = DataLoader(
        CocoDataset(train_items, label_to_id, args.image_size, normalize=args.architecture == "resnet18_transfer"),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        CocoDataset(test_items, label_to_id, args.image_size, normalize=args.architecture == "resnet18_transfer"),
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(labels), image_size=args.image_size, architecture=args.architecture, freeze_backbone=args.freeze_backbone).to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
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

    truth, pred = evaluate(model, test_loader, device, labels)
    metrics = {
        "model": "ImageNet-pretrained ResNet18 transfer" if args.architecture == "resnet18_transfer" else "enhanced 4-block CNN",
        "task": "COCO top-category image classification",
        "samples": len(items),
        "top_k": args.top_k,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "training_unit": args.training_unit,
        "min_bbox_area": args.min_bbox_area if args.training_unit == "object_crop" else None,
        "architecture": args.architecture if args.architecture == "resnet18_transfer" else "enhanced_cnn_v2",
        "freeze_backbone": bool(args.freeze_backbone),
        "image_size": args.image_size,
        "split": split_details,
        "accuracy": round(float(accuracy_score(truth, pred)), 4),
        "classification_report": classification_report(truth, pred, output_dict=True, zero_division=0),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_id": label_to_id,
            "architecture": args.architecture if args.architecture == "resnet18_transfer" else "enhanced_cnn_v2",
            "freeze_backbone": bool(args.freeze_backbone),
            "image_size": args.image_size,
            "training_unit": args.training_unit,
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    REPORT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_model(num_classes: int, image_size: int, architecture: str, freeze_backbone: bool) -> nn.Module:
    if architecture == "resnet18_transfer":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    return ImageCNN(num_classes=num_classes, image_size=image_size)


def build_items(top_k: int, max_samples: int, training_unit: str, min_bbox_area: float) -> list[CocoItem]:
    data = json.loads(ANN_PATH.read_text(encoding="utf-8"))
    category_names = {item["id"]: item["name"] for item in data["categories"]}
    image_names = {item["id"]: item["file_name"] for item in data["images"]}
    counts = Counter(ann["category_id"] for ann in data["annotations"])
    top_categories = [category for category, _ in counts.most_common(top_k)]
    if training_unit == "object_crop":
        return build_object_crop_items(data, category_names, image_names, top_categories, top_k, max_samples, min_bbox_area)
    return build_full_image_items(data, category_names, image_names, top_categories, top_k, max_samples)


def build_full_image_items(
    data: dict,
    category_names: dict[int, str],
    image_names: dict[int, str],
    top_categories: list[int],
    top_k: int,
    max_samples: int,
) -> list[CocoItem]:
    by_category = defaultdict(list)
    seen_images = set()
    for ann in data["annotations"]:
        category = ann["category_id"]
        image_id = ann["image_id"]
        if category not in top_categories or image_id in seen_images:
            continue
        path = IMAGE_DIR / image_names[image_id]
        if path.exists():
            by_category[category_names[category]].append(
                {
                    "path": path,
                    "label": category_names[category],
                    "bbox": None,
                    "image_id": image_id,
                    "group": image_id,
                }
            )
            seen_images.add(image_id)

    per_class = max(1, max_samples // top_k)
    items = []
    for label in sorted(by_category):
        items.extend(by_category[label][:per_class])
    return items


def build_object_crop_items(
    data: dict,
    category_names: dict[int, str],
    image_names: dict[int, str],
    top_categories: list[int],
    top_k: int,
    max_samples: int,
    min_bbox_area: float,
) -> list[CocoItem]:
    by_category = defaultdict(list)
    for ann in data["annotations"]:
        category = ann["category_id"]
        image_id = ann["image_id"]
        if category not in top_categories:
            continue
        if float(ann.get("area", 0.0)) < min_bbox_area:
            continue
        path = IMAGE_DIR / image_names[image_id]
        bbox = ann.get("bbox")
        if path.exists() and bbox and bbox[2] >= 8 and bbox[3] >= 8:
            by_category[category_names[category]].append(
                {
                    "path": path,
                    "label": category_names[category],
                    "bbox": [float(value) for value in bbox],
                    "image_id": image_id,
                    "group": image_id,
                    "annotation_id": ann.get("id"),
                }
            )

    per_class = max(1, max_samples // top_k)
    items: list[CocoItem] = []
    for label in sorted(by_category):
        category_items = sorted(
            by_category[label],
            key=lambda item: (str(item["path"]), int(item.get("annotation_id") or 0)),
        )
        items.extend(category_items[:per_class])
    if len({str(item["label"]) for item in items}) < top_k:
        raise SystemExit("Not enough COCO object-crop samples after filtering. Lower --min_bbox_area or --top_k.")
    return items


def split_items(
    items: list[CocoItem],
    test_size: float,
    seed: int,
    group_by_image: bool,
) -> tuple[list[CocoItem], list[CocoItem], dict]:
    labels = [str(item["label"]) for item in items]
    if not group_by_image:
        train_items, test_items = train_test_split(
            items,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
        return train_items, test_items, {
            "strategy": "stratified random split",
            "group_by_image": False,
            "train_label_counts": dict(sorted(Counter(str(item["label"]) for item in train_items).items())),
            "test_label_counts": dict(sorted(Counter(str(item["label"]) for item in test_items).items())),
        }

    label_set = set(labels)
    rng = np.random.default_rng(seed)
    by_label_group: dict[str, dict[object, list[CocoItem]]] = {label: defaultdict(list) for label in label_set}
    for item in items:
        by_label_group[str(item["label"])][item["group"]].append(item)

    train_items: list[CocoItem] = []
    test_items: list[CocoItem] = []
    details = {"strategy": "per-label image-group holdout", "group_by_image": True, "labels": {}}
    for label in sorted(label_set):
        group_names = np.asarray(sorted(by_label_group[label], key=str), dtype=object)
        rng.shuffle(group_names)
        if len(group_names) < 2:
            raise SystemExit(f"COCO label {label!r} needs at least two image groups for grouped evaluation.")
        target_test = max(1, round(sum(len(by_label_group[label][group]) for group in group_names) * test_size))
        test_groups = []
        test_images = 0
        for group in group_names:
            if len(test_groups) and test_images >= target_test:
                break
            test_groups.append(group)
            test_images += len(by_label_group[label][group])
        if len(test_groups) >= len(group_names):
            test_groups = test_groups[:-1]
        test_group_set = set(test_groups)
        train_group_set = {group for group in group_names if group not in test_group_set}
        for group in train_group_set:
            train_items.extend(by_label_group[label][group])
        for group in test_group_set:
            test_items.extend(by_label_group[label][group])
        details["labels"][label] = {
            "train_groups": len(train_group_set),
            "test_groups": len(test_group_set),
            "train_images": sum(len(by_label_group[label][group]) for group in train_group_set),
            "test_images": sum(len(by_label_group[label][group]) for group in test_group_set),
        }

    return train_items, test_items, {
        **details,
        "train_label_counts": dict(sorted(Counter(str(item["label"]) for item in train_items).items())),
        "test_label_counts": dict(sorted(Counter(str(item["label"]) for item in test_items).items())),
    }


def crop_coco_bbox(image: Image.Image, bbox: object) -> Image.Image:
    x, y, width, height = [float(value) for value in bbox]
    left = max(0, int(np.floor(x)))
    top = max(0, int(np.floor(y)))
    right = min(image.width, int(np.ceil(x + width)))
    bottom = min(image.height, int(np.ceil(y + height)))
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def evaluate(model, loader, device, labels):
    model.eval()
    truth, pred = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)).cpu()
            truth.extend(labels[int(item)] for item in y.numpy())
            pred.extend(labels[int(item)] for item in logits.argmax(dim=1).numpy())
    return truth, pred


if __name__ == "__main__":
    main()
