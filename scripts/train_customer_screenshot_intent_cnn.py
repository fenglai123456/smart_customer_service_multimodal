from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models


ROOT = Path(__file__).resolve().parents[1]
CUSTOMER_SCREENSHOT_DIR = ROOT / "datasets" / "images" / "customer_screenshots"
MODEL_PATH = ROOT / "models" / "customer_screenshot_intent_cnn.pt"
REPORT_PATH = ROOT / "reports" / "customer_screenshot_intent_metrics.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STRICT_LABELS = ["fault", "order", "refund", "howto", "normal"]


class ScreenshotDataset(Dataset):
    def __init__(
        self,
        items: list[tuple[Path, str, str]],
        label_to_id: dict[str, int],
        image_size: int,
        augment: bool,
        mirror_aug: bool,
        normalize: bool,
    ):
        self.items = items
        self.label_to_id = label_to_id
        self.image_size = image_size
        self.augment = augment
        self.mirror_aug = mirror_aug
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label, _ = self.items[index]
        image = Image.open(path).convert("RGB")
        if self.augment:
            image = augment_image(image, mirror_aug=self.mirror_aug)
        image = image.resize((self.image_size, self.image_size))
        arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        if self.normalize:
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            arr = (arr - mean) / std
        return torch.tensor(arr), torch.tensor(self.label_to_id[label], dtype=torch.long)


class ScreenshotIntentCNN(nn.Module):
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
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a five-class CNN for customer-service screenshot intent.")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--learning_rate", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_group_fraction", type=float, default=0.2)
    parser.add_argument("--min_images_per_label", type=int, default=30)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--architecture",
        choices=["small_cnn", "resnet18_transfer"],
        default="small_cnn",
    )
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument(
        "--class_weight",
        action="store_true",
        help="Use inverse-frequency class weights for the training split.",
    )
    parser.add_argument(
        "--mirror_aug",
        action="store_true",
        help="Allow horizontal mirror augmentation. Disabled by default because it corrupts UI text/screenshots.",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Initialize ResNet18 from scratch instead of ImageNet weights.",
    )
    parser.add_argument(
        "--source_kinds",
        default="all",
        help="Comma-separated source kinds to include: webpage_viewport,article_tile,embedded_asset,seed, or all.",
    )
    parser.add_argument("--min_edge_score", type=float, default=0.0)
    parser.add_argument("--min_contrast", type=float, default=0.0)
    parser.add_argument(
        "--drop_cross_label_duplicates",
        action="store_true",
        help="Drop exact image files whose bytes occur under more than one label.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=MODEL_PATH,
        help="Where to save the trained checkpoint.",
    )
    parser.add_argument(
        "--report_path",
        type=Path,
        default=REPORT_PATH,
        help="Where to save the JSON metrics report.",
    )
    parser.add_argument(
        "--split_strategy",
        choices=["source_group", "random"],
        default="source_group",
        help="Use source_group for strict evaluation; random is diagnostic only.",
    )
    parser.add_argument(
        "--strict_sources",
        action="store_true",
        help="Require the public/teacher-provided customer screenshot subset with all five labels.",
    )
    args = parser.parse_args()
    set_seed(args.seed)

    source_kinds = parse_source_kinds(args.source_kinds)
    items = collect_items(
        source_kinds=source_kinds,
        min_edge_score=args.min_edge_score,
        min_contrast=args.min_contrast,
        drop_cross_label_duplicates=args.drop_cross_label_duplicates,
    )
    label_counts = Counter(label for _, label, _ in items)
    if args.strict_sources:
        missing = [label for label in STRICT_LABELS if label_counts[label] < args.min_images_per_label]
        if missing:
            details = ", ".join(f"{label}={label_counts[label]}" for label in STRICT_LABELS)
            raise SystemExit(
                "Strict screenshot intent training requires at least "
                f"{args.min_images_per_label} images for each label. Current counts: {details}"
            )

    labels = STRICT_LABELS
    label_to_id = {label: index for index, label in enumerate(labels)}
    if args.split_strategy == "random":
        train_items, test_items, split_details = split_random(items, args.test_group_fraction, args.seed)
    else:
        train_items, test_items, split_details = split_by_source_group(items, labels, args.test_group_fraction, args.seed)
    if not train_items or not test_items:
        raise SystemExit("Unable to create a grouped train/test split from the current screenshot dataset.")

    train_loader = DataLoader(
        ScreenshotDataset(
            train_items,
            label_to_id,
            args.image_size,
            augment=True,
            mirror_aug=args.mirror_aug,
            normalize=args.architecture == "resnet18_transfer",
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        ScreenshotDataset(
            test_items,
            label_to_id,
            args.image_size,
            augment=False,
            mirror_aug=False,
            normalize=args.architecture == "resnet18_transfer",
        ),
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=len(labels),
        architecture=args.architecture,
        freeze_backbone=args.freeze_backbone,
        pretrained=not args.no_pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weight_tensor(train_items, labels, label_to_id, device) if args.class_weight else None
    )
    best_state = copy.deepcopy(model.state_dict())
    best_accuracy = -1.0
    best_epoch = 0
    history = []
    epochs_without_improvement = 0
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
        epoch_loss = running_loss / max(1, batches)
        truth, pred = evaluate(model, test_loader, device, labels)
        epoch_accuracy = float(accuracy_score(truth, pred))
        history.append({"epoch": epoch + 1, "loss": round(epoch_loss, 6), "accuracy": round(epoch_accuracy, 6)})
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(f"epoch={epoch + 1} loss={epoch_loss:.4f} val_accuracy={epoch_accuracy:.4f}", flush=True)
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"early_stop epoch={epoch + 1} best_epoch={best_epoch} best_accuracy={best_accuracy:.4f}")
            break

    model.load_state_dict(best_state)
    truth, pred = evaluate(model, test_loader, device, labels)
    metrics = {
        "model": "ScreenshotIntentCNN",
        "task": "five-class customer-service screenshot intent classification",
        "strict_sources": bool(args.strict_sources),
        "source_contract": "customer screenshot public subset; COCO branch is trained separately by scripts/train_image_cnn.py",
        "source_note": (
            "Evaluation is grouped by public page/source id so viewport variants of the same page "
            "do not appear in both train and test."
        ),
        "filters": {
            "source_kinds": sorted(source_kinds) if source_kinds else "all",
            "min_edge_score": args.min_edge_score,
            "min_contrast": args.min_contrast,
            "drop_cross_label_duplicates": bool(args.drop_cross_label_duplicates),
        },
        "split_strategy": args.split_strategy,
        "labels": labels,
        "samples": int(len(items)),
        "label_counts": dict(sorted(label_counts.items())),
        "train_size": int(len(train_items)),
        "test_size": int(len(test_items)),
        "split": split_details,
        "epochs": args.epochs,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_accuracy": round(float(best_accuracy), 4),
        "history": history,
        "image_size": args.image_size,
        "learning_rate": args.learning_rate,
        "class_weight": bool(args.class_weight),
        "mirror_aug": bool(args.mirror_aug),
        "architecture": args.architecture if args.architecture == "resnet18_transfer" else "screenshot_intent_cnn_v1",
        "freeze_backbone": bool(args.freeze_backbone),
        "pretrained": not args.no_pretrained,
        "device": str(device),
        "accuracy": round(float(accuracy_score(truth, pred)), 4),
        "classification_report": classification_report(truth, pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(truth, pred, labels=labels).tolist(),
        },
    }

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_id": label_to_id,
            "image_size": args.image_size,
            "architecture": args.architecture if args.architecture == "resnet18_transfer" else "screenshot_intent_cnn_v1",
            "freeze_backbone": bool(args.freeze_backbone),
            "pretrained": not args.no_pretrained,
            "metrics": metrics,
        },
        args.model_path,
    )
    args.report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(num_classes: int, architecture: str, freeze_backbone: bool, pretrained: bool) -> nn.Module:
    if architecture == "resnet18_transfer":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    return ScreenshotIntentCNN(num_classes=num_classes)


def collect_items(
    source_kinds: set[str] | None = None,
    min_edge_score: float = 0.0,
    min_contrast: float = 0.0,
    drop_cross_label_duplicates: bool = False,
) -> list[tuple[Path, str, str]]:
    items: list[tuple[Path, str, str]] = []
    for label in STRICT_LABELS:
        folder = CUSTOMER_SCREENSHOT_DIR / label
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if source_kinds and infer_kind(path) not in source_kinds:
                continue
            if min_edge_score > 0.0 or min_contrast > 0.0:
                edge_score, contrast = image_quality(path)
                if edge_score < min_edge_score or contrast < min_contrast:
                    continue
            items.append((path, label, source_key(path)))
    if drop_cross_label_duplicates:
        items = drop_exact_cross_label_duplicates(items)
    return items


def drop_exact_cross_label_duplicates(items: list[tuple[Path, str, str]]) -> list[tuple[Path, str, str]]:
    by_digest: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
    for item in items:
        path, _, _ = item
        by_digest[file_digest(path)].append(item)
    bad_digests = {digest for digest, group in by_digest.items() if len({label for _, label, _ in group}) > 1}
    return [item for item in items if file_digest(item[0]) not in bad_digests]


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def class_weight_tensor(
    train_items: list[tuple[Path, str, str]],
    labels: list[str],
    label_to_id: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    counts = Counter(label for _, label, _ in train_items)
    total = sum(counts.values())
    weights = torch.ones(len(labels), dtype=torch.float32, device=device)
    for label in labels:
        label_count = max(1, counts[label])
        weights[label_to_id[label]] = total / (len(labels) * label_count)
    return weights


def parse_source_kinds(value: str) -> set[str] | None:
    if not value or value.strip().lower() == "all":
        return None
    allowed = {"webpage_viewport", "article_tile", "embedded_asset", "seed"}
    kinds = {item.strip() for item in value.split(",") if item.strip()}
    unknown = kinds - allowed
    if unknown:
        raise SystemExit(f"Unknown source_kinds: {', '.join(sorted(unknown))}")
    return kinds


def infer_kind(path: Path) -> str:
    stem = path.stem
    if stem.startswith("public_web_"):
        return "webpage_viewport"
    if stem.startswith("public_tile_"):
        return "article_tile"
    if stem.startswith("public_asset_"):
        return "embedded_asset"
    return "seed"


def image_quality(path: Path) -> tuple[float, float]:
    with Image.open(path) as image:
        gray = image.convert("L")
        arr = np.asarray(gray.resize((128, 128)), dtype=np.float32)
        edge_score = float(np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean())
        contrast = float(np.asarray(gray, dtype=np.float32).std())
    return edge_score, contrast


def source_key(path: Path) -> str:
    stem = path.stem
    match = re.match(r"^public_(?:web|asset|tile)_(\d+)", stem)
    if match:
        return f"public_source_{match.group(1)}"
    return stem


def split_by_source_group(
    items: list[tuple[Path, str, str]],
    labels: list[str],
    test_fraction: float,
    seed: int,
) -> tuple[list[tuple[Path, str, str]], list[tuple[Path, str, str]], dict]:
    rng = random.Random(seed)
    grouped: dict[str, dict[str, list[tuple[Path, str, str]]]] = {label: defaultdict(list) for label in labels}
    for item in items:
        _, label, group = item
        grouped[label][group].append(item)

    train_items: list[tuple[Path, str, str]] = []
    test_items: list[tuple[Path, str, str]] = []
    details = {"strategy": "per-label source-group holdout", "labels": {}}
    for label in labels:
        group_names = sorted(grouped[label])
        rng.shuffle(group_names)
        if len(group_names) < 2:
            raise SystemExit(f"Label {label!r} needs at least two source groups for grouped evaluation.")
        test_count = max(1, round(len(group_names) * test_fraction))
        test_count = min(test_count, len(group_names) - 1)
        test_groups = set(group_names[:test_count])
        train_groups = set(group_names[test_count:])
        for group in train_groups:
            train_items.extend(grouped[label][group])
        for group in test_groups:
            test_items.extend(grouped[label][group])
        details["labels"][label] = {
            "train_groups": sorted(train_groups),
            "test_groups": sorted(test_groups),
            "train_images": sum(len(grouped[label][group]) for group in train_groups),
            "test_images": sum(len(grouped[label][group]) for group in test_groups),
        }
    return train_items, test_items, details


def split_random(
    items: list[tuple[Path, str, str]],
    test_fraction: float,
    seed: int,
) -> tuple[list[tuple[Path, str, str]], list[tuple[Path, str, str]], dict]:
    rng = random.Random(seed)
    train_items: list[tuple[Path, str, str]] = []
    test_items: list[tuple[Path, str, str]] = []
    details = {"strategy": "random per-label holdout; diagnostic only", "labels": {}}
    by_label: dict[str, list[tuple[Path, str, str]]] = {label: [] for label in STRICT_LABELS}
    for item in items:
        by_label[item[1]].append(item)
    for label, label_items in by_label.items():
        shuffled = list(label_items)
        rng.shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * test_fraction))
        test_items.extend(shuffled[:test_count])
        train_items.extend(shuffled[test_count:])
        details["labels"][label] = {
            "train_images": len(shuffled) - test_count,
            "test_images": test_count,
        }
    return train_items, test_items, details


def augment_image(image: Image.Image, mirror_aug: bool = False) -> Image.Image:
    if mirror_aug and random.random() < 0.35:
        image = ImageOps.mirror(image)
    width, height = image.size
    crop_ratio = random.uniform(0.90, 1.0)
    crop_w = max(1, int(width * crop_ratio))
    crop_h = max(1, int(height * crop_ratio))
    if crop_w < width or crop_h < height:
        left = random.randint(0, width - crop_w)
        top = random.randint(0, height - crop_h)
        image = image.crop((left, top, left + crop_w, top + crop_h))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.92, 1.08))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.92, 1.10))
    return image


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, labels: list[str]) -> tuple[list[str], list[str]]:
    model.eval()
    truth: list[str] = []
    pred: list[str] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device)).cpu()
            truth.extend(labels[int(index)] for index in batch_y.numpy())
            pred.extend(labels[int(index)] for index in logits.argmax(dim=1).numpy())
    return truth, pred


if __name__ == "__main__":
    main()
