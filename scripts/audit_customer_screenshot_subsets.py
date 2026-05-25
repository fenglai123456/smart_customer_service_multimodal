from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "images" / "customer_screenshots"
REPORT_PATH = ROOT / "reports" / "customer_screenshot_subset_audit.json"
LABELS = ["fault", "order", "refund", "howto", "normal"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    rows = []
    for label in LABELS:
        for path in sorted((DATA_DIR / label).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rows.append(inspect(path, label))

    by_label = Counter(row["label"] for row in rows)
    by_kind = Counter(row["kind"] for row in rows)
    by_label_kind: dict[str, dict[str, int]] = defaultdict(dict)
    for label in LABELS:
        counts = Counter(row["kind"] for row in rows if row["label"] == label)
        by_label_kind[label] = dict(sorted(counts.items()))

    high_detail = [row for row in rows if row["edge_score"] >= 12.0 and row["contrast"] >= 25.0]
    report = {
        "samples": len(rows),
        "label_counts": dict(sorted(by_label.items())),
        "kind_counts": dict(sorted(by_kind.items())),
        "label_kind_counts": by_label_kind,
        "high_detail_counts": dict(sorted(Counter(row["label"] for row in high_detail).items())),
        "rows": rows,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "rows"}, ensure_ascii=False, indent=2))


def inspect(path: Path, label: str) -> dict:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        gray = rgb.convert("L")
        arr = np.asarray(gray.resize((128, 128)), dtype=np.float32)
        edge_score = float(np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean())
        contrast = float(ImageStat.Stat(gray).stddev[0])
    return {
        "path": str(path),
        "label": label,
        "kind": infer_kind(path),
        "source_group": source_group(path),
        "width": int(rgb.width),
        "height": int(rgb.height),
        "edge_score": round(edge_score, 4),
        "contrast": round(contrast, 4),
        "aspect_ratio": round(rgb.width / max(1, rgb.height), 4),
    }


def infer_kind(path: Path) -> str:
    stem = path.stem
    if stem.startswith("public_web_"):
        return "webpage_viewport"
    if stem.startswith("public_tile_"):
        return "article_tile"
    if stem.startswith("public_asset_"):
        return "embedded_asset"
    return "seed"


def source_group(path: Path) -> str:
    stem = path.stem
    match = re.match(r"^public_(?:web|asset|tile)_(\d+)", stem)
    if match:
        return f"public_source_{match.group(1)}"
    return stem


if __name__ == "__main__":
    main()
