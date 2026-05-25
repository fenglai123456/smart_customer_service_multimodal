from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "strict_dataset_audit.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REQUIRED_FILES = {
    "SQuAD v2.0 train": ROOT / "datasets" / "text" / "squad" / "train-v2.0.json",
    "SQuAD v2.0 dev": ROOT / "datasets" / "text" / "squad" / "dev-v2.0.json",
    "ABCD customer-dialogue source": ROOT
    / "datasets"
    / "text"
    / "customer_dialogue_public"
    / "abcd"
    / "data"
    / "abcd_v1.1.json.gz",
    "ABCD mapped five-intent CSV": ROOT
    / "datasets"
    / "text"
    / "customer_dialogue_public"
    / "abcd_customer_intents.csv",
    "Strict SQuAD+ABCD text CSV": ROOT / "datasets" / "text" / "strict_text_intents.csv",
    "LibriSpeech dev-clean archive": ROOT / "datasets" / "audio" / "librispeech" / "dev-clean.tar.gz",
    "LibriSpeech train-clean-100 archive": ROOT / "datasets" / "audio" / "librispeech" / "train-clean-100.tar.gz",
    "LibriSpeech dev-clean extracted": ROOT
    / "datasets"
    / "audio"
    / "librispeech"
    / "LibriSpeech"
    / "dev-clean",
    "LibriSpeech train-clean-100 extracted": ROOT
    / "datasets"
    / "audio"
    / "librispeech"
    / "LibriSpeech"
    / "train-clean-100",
    "FER-2013 CSV": ROOT / "datasets" / "emotion" / "fer2013" / "fer2013.csv",
    "COCO val2017 archive": ROOT / "datasets" / "images" / "coco" / "val2017.zip",
    "COCO annotations archive": ROOT / "datasets" / "images" / "coco" / "annotations_trainval2017.zip",
    "COCO val2017 extracted": ROOT / "datasets" / "images" / "coco" / "val2017",
    "COCO instances_val2017": ROOT / "datasets" / "images" / "coco" / "annotations" / "instances_val2017.json",
}

CUSTOMER_SCREENSHOT_DIR = ROOT / "datasets" / "images" / "customer_screenshots"
CUSTOMER_SCREENSHOT_LABELS = ["fault", "order", "refund", "howto", "normal"]
MIN_SCREENSHOTS_PER_LABEL = 30
MIN_MULTIMODAL_CASES_PER_LABEL = 10


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly audit datasets against the screenshot technical route.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any strict requirement is missing.")
    args = parser.parse_args()

    audit = build_audit()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print_human_report(audit)

    if args.strict and not audit["strict_ready"]:
        raise SystemExit(1)


def build_audit() -> dict:
    file_checks = {
        name: {"path": str(path), "exists": path.exists()}
        for name, path in REQUIRED_FILES.items()
    }
    strict_text_csv = REQUIRED_FILES["Strict SQuAD+ABCD text CSV"]
    text_public = (
        inspect_text_csv(strict_text_csv) if strict_text_csv.exists() else {"exists": False, "rows": 0, "label_counts": {}}
    )
    screenshots = inspect_customer_screenshots()
    multimodal = inspect_multimodal_cases()
    modules = {
        "text": {
            "required_sources": ["SQuAD v2.0", "ABCD public customer-dialogue subset"],
            "ready": all(
                file_checks[name]["exists"]
                for name in ["SQuAD v2.0 train", "SQuAD v2.0 dev", "ABCD mapped five-intent CSV", "Strict SQuAD+ABCD text CSV"]
            )
            and set(text_public.get("label_counts", {})) == {"consult", "complaint", "refund", "repair", "howto"}
            and text_public.get("contains_required_sources", False),
            "details": text_public,
        },
        "audio": {
            "required_sources": ["LibriSpeech"],
            "ready": all(
                file_checks[name]["exists"]
                for name in ["LibriSpeech dev-clean extracted", "LibriSpeech train-clean-100 extracted"]
            ),
            "details": "LibriSpeech does not contain customer-service intent labels; it is valid only for the requested LSTM speech branch.",
        },
        "emotion": {
            "required_sources": ["FER-2013"],
            "ready": file_checks["FER-2013 CSV"]["exists"],
            "details": file_checks["FER-2013 CSV"],
        },
        "image": {
            "required_sources": ["COCO", "customer screenshot public subset"],
            "ready": all(
                file_checks[name]["exists"]
                for name in ["COCO val2017 extracted", "COCO instances_val2017"]
            )
            and screenshots["ready"],
            "details": screenshots,
        },
        "fusion": {
            "required_sources": ["real multimodal cases assembled from the strict sources above"],
            "ready": multimodal["ready"],
            "details": multimodal,
        },
    }
    strict_ready = all(item["ready"] for item in modules.values())
    return {
        "contract": {
            "text": "SQuAD + customer-dialogue public subset",
            "audio": "LibriSpeech",
            "emotion": "FER-2013",
            "image": "COCO + customer screenshot public subset",
            "fusion": "feature-level fusion using real multimodal cases",
        },
        "allowed_public_sources": [
            "SQuAD v2.0",
            "ABCD public customer-dialogue subset",
            "LibriSpeech",
            "FER-2013",
            "COCO 2017",
            "customer_screenshots directory only when it contains real/public/teacher-provided customer-service screenshots",
        ],
        "disallowed_for_strict_training": [
            "template-generated text",
            "synthetic customer screenshots",
            "generic UI screenshot datasets without customer-service labels",
            "demo multimodal samples",
            "COCO-only image classifier as a replacement for customer screenshots",
        ],
        "files": file_checks,
        "modules": modules,
        "strict_ready": strict_ready,
        "report_path": str(REPORT_PATH),
    }


def inspect_text_csv(path: Path) -> dict:
    counts: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            counts[str(row.get("label", ""))] += 1
            sources[str(row.get("source_dataset", ""))] += 1
    return {
        "exists": True,
        "path": str(path),
        "rows": rows,
        "label_counts": dict(sorted(counts.items())),
        "source_counts": dict(sorted(sources.items())),
        "contains_required_sources": all(source in sources for source in ["ABCD", "SQuAD"]),
    }


def inspect_customer_screenshots() -> dict:
    label_counts = {}
    for label in CUSTOMER_SCREENSHOT_LABELS:
        folder = CUSTOMER_SCREENSHOT_DIR / label
        label_counts[label] = count_images(folder)
    total = sum(label_counts.values())
    missing_labels = [label for label, count in label_counts.items() if count < MIN_SCREENSHOTS_PER_LABEL]
    return {
        "path": str(CUSTOMER_SCREENSHOT_DIR),
        "required_labels": CUSTOMER_SCREENSHOT_LABELS,
        "min_images_per_label": MIN_SCREENSHOTS_PER_LABEL,
        "label_counts": label_counts,
        "total_images": total,
        "ready": total > 0 and not missing_labels,
        "missing_labels": missing_labels,
        "note": "Only real/public/teacher-provided customer-service screenshots belong here.",
    }


def inspect_multimodal_cases() -> dict:
    from validate_multimodal_cases import validate

    path = ROOT / "datasets" / "multimodal" / "customer_cases.csv"
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "ready": False,
            "min_cases_per_label": MIN_MULTIMODAL_CASES_PER_LABEL,
            "note": "The current demo CSV is not strict training data.",
        }
    validation = validate(path, MIN_MULTIMODAL_CASES_PER_LABEL)
    return validation | {
        "path": str(path),
        "exists": True,
        "note": "Rows must be real multimodal cases assembled from strict sources; demo rows do not count.",
    }


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def print_human_report(audit: dict) -> None:
    print("Strict dataset audit")
    print("=" * 22)
    for module, info in audit["modules"].items():
        status = "OK" if info["ready"] else "MISSING"
        print(f"[{status}] {module}: {', '.join(info['required_sources'])}")
    print()
    print(f"strict_ready={audit['strict_ready']}")
    print(f"json_report={audit['report_path']}")


if __name__ == "__main__":
    main()
