from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "multimodal" / "customer_cases.csv"
REPORT_PATH = ROOT / "reports" / "multimodal_case_validation.json"
INTENT_LABELS = ["consult", "complaint", "refund", "repair", "howto"]
REQUIRED_COLUMNS = [
    "case_id",
    "label",
    "text",
    "voice_text",
    "audio_path",
    "emotion_label",
    "emotion_image_path",
    "screenshot_path",
    "source_dataset",
    "source_note",
]
FORBIDDEN_MARKERS = ["demo", "synthetic", "template", "generated", "fake", "random_pair"]
ALLOWED_EMOTIONS = {"", "neutral", "happy", "anxious", "angry", "sad", "fear", "disgust", "surprise"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate strict multimodal customer-service case rows.")
    parser.add_argument("--path", default=str(DATA_PATH))
    parser.add_argument("--min_cases_per_label", type=int, default=10)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    path = Path(args.path)
    report = validate(path, args.min_cases_per_label)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not report["ready"]:
        raise SystemExit(1)


def validate(path: Path, min_cases_per_label: int) -> dict:
    if not path.exists():
        return {
            "ready": False,
            "path": str(path),
            "rows": 0,
            "problems": [f"CSV does not exist: {path}"],
        }

    rows = read_rows(path)
    problems: list[str] = []
    if not rows:
        problems.append("CSV has no real multimodal case rows.")

    header = list(rows[0].keys()) if rows else read_header(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing_columns:
        problems.append(f"Missing required columns: {', '.join(missing_columns)}")

    label_counts: Counter[str] = Counter()
    seen_case_ids: set[str] = set()
    for index, row in enumerate(rows, start=2):
        case_id = clean(row.get("case_id"))
        label = clean(row.get("label"))
        label_counts[label] += 1
        row_prefix = case_id or f"line {index}"
        if not case_id:
            problems.append(f"{row_prefix}: case_id is required.")
        elif case_id in seen_case_ids:
            problems.append(f"{row_prefix}: duplicate case_id.")
        seen_case_ids.add(case_id)

        if label not in INTENT_LABELS:
            problems.append(f"{row_prefix}: label must be one of {', '.join(INTENT_LABELS)}.")
        if not (clean(row.get("text")) or clean(row.get("voice_text"))):
            problems.append(f"{row_prefix}: text or voice_text is required.")
        if not existing_path(row.get("audio_path")):
            problems.append(f"{row_prefix}: audio_path must point to an existing real audio file.")
        if not existing_path(row.get("screenshot_path")):
            problems.append(f"{row_prefix}: screenshot_path must point to an existing screenshot.")
        emotion_label = clean(row.get("emotion_label")).lower()
        if emotion_label not in ALLOWED_EMOTIONS:
            problems.append(f"{row_prefix}: emotion_label is not recognized.")
        if not emotion_label and not existing_path(row.get("emotion_image_path")):
            problems.append(f"{row_prefix}: emotion_label or emotion_image_path is required.")

        source_text = f"{clean(row.get('source_dataset'))} {clean(row.get('source_note'))}".lower()
        if not source_text.strip():
            problems.append(f"{row_prefix}: source_dataset/source_note is required.")
        if any(marker in source_text for marker in FORBIDDEN_MARKERS):
            problems.append(f"{row_prefix}: source note looks like demo/synthetic/template data.")

    insufficient = [label for label in INTENT_LABELS if label_counts[label] < min_cases_per_label]
    if insufficient:
        details = ", ".join(f"{label}={label_counts[label]}" for label in INTENT_LABELS)
        problems.append(f"Need at least {min_cases_per_label} real cases per label. Current: {details}.")

    return {
        "ready": not problems,
        "path": str(path),
        "rows": len(rows),
        "label_counts": {label: label_counts[label] for label in INTENT_LABELS},
        "min_cases_per_label": min_cases_per_label,
        "problems": problems[:80],
        "problem_count": len(problems),
        "strict_note": (
            "A valid row must represent one real/traceable customer-service case. "
            "Do not randomly pair unrelated text, audio, emotion images, and screenshots."
        ),
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def existing_path(value) -> Path | None:
    raw = clean(value)
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path if path.exists() else None


def clean(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


if __name__ == "__main__":
    main()
