from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "datasets" / "multimodal"
OUT_CSV = OUT_DIR / "customer_cases.csv"
REPORT = ROOT / "reports" / "multimodal_case_template_report.json"

FIELDS = [
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not OUT_CSV.exists():
        with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
    report = {
        "path": str(OUT_CSV),
        "fields": FIELDS,
        "strict_note": (
            "Rows must be real multimodal cases assembled from screenshot-allowed sources. "
            "Do not fill this with demo/template samples for strict training."
        ),
        "minimum_recommendation": "At least dozens per intent for a classroom demo; preferably 100+ per intent for a credible 82%+ claim.",
        "allowed_labels": ["consult", "complaint", "refund", "repair", "howto"],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
