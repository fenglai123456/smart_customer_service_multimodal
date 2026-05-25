from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from customer_ai.image_analyzer import analyze_image


DATA_DIR = ROOT / "demo_assets" / "customer_screenshots" / "test"
REPORT_PATH = ROOT / "reports" / "demo_screenshot_system_predictions.json"
LABELS = ["fault", "order", "refund", "howto", "normal"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    truth: list[str] = []
    pred: list[str] = []
    rows = []
    for label in LABELS:
        folder = DATA_DIR / label
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            result = analyze_image(path)
            class_name = ""
            confidence = 0.0
            if result:
                screenshot = result.details.get("screenshot_intent_cnn") or {}
                class_name = str(screenshot.get("class_name") or "")
                confidence = float(screenshot.get("confidence") or result.confidence)
            truth.append(label)
            pred.append(class_name)
            rows.append(
                {
                    "path": str(path),
                    "expected": label,
                    "predicted": class_name,
                    "confidence": round(confidence, 4),
                    "ok": label == class_name,
                }
            )
    accuracy = accuracy_score(truth, pred) if truth else 0.0
    report = {
        "task": "system-level demo screenshot prediction verification",
        "samples": len(rows),
        "accuracy": round(float(accuracy), 4),
        "label_counts": dict(sorted(Counter(truth).items())),
        "classification_report": classification_report(truth, pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": LABELS, "matrix": confusion_matrix(truth, pred, labels=LABELS).tolist()},
        "failed": [row for row in rows if not row["ok"]],
        "rows": rows,
        "note": "Uses the same customer_ai.image_analyzer.analyze_image path as the Flask app.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if accuracy < 0.98:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
