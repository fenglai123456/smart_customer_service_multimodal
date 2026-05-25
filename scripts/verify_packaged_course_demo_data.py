from __future__ import annotations

import io
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402


DATA_DIR = ROOT / "course_demo_test_data"
REPORT_PATH = ROOT / "reports" / "packaged_course_demo_data_predictions.json"


def main() -> None:
    rows = []
    with app.test_client() as client:
        for case_file in sorted(DATA_DIR.glob("*/case.json")):
            case_dir = case_file.parent
            case = json.loads(case_file.read_text(encoding="utf-8"))
            screenshot = case_dir / case["screenshot_file"]
            emotion_image = case_dir / case["emotion_image_file"]
            audio = case_dir / case["audio_file"]

            with screenshot.open("rb") as screenshot_reader, emotion_image.open("rb") as emotion_reader, audio.open("rb") as audio_reader:
                response = client.post(
                    "/api/predict",
                    data={
                        "text": case["text"],
                        "voice_text": case["voice_text"],
                        "emotion": case["emotion"],
                        "image": (io.BytesIO(screenshot_reader.read()), screenshot.name),
                        "emotion_image": (io.BytesIO(emotion_reader.read()), emotion_image.name),
                        "audio": (io.BytesIO(audio_reader.read()), audio.name),
                    },
                    content_type="multipart/form-data",
                )

            payload = response.get_json()
            prediction = payload["prediction"]
            image_signal = prediction["modality_signals"]["image"] or {}
            screenshot_decision = (image_signal.get("details") or {}).get("screenshot_demo_decision") or {}
            rows.append(
                {
                    "case_id": case["case_id"],
                    "scenario": case["scenario"],
                    "expected_intent": case["expected_intent"],
                    "predicted_intent": prediction["intent"],
                    "expected_screenshot_class": case["expected_screenshot_class"],
                    "predicted_screenshot_class": screenshot_decision.get("class_name"),
                    "confidence": prediction["confidence"],
                    "ok": prediction["intent"] == case["expected_intent"],
                }
            )

    passed = sum(1 for row in rows if row["ok"])
    report = {
        "task": "packaged course demo test data verification",
        "data_dir": str(DATA_DIR),
        "samples": len(rows),
        "passed": passed,
        "accuracy": round(passed / len(rows), 4) if rows else 0.0,
        "rows": rows,
        "note": "Uploads each packaged course_demo_test_data case to Flask /api/predict.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if passed != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
