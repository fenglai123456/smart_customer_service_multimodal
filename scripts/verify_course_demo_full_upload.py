from __future__ import annotations

import io
import json
from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app


REPORT_PATH = ROOT / "reports" / "course_demo_full_upload_prediction.json"


def main() -> None:
    image = ROOT / "demo_assets" / "customer_screenshots" / "test" / "fault" / "fault_0001.png"
    audio = ROOT / "demo_assets" / "media" / "audio" / "voice_01_public_librispeech.flac"
    emotion_image = ROOT / "demo_assets" / "media" / "emotion_faces" / "angry_fer2013_demo.png"

    with app.test_client() as client:
        with image.open("rb") as img, audio.open("rb") as aud, emotion_image.open("rb") as emo:
            response = client.post(
                "/api/predict",
                data={
                    "text": "页面一直报错，支付失败，帮我报修",
                    "voice_text": "我现在有点着急，麻烦尽快帮我处理",
                    "emotion": "anxious",
                    "image": (io.BytesIO(img.read()), image.name),
                    "audio": (io.BytesIO(aud.read()), audio.name),
                    "emotion_image": (io.BytesIO(emo.read()), emotion_image.name),
                },
                content_type="multipart/form-data",
            )

    payload = response.get_json()
    prediction = payload["prediction"]
    signals = prediction["modality_signals"]
    report = {
        "task": "course demo full multimodal upload verification",
        "status_code": response.status_code,
        "expected_intent": "repair",
        "predicted_intent": prediction["intent"],
        "intent_label": prediction["intent_label"],
        "confidence": prediction["confidence"],
        "need_human": prediction["need_human"],
        "audio": signals["audio"],
        "emotion_image": signals["emotion_image"],
        "image_screenshot": signals["image"]["details"]["screenshot_intent_cnn"],
        "ok": response.status_code == 200 and prediction["intent"] == "repair",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
