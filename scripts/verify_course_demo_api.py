from __future__ import annotations

import io
import json
from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app


REPORT_PATH = ROOT / "reports" / "course_demo_api_predictions.json"

CASES = [
    {
        "scenario": "故障报修",
        "image": ROOT / "demo_assets" / "customer_screenshots" / "test" / "fault" / "fault_0001.png",
        "text": "页面一直报错，支付失败，帮我报修",
        "emotion": "anxious",
        "expected": "repair",
    },
    {
        "scenario": "订单咨询",
        "image": ROOT / "demo_assets" / "customer_screenshots" / "test" / "order" / "order_0001.png",
        "text": "我想查订单物流和配送进度",
        "emotion": "neutral",
        "expected": "consult",
    },
    {
        "scenario": "退款售后",
        "image": ROOT / "demo_assets" / "customer_screenshots" / "test" / "refund" / "refund_0001.png",
        "text": "我要申请退款，想知道钱什么时候退回",
        "emotion": "neutral",
        "expected": "refund",
    },
    {
        "scenario": "操作指引",
        "image": ROOT / "demo_assets" / "customer_screenshots" / "test" / "howto" / "howto_0001.png",
        "text": "我不会操作，帮我一步一步说明",
        "emotion": "neutral",
        "expected": "howto",
    },
    {
        "scenario": "投诉",
        "image": ROOT / "demo_assets" / "customer_screenshots" / "test" / "normal" / "normal_0001.png",
        "text": "客服一直不回复，服务态度太差了，我要投诉",
        "emotion": "angry",
        "expected": "complaint",
    },
]


def main() -> None:
    rows = []
    with app.test_client() as client:
        for case in CASES:
            image_path = case["image"]
            with image_path.open("rb") as reader:
                response = client.post(
                    "/api/predict",
                    data={
                        "text": case["text"],
                        "voice_text": "",
                        "emotion": case["emotion"],
                        "image": (io.BytesIO(reader.read()), image_path.name),
                    },
                    content_type="multipart/form-data",
                )
            payload = response.get_json()
            prediction = payload["prediction"]
            image_signal = prediction["modality_signals"]["image"]
            screenshot_cnn = image_signal["details"].get("screenshot_intent_cnn") if image_signal else None
            rows.append(
                {
                    "scenario": case["scenario"],
                    "image": str(image_path),
                    "text": case["text"],
                    "expected": case["expected"],
                    "predicted": prediction["intent"],
                    "intent_label": prediction["intent_label"],
                    "confidence": prediction["confidence"],
                    "need_human": prediction["need_human"],
                    "screenshot_cnn": screenshot_cnn,
                    "ok": prediction["intent"] == case["expected"],
                }
            )

    passed = sum(1 for row in rows if row["ok"])
    report = {
        "task": "course demo Flask API verification",
        "samples": len(rows),
        "passed": passed,
        "accuracy": round(passed / len(rows), 4) if rows else 0.0,
        "rows": rows,
        "note": "Uploads the recommended demo screenshots to Flask /api/predict and checks final fused intent.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if passed != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
