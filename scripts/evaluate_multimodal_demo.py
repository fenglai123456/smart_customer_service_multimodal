from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from customer_ai.intent_engine import IntentEngine

DATA_PATH = ROOT / "data" / "demo" / "multimodal_samples.csv"
REPORT_PATH = ROOT / "reports" / "multimodal_demo_metrics.json"


@dataclass
class FakeImageSignal:
    label: str
    confidence: float
    summary: str
    details: dict


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Run scripts/generate_demo_data.py first: {DATA_PATH}")

    engine = IntentEngine()
    df = pd.read_csv(DATA_PATH)
    truth = []
    pred = []
    rows = []

    for row in df.to_dict(orient="records"):
        image = build_image_signal(row["image_signal"])
        result = engine.predict(
            text=str(row["text"]),
            voice_text=str(row["voice_text"]),
            emotion=str(row["emotion"]),
            image=image,
        )
        truth.append(row["label"])
        pred.append(result.intent)
        rows.append(
            {
                "text": row["text"],
                "voice_text": row["voice_text"],
                "emotion": row["emotion"],
                "image_signal": row["image_signal"],
                "label": row["label"],
                "pred": result.intent,
                "confidence": result.confidence,
                "need_human": result.need_human,
            }
        )

    metrics = {
        "model": "Rule-weighted feature-level fusion demo",
        "data": str(DATA_PATH),
        "sample_size": len(rows),
        "accuracy": round(float(accuracy_score(truth, pred)), 4),
        "classification_report": classification_report(truth, pred, output_dict=True, zero_division=0),
        "predictions": rows,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved report: {REPORT_PATH}")


def build_image_signal(name: str):
    if name == "error_screenshot":
        return FakeImageSignal(
            label="疑似故障截图",
            confidence=0.82,
            summary="模拟截图中包含报错/异常线索",
            details={"source": "demo"},
        )
    if name == "low_info":
        return FakeImageSignal(
            label="信息较少图片",
            confidence=0.61,
            summary="模拟图片信息较少，需要追问",
            details={"source": "demo"},
        )
    if name == "normal":
        return FakeImageSignal(
            label="普通客服图片",
            confidence=0.64,
            summary="普通辅助图片",
            details={"source": "demo"},
        )
    return None


if __name__ == "__main__":
    main()
