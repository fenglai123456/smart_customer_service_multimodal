from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402


CASES = [
    ("咨询", "我想咨询一下订单物流什么时候更新", "neutral"),
    ("投诉", "客服一直不回复，服务态度太差了，我要投诉", "angry"),
    ("退款", "订单取消了，我想申请退款，钱什么时候退回", "neutral"),
    ("故障报修", "页面上传截图后一直报错，支付失败提示系统异常", "anxious"),
    ("不会操作", "我不会操作，找不到修改收货地址的入口", "neutral"),
]


def main() -> None:
    print("智能客服五类意图 API 演示")
    print("=" * 40)
    with app.test_client() as client:
        for expected_label, text, emotion in CASES:
            response = client.post(
                "/api/predict",
                data={"text": text, "voice_text": "", "emotion": emotion},
            )
            payload = response.get_json()
            prediction = payload["prediction"]
            result = {
                "expected": expected_label,
                "predicted": prediction["intent_label"],
                "confidence": prediction["confidence"],
                "need_human": prediction["need_human"],
                "reply": prediction["response"]["reply"],
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("-" * 40)


if __name__ == "__main__":
    main()
