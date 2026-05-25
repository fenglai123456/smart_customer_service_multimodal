from __future__ import annotations

import unittest

from app import app


class PredictApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def test_health(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_five_core_intents(self) -> None:
        cases = [
            {
                "name": "consult",
                "text": "我想咨询一下订单物流什么时候更新",
                "emotion": "neutral",
                "expected": "consult",
            },
            {
                "name": "complaint",
                "text": "客服一直不回复，服务态度太差了，我要投诉",
                "emotion": "angry",
                "expected": "complaint",
                "need_human": True,
            },
            {
                "name": "refund",
                "text": "订单取消了，我想申请退款，钱什么时候退回",
                "emotion": "neutral",
                "expected": "refund",
            },
            {
                "name": "repair",
                "text": "页面上传截图后一直报错，支付失败提示系统异常",
                "emotion": "anxious",
                "expected": "repair",
            },
            {
                "name": "howto",
                "text": "我不会操作，找不到修改收货地址的入口",
                "emotion": "neutral",
                "expected": "howto",
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                payload = self._predict(text=case["text"], emotion=case["emotion"])
                prediction = payload["prediction"]
                self.assertEqual(prediction["intent"], case["expected"])
                self.assertGreater(prediction["confidence"], 0.2)
                self.assertTrue(prediction["response"]["reply"])
                self.assertGreaterEqual(len(prediction["ranking"]), 5)
                if "need_human" in case:
                    self.assertEqual(prediction["need_human"], case["need_human"])

    def test_voice_text_is_merged(self) -> None:
        payload = self._predict(
            text="订单没有更新",
            voice_text="我想咨询物流进度",
            emotion="neutral",
        )
        prediction = payload["prediction"]
        self.assertEqual(prediction["intent"], "consult")
        self.assertTrue(prediction["modality_signals"]["voice"]["provided"])

    def test_feature_concatenation_mlp_fusion_is_exposed(self) -> None:
        payload = self._predict(
            text="页面一直报错，上传截图后支付失败",
            voice_text="我现在很着急，麻烦尽快处理",
            emotion="anxious",
        )
        fusion = payload["prediction"]["modality_signals"]["fusion"]
        details = fusion["details"]
        blocks = {block["key"]: block for block in details["feature_blocks"]}

        self.assertEqual(fusion["source"], "fully_connected_fusion")
        self.assertEqual(details["fusion_type"], "feature_concatenation_plus_fully_connected")
        self.assertEqual(details["feature_dim"], 26)
        self.assertEqual(details["mlp_layers"], [26, 48, 24, 5])
        self.assertEqual(len(details["feature_vector"]), 26)
        for key in ("text", "audio", "emotion", "image"):
            self.assertEqual(blocks[key]["dim"], 5)
            self.assertEqual(len(blocks[key]["values"]), 5)
        self.assertEqual(blocks["numeric"]["dim"], 6)
        self.assertEqual(len(blocks["numeric"]["values"]), 6)

    def _predict(self, text: str, emotion: str, voice_text: str = "") -> dict:
        response = self.client.post(
            "/api/predict",
            data={"text": text, "voice_text": voice_text, "emotion": emotion},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("ticket_id", payload)
        self.assertIn("prediction", payload)
        return payload


if __name__ == "__main__":
    unittest.main()
