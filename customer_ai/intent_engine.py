from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .audio_analyzer import AudioSignals
from .emotion_analyzer import EmotionSignals
from .fusion_model import FusionIntentModel, normalize_scores
from .image_analyzer import ImageSignals
from .knowledge_base import (
    INTENT_DESCRIPTIONS,
    INTENT_LABELS,
    KEYWORD_HINTS,
    RESPONSE_TEMPLATES,
    SENTIMENT_WORDS,
    TRAINING_SAMPLES,
)
from .text_bert_analyzer import BertIntentAnalyzer, blend_scores


@dataclass
class Prediction:
    intent: str
    intent_label: str
    confidence: float
    response: dict
    need_human: bool
    reasons: list[str]
    modality_signals: dict
    ranking: list[dict]


class IntentEngine:
    EMOTION_IMAGE_OVERRIDE_THRESHOLD = 0.65

    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[1]
        model_candidates = [
            root / "models" / "text_intent_augmented.joblib",
            root / "models" / "text_intent_tfidf_lr.joblib",
        ]
        model_path = next((path for path in model_candidates if path.exists()), None)
        if model_path:
            self.model = joblib.load(model_path)
        else:
            texts, labels = zip(*TRAINING_SAMPLES)
            self.model: Pipeline = Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
                    (
                        "clf",
                        LogisticRegression(max_iter=600, class_weight="balanced", random_state=7),
                    ),
                ]
            )
            self.model.fit(texts, labels)
        self.labels = list(self.model.named_steps["clf"].classes_)
        self.bert_analyzer = BertIntentAnalyzer()
        self.fusion_model = FusionIntentModel()

    def predict(
        self,
        text: str = "",
        voice_text: str = "",
        emotion: str = "neutral",
        image: ImageSignals | None = None,
        audio: AudioSignals | None = None,
        emotion_image: EmotionSignals | None = None,
        asr: dict | None = None,
    ) -> Prediction:
        if (
            emotion_image
            and emotion_image.emotion != "neutral"
            and emotion_image.confidence >= self.EMOTION_IMAGE_OVERRIDE_THRESHOLD
        ):
            emotion = emotion_image.emotion
        merged_text = self._normalize_text(f"{text} {voice_text}")
        baseline_probs = self._model_scores(merged_text)
        bert_signals = self.bert_analyzer.predict(merged_text)
        if bert_signals:
            # Tiny local BERT models can be weak, so the strict BERT branch is fused with
            # the demo text classifier for stable classroom behavior.
            model_probs = blend_scores(bert_signals.scores, baseline_probs, primary_weight=0.62)
        else:
            model_probs = baseline_probs
        keyword_scores = self._keyword_scores(merged_text)
        emotion_scores, emotion_reasons = self._emotion_scores(emotion, merged_text, emotion_image)
        image_scores, image_reason = self._image_scores(image)
        audio_scores = self._audio_scores(audio)

        # Text remains the strongest signal in a customer-service ticket, but a
        # small keyword component makes short classroom examples more stable and
        # easier to explain in the report.
        strict_text_scores = normalize_scores(
            {
                label: model_probs.get(label, 0.0) * 0.78 + keyword_scores.get(label, 0.0) * 0.22
                for label in self.labels
            }
        )
        fusion_text_scores = self._demo_screenshot_fusion_text_scores(
            text_scores=strict_text_scores,
            image_scores=image_scores,
            image=image,
            merged_text=merged_text,
        )

        # These six scalar features become positions 20-25 in the fusion vector.
        # They tell the MLP which modalities were present and how confident the
        # audio/emotion/screenshot analyzers were.
        numeric_features = [
            min(len(merged_text) / 80.0, 1.0),
            1.0 if text.strip() else 0.0,
            1.0 if voice_text.strip() else 0.0,
            float(audio.confidence) if audio else 0.0,
            float(emotion_image.confidence) if emotion_image else 0.0,
            float(image.confidence) if image else 0.0,
        ]
        fusion_result = self.fusion_model.predict(
            text_scores=fusion_text_scores,
            audio_scores=audio_scores,
            emotion_scores=emotion_scores,
            image_scores=image_scores,
            numeric_features=numeric_features,
        )
        normalized = fusion_result.scores
        ranking = sorted(
            (
                {
                    "intent": label,
                    "label": INTENT_LABELS[label],
                    "description": INTENT_DESCRIPTIONS[label],
                    "score": round(score, 4),
                }
                for label, score in normalized.items()
            ),
            key=lambda item: item["score"],
            reverse=True,
        )

        top = ranking[0]
        confidence = float(top["score"])
        reasons = self._build_reasons(
            merged_text=merged_text,
            intent=top["intent"],
            confidence=confidence,
            emotion=emotion,
            emotion_reasons=emotion_reasons,
            image_reason=image_reason,
            audio=audio,
            emotion_image=emotion_image,
            asr=asr,
            bert=bert_signals,
            fusion=fusion_result,
        )
        need_human = self._need_human(confidence, top["intent"], emotion, merged_text)
        response = self._build_response(top["intent"], confidence, need_human)

        return Prediction(
            intent=top["intent"],
            intent_label=top["label"],
            confidence=round(confidence, 4),
            response=response,
            need_human=need_human,
            reasons=reasons,
            modality_signals={
                "text": {
                    "provided": bool(text.strip()),
                    "chars": len(text.strip()),
                },
                "voice": {
                    "provided": bool(voice_text.strip()),
                    "transcript": voice_text.strip(),
                },
                "audio": asdict(audio) if audio else None,
                "asr": asr,
                "emotion": {
                    "value": emotion,
                    "summary": self._emotion_summary(emotion),
                },
                "emotion_image": asdict(emotion_image) if emotion_image else None,
                "image": asdict(image) if image else None,
                "text_model": asdict(bert_signals)
                if bert_signals
                else {
                    "label": None,
                    "confidence": 0.0,
                    "scores": baseline_probs,
                    "summary": "BERT 模型不可用，文本分支使用 TF-IDF 兜底。",
                    "details": {"source": "tfidf_fallback", "error": self.bert_analyzer.error},
                },
                "fusion": asdict(fusion_result),
            },
            ranking=ranking,
        )

    def _model_scores(self, text: str) -> dict[str, float]:
        if not text:
            return {label: 1 / len(self.labels) for label in self.labels}

        probabilities = self.model.predict_proba([text])[0]
        return {label: float(prob) for label, prob in zip(self.labels, probabilities)}

    def _keyword_scores(self, text: str) -> dict[str, float]:
        if not text:
            return {label: 0.2 for label in self.labels}

        raw = {}
        for label, words in KEYWORD_HINTS.items():
            raw[label] = sum(1.0 for word in words if word in text)

        if max(raw.values(), default=0) == 0:
            return {label: 0.2 for label in self.labels}

        total = sum(raw.values()) or 1
        return {label: raw.get(label, 0.0) / total for label in self.labels}

    def _emotion_scores(
        self,
        emotion: str,
        text: str,
        emotion_image: EmotionSignals | None = None,
    ) -> tuple[dict[str, float], list[str]]:
        scores = {label: 0.2 for label in self.labels}
        reasons = []

        if emotion_image and emotion_image.intent_scores:
            scores = {label: scores[label] + emotion_image.intent_scores.get(label, 0.0) * 0.9 for label in self.labels}

        if emotion in {"angry", "sad"}:
            boost = 0.5
            if (
                emotion_image
                and emotion_image.emotion == "angry"
                and emotion_image.confidence >= self.EMOTION_IMAGE_OVERRIDE_THRESHOLD
            ):
                boost += 0.22 * emotion_image.confidence
            scores["complaint"] += boost
            scores["refund"] += 0.18
            reasons.append("表情/情绪偏负向，提高投诉与退款权重")
            if emotion_image and emotion_image.confidence >= self.EMOTION_IMAGE_OVERRIDE_THRESHOLD:
                reasons.append(f"表情图片模型输出：{emotion_image.label}，置信度 {emotion_image.confidence:.0%}")
        elif emotion == "anxious":
            scores["repair"] += 0.28
            scores["howto"] += 0.22
            reasons.append("用户表现焦急，提高故障报修与操作引导权重")
        elif emotion == "happy":
            scores["consult"] += 0.25
            reasons.append("用户情绪平稳偏正向，优先保持自助接待")

        negative_hits = [word for word in SENTIMENT_WORDS["negative"] if word in text]
        if negative_hits:
            scores["complaint"] += 0.35
            reasons.append(f"文本中出现负向词：{', '.join(negative_hits[:3])}")

        total = sum(scores.values()) or 1
        return {label: scores[label] / total for label in self.labels}, reasons

    def _image_scores(self, image: ImageSignals | None) -> tuple[dict[str, float], str | None]:
        scores = {label: 0.2 for label in self.labels}
        if image is None:
            return scores, None

        details = image.details or {}
        screenshot_decision = details.get("screenshot_demo_decision") or details.get("screenshot_intent_cnn") or {}
        screenshot_class = str(screenshot_decision.get("class_name", ""))
        decision_source = str(screenshot_decision.get("decision_source", screenshot_decision.get("source", "")))
        screenshot_intent_map = {
            "fault": "repair",
            "order": "consult",
            "refund": "refund",
            "howto": "howto",
            "normal": "consult",
        }
        is_demo_screenshot = any(
            marker in decision_source.lower()
            for marker in ("demo", "keyword", "ocr")
        ) or "demo" in str(screenshot_decision.get("model", "")).lower()

        if image.intent_scores:
            weight = 2.0 if is_demo_screenshot else 0.9
            scores = {label: scores[label] + image.intent_scores.get(label, 0.0) * weight for label in self.labels}

        mapped_intent = screenshot_intent_map.get(screenshot_class)
        if mapped_intent and is_demo_screenshot:
            scores[mapped_intent] += 0.9 * max(float(image.confidence), 0.7)
            if screenshot_class == "fault":
                scores["complaint"] += 0.16
            elif screenshot_class == "refund":
                scores["complaint"] += 0.10
            reason = f"演示截图 CNN/OCR关键词命中 {screenshot_class}，图片模态提高 {INTENT_LABELS[mapped_intent]} 权重"
        elif "故障" in image.label:
            scores["repair"] += 0.55
            reason = "图片特征接近故障截图，提高报修权重"
        elif "较暗" in image.label or "信息较少" in image.label:
            scores["howto"] += 0.22
            scores["repair"] += 0.18
            reason = "图片信息不足，建议继续追问并保留人工入口"
        else:
            scores["consult"] += 0.18
            scores["howto"] += 0.12
            reason = "图片作为辅助信息参与融合判断"

        total = sum(scores.values()) or 1
        return {label: scores[label] / total for label in self.labels}, reason

    def _demo_screenshot_fusion_text_scores(
        self,
        text_scores: dict[str, float],
        image_scores: dict[str, float],
        image: ImageSignals | None,
        merged_text: str,
    ) -> dict[str, float]:
        """Let demo screenshot evidence assist sparse text before final fusion.

        The screenshot branch is intentionally transparent for classroom use:
        it only changes text scores when the screenshot decision came from the
        demo CNN/OCR rule layer and either text is empty or text keywords agree.
        """
        if image is None:
            return text_scores

        details = image.details or {}
        screenshot_decision = details.get("screenshot_demo_decision") or {}
        decision_source = str(screenshot_decision.get("decision_source", screenshot_decision.get("source", ""))).lower()
        model_name = str(screenshot_decision.get("model", "")).lower()
        if "demo" not in decision_source and "keyword" not in decision_source and "demo" not in model_name:
            return text_scores

        class_name = str(screenshot_decision.get("class_name", ""))
        mapped_intent = {
            "fault": "repair",
            "order": "consult",
            "refund": "refund",
            "howto": "howto",
            "normal": "consult",
        }.get(class_name)
        if not mapped_intent:
            return text_scores

        text_keyword_hits = [word for word in KEYWORD_HINTS.get(mapped_intent, []) if word in merged_text]
        if merged_text.strip() and not text_keyword_hits:
            return text_scores

        if merged_text.strip():
            blended = {
                label: text_scores.get(label, 0.0) * 0.72 + image_scores.get(label, 0.0) * 0.28
                for label in self.labels
            }
            blended[mapped_intent] += 0.24 * max(float(image.confidence), 0.7)
            return normalize_scores(blended)

        blended = {
            label: text_scores.get(label, 0.0) * 0.65 + image_scores.get(label, 0.0) * 0.35
            for label in self.labels
        }
        blended[mapped_intent] += 0.35 * max(float(image.confidence), 0.7)
        return normalize_scores(blended)

    def _audio_scores(self, audio: AudioSignals | None) -> dict[str, float]:
        if audio and audio.intent_scores:
            return normalize_scores(audio.intent_scores)
        return {label: 1 / len(self.labels) for label in self.labels}

    def _need_human(self, confidence: float, intent: str, emotion: str, text: str) -> bool:
        direct_words = ["人工", "真人", "客服", "经理", "投诉到底", "升级处理"]
        if any(word in text for word in direct_words):
            return True
        if confidence < 0.42:
            return True
        if intent == "complaint" and emotion in {"angry", "sad"}:
            return True
        return False

    def _build_response(self, intent: str, confidence: float, need_human: bool) -> dict:
        template = RESPONSE_TEMPLATES[intent]
        reply = template["reply"]
        if need_human:
            reply += " 当前已触发人工联动条件，我会把多模态记录同步给人工坐席。"
        elif confidence < 0.52:
            reply += " 当前置信度中等，我会先追问一个关键信息来确认。"

        return {
            "reply": reply,
            "action": template["action"],
            "steps": template["steps"],
            "handoff_message": "已整理用户文本、语音转写、音频声学、表情状态和图片摘要，可交由人工继续处理。"
            if need_human
            else "",
        }

    def _build_reasons(
        self,
        merged_text: str,
        intent: str,
        confidence: float,
        emotion: str,
        emotion_reasons: list[str],
        image_reason: str | None,
        audio: AudioSignals | None,
        emotion_image: EmotionSignals | None = None,
        asr: dict | None = None,
        bert=None,
        fusion=None,
    ) -> list[str]:
        reasons = []
        matched = [word for word in KEYWORD_HINTS[intent] if word in merged_text]
        if matched:
            reasons.append(f"命中 {INTENT_LABELS[intent]} 关键词：{', '.join(matched[:4])}")
        if merged_text:
            if bert:
                reasons.append(f"BERT 文本意图分支输出：{INTENT_LABELS.get(bert.label, bert.label)}，置信度 {bert.confidence:.0%}")
            else:
                reasons.append("文字与语音转写已合并后送入文本意图模型")
        if emotion != "neutral":
            reasons.extend(emotion_reasons)
        if image_reason:
            reasons.append(image_reason)
        if audio:
            reasons.append(f"LSTM 语音声学分支已参与分析：{audio.summary}")
        if asr:
            if asr.get("text"):
                reasons.append("ASR 自动转写文本已并入意图识别")
            else:
                reasons.append(asr.get("summary", "ASR 自动转写未产生有效文本"))
        if emotion_image and (
            emotion_image.emotion == "neutral"
            or emotion_image.confidence < self.EMOTION_IMAGE_OVERRIDE_THRESHOLD
        ):
            reasons.append(f"表情图片模型输出：{emotion_image.label}，未达到覆盖手动情绪的置信度")
        if confidence < 0.42:
            reasons.append("融合置信度低于阈值，建议转人工复核")
        if fusion:
            reasons.append(f"多模态特征已拼接后进入融合层：{fusion.summary}")
        if not reasons:
            reasons.append("信息较少，系统根据当前输入给出初步接待判断")
        return reasons

    def _emotion_summary(self, emotion: str) -> str:
        return {
            "neutral": "平静",
            "happy": "积极",
            "angry": "愤怒/不满",
            "sad": "低落",
            "anxious": "焦急",
        }.get(emotion, "未知")

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
