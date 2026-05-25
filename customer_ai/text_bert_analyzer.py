from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .knowledge_base import INTENT_LABELS


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "text_bert_classifier"
INTENT_ORDER = list(INTENT_LABELS.keys())


@dataclass
class TextIntentSignals:
    label: str
    confidence: float
    scores: dict[str, float]
    summary: str
    details: dict


class BertIntentAnalyzer:
    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.id_to_label: dict[int, str] = {}
        self.error = ""
        self._load()

    @property
    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def predict(self, text: str) -> TextIntentSignals | None:
        text = text.strip()
        if not text or not self.available:
            return None

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=0).cpu().numpy()

        scores = {label: 0.0 for label in INTENT_ORDER}
        for index, probability in enumerate(probabilities):
            label = self.id_to_label.get(index)
            if label in scores:
                scores[label] = float(probability)

        scores = _normalize_scores(scores)
        label = max(scores, key=scores.get)
        confidence = float(scores[label])
        return TextIntentSignals(
            label=label,
            confidence=round(confidence, 4),
            scores={key: round(value, 6) for key, value in scores.items()},
            summary="BERT 文本意图分支已完成推理。",
            details={
                "model_dir": str(self.model_dir),
                "model": self.model.config._name_or_path,
                "source": "bert_sequence_classifier",
            },
        )

    def _load(self) -> None:
        if not self.model_dir.exists():
            self.error = f"BERT 模型目录不存在：{self.model_dir}"
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir),
                local_files_only=True,
            )
            self.model.eval()
            raw_id_to_label = getattr(self.model.config, "id2label", {}) or {}
            self.id_to_label = {int(index): str(label) for index, label in raw_id_to_label.items()}
        except Exception as exc:  # pragma: no cover - defensive for optional local model files
            self.error = str(exc)
            self.tokenizer = None
            self.model = None
            self.id_to_label = {}


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    total = float(sum(scores.values()))
    if total <= 0:
        return {label: 1.0 / len(scores) for label in scores}
    return {label: float(value / total) for label, value in scores.items()}


def blend_scores(
    primary: dict[str, float],
    secondary: dict[str, float],
    primary_weight: float,
) -> dict[str, float]:
    primary_weight = float(np.clip(primary_weight, 0.0, 1.0))
    blended = {
        label: primary.get(label, 0.0) * primary_weight
        + secondary.get(label, 0.0) * (1.0 - primary_weight)
        for label in INTENT_ORDER
    }
    return _normalize_scores(blended)
