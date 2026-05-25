from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .knowledge_base import INTENT_LABELS


INTENT_ORDER = list(INTENT_LABELS.keys())
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "fusion_intent_mlp.pt"

# Four modality branches each emit five intent scores. The final six values are
# confidence/presence features, so the MLP input is fixed at 5*4+6 = 26 dims.
FEATURE_DIM = len(INTENT_ORDER) * 4 + 6
NUMERIC_FEATURE_NAMES = [
    "text_length_ratio",
    "has_text",
    "has_voice_text",
    "audio_confidence",
    "emotion_image_confidence",
    "image_confidence",
]
FUSION_ARCHITECTURE = [
    "Concat(text_scores:5, audio_scores:5, emotion_scores:5, image_scores:5, numeric_features:6)",
    "Linear(26, 48)",
    "ReLU",
    "Dropout(0.1)",
    "Linear(48, 24)",
    "ReLU",
    "Linear(24, 5)",
    "Softmax",
]
MLP_LAYER_DIMS = [FEATURE_DIM, 48, 24, len(INTENT_ORDER)]


class FusionMLP(nn.Module):
    """Fully-connected fusion head used by the multimodal intent engine."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 48, num_classes: int = len(INTENT_ORDER)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class FusionResult:
    scores: dict[str, float]
    source: str
    summary: str
    details: dict


class FusionIntentModel:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.model: FusionMLP | None = None
        self.labels = INTENT_ORDER
        self.error = ""
        self._load()

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(
        self,
        text_scores: dict[str, float],
        audio_scores: dict[str, float],
        emotion_scores: dict[str, float],
        image_scores: dict[str, float],
        numeric_features: list[float],
    ) -> FusionResult:
        features = build_fusion_features(
            text_scores,
            audio_scores,
            emotion_scores,
            image_scores,
            numeric_features,
        )
        if self.model is None:
            # Keep the web demo usable when a trained .pt file is absent; the
            # returned details still expose the same 26-dimensional feature vector.
            scores = weighted_average_scores(text_scores, audio_scores, emotion_scores, image_scores)
            return FusionResult(
                scores=scores,
                source="weighted_fallback",
                summary="未找到全连接融合模型，已使用同维度特征的加权融合兜底。",
                details=build_fusion_details(
                    text_scores=text_scores,
                    audio_scores=audio_scores,
                    emotion_scores=emotion_scores,
                    image_scores=image_scores,
                    numeric_features=numeric_features,
                    features=features,
                    model_path=self.model_path,
                    labels=self.labels,
                    error=self.error,
                ),
            )

        with torch.no_grad():
            # The trained MLP receives exactly one concatenated feature vector
            # and returns logits for consult/complaint/refund/repair/howto.
            logits = self.model(torch.tensor(features[None, :], dtype=torch.float32))[0]
            probabilities = torch.softmax(logits, dim=0).cpu().numpy()
        scores = {label: float(probabilities[index]) for index, label in enumerate(self.labels)}
        return FusionResult(
            scores=normalize_scores(scores),
            source="fully_connected_fusion",
            summary="文字、语音、表情、图片特征已拼接并输入全连接融合层。",
            details=build_fusion_details(
                text_scores=text_scores,
                audio_scores=audio_scores,
                emotion_scores=emotion_scores,
                image_scores=image_scores,
                numeric_features=numeric_features,
                features=features,
                model_path=self.model_path,
                labels=self.labels,
            ),
        )

    def _load(self) -> None:
        if not self.model_path.exists():
            self.error = f"融合模型不存在：{self.model_path}"
            return

        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
            labels = checkpoint.get("labels", INTENT_ORDER)
            input_dim = int(checkpoint.get("input_dim", FEATURE_DIM))
            self.labels = list(labels)
            self.model = FusionMLP(input_dim=input_dim, num_classes=len(self.labels))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
        except Exception as exc:  # pragma: no cover - defensive for optional local model files
            self.error = str(exc)
            self.model = None
            self.labels = INTENT_ORDER


def build_fusion_features(
    text_scores: dict[str, float],
    audio_scores: dict[str, float],
    emotion_scores: dict[str, float],
    image_scores: dict[str, float],
    numeric_features: list[float],
) -> np.ndarray:
    """Build the report-required multimodal feature vector.

    Layout:
    0-4   text intent scores
    5-9   audio intent scores
    10-14 emotion/image-expression scores
    15-19 business screenshot scores
    20-25 auxiliary numeric features
    """
    values: list[float] = []
    for scores in (text_scores, audio_scores, emotion_scores, image_scores):
        normalized = normalize_scores(scores)
        values.extend(float(normalized.get(label, 0.0)) for label in INTENT_ORDER)

    # Pad missing auxiliary features so manual uploads and demo cases share the
    # same fixed-size tensor before entering the full-connected fusion model.
    padded_numeric = list(numeric_features[:6])
    padded_numeric.extend([0.0] * (6 - len(padded_numeric)))
    values.extend(float(np.clip(value, 0.0, 1.0)) for value in padded_numeric)
    return np.asarray(values, dtype=np.float32)


def build_fusion_details(
    text_scores: dict[str, float],
    audio_scores: dict[str, float],
    emotion_scores: dict[str, float],
    image_scores: dict[str, float],
    numeric_features: list[float],
    features: np.ndarray,
    model_path: Path,
    labels: list[str],
    error: str = "",
) -> dict:
    blocks = []
    offset = 0
    for key, name, scores in (
        ("text", "文字意图分支", text_scores),
        ("audio", "语音声学分支", audio_scores),
        ("emotion", "表情/情绪分支", emotion_scores),
        ("image", "业务截图分支", image_scores),
    ):
        normalized = normalize_scores(scores)
        top_label = max(INTENT_ORDER, key=lambda label: normalized.get(label, 0.0))
        blocks.append(
            {
                "key": key,
                "name": name,
                "dim": len(INTENT_ORDER),
                "range": [offset, offset + len(INTENT_ORDER) - 1],
                "label_order": INTENT_ORDER,
                "values": _rounded_scores(normalized),
                "top": {
                    "label": top_label,
                    "label_name": INTENT_LABELS.get(top_label, top_label),
                    "score": _rounded_float(normalized.get(top_label, 0.0)),
                },
            }
        )
        offset += len(INTENT_ORDER)

    numeric_values = _padded_numeric_features(numeric_features)
    blocks.append(
        {
            "key": "numeric",
            "name": "辅助数值特征",
            "dim": len(NUMERIC_FEATURE_NAMES),
            "range": [offset, offset + len(NUMERIC_FEATURE_NAMES) - 1],
            "feature_order": NUMERIC_FEATURE_NAMES,
            "values": {
                name: _rounded_float(value)
                for name, value in zip(NUMERIC_FEATURE_NAMES, numeric_values)
            },
        }
    )

    details = {
        "model_path": str(model_path),
        "fusion_type": "feature_concatenation_plus_fully_connected",
        "feature_dim": int(features.shape[0]),
        "expected_feature_dim": FEATURE_DIM,
        "concat_formula": "text_scores(5)+audio_scores(5)+emotion_scores(5)+image_scores(5)+numeric_features(6)=26",
        "architecture": FUSION_ARCHITECTURE,
        "mlp_layers": MLP_LAYER_DIMS,
        "labels": labels,
        "label_order": labels,
        "feature_label_order": INTENT_ORDER,
        "label_names": {label: INTENT_LABELS.get(label, label) for label in INTENT_ORDER},
        "numeric_feature_names": NUMERIC_FEATURE_NAMES,
        "feature_blocks": blocks,
        "feature_vector": [_rounded_float(value) for value in features.tolist()],
    }
    if error:
        details["error"] = error
    return details


def _padded_numeric_features(numeric_features: list[float]) -> list[float]:
    padded_numeric = list(numeric_features[: len(NUMERIC_FEATURE_NAMES)])
    padded_numeric.extend([0.0] * (len(NUMERIC_FEATURE_NAMES) - len(padded_numeric)))
    return [float(np.clip(value, 0.0, 1.0)) for value in padded_numeric]


def _rounded_scores(scores: dict[str, float]) -> dict[str, float]:
    return {label: _rounded_float(scores.get(label, 0.0)) for label in INTENT_ORDER}


def _rounded_float(value: float) -> float:
    return round(float(value), 4)


def weighted_average_scores(
    text_scores: dict[str, float],
    audio_scores: dict[str, float],
    emotion_scores: dict[str, float],
    image_scores: dict[str, float],
) -> dict[str, float]:
    weights = (0.52, 0.16, 0.16, 0.16)
    scores = {}
    for label in INTENT_ORDER:
        scores[label] = (
            normalize_scores(text_scores).get(label, 0.0) * weights[0]
            + normalize_scores(audio_scores).get(label, 0.0) * weights[1]
            + normalize_scores(emotion_scores).get(label, 0.0) * weights[2]
            + normalize_scores(image_scores).get(label, 0.0) * weights[3]
        )
    return normalize_scores(scores)


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    values = {label: max(0.0, float(scores.get(label, 0.0))) for label in INTENT_ORDER}
    total = sum(values.values())
    if total <= 0:
        return {label: 1.0 / len(INTENT_ORDER) for label in INTENT_ORDER}
    return {label: value / total for label, value in values.items()}
