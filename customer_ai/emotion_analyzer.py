from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fer2013_cnn.pt"
FALLBACK_MODEL_PATH = ROOT / "models" / "fer2013_mlp.joblib"
_EMOTION_MODEL = None
_FALLBACK_MODEL = None
_EMOTION_THRESHOLD = 0.5
FER_LABELS = {
    0: ("angry", "愤怒"),
    1: ("neutral", "厌恶"),
    2: ("anxious", "恐惧"),
    3: ("happy", "开心"),
    4: ("sad", "悲伤"),
    5: ("anxious", "惊讶"),
    6: ("neutral", "平静"),
}


class EmotionCNN(nn.Module):
    def __init__(self, num_classes: int = 7, architecture: str = "legacy"):
        super().__init__()
        if architecture == "enhanced_cnn_v2":
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.15),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.20),
                nn.Flatten(),
                nn.Linear(64 * 12 * 12, 256),
                nn.ReLU(),
                nn.Dropout(0.35),
                nn.Linear(256, num_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * 12 * 12, 96),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(96, num_classes),
            )

    def forward(self, x):
        return self.net(x)


@dataclass
class EmotionSignals:
    label: str
    emotion: str
    confidence: float
    summary: str
    details: dict
    intent_scores: dict[str, float]


def analyze_emotion_image(path: str | Path | None) -> EmotionSignals | None:
    if not path:
        return None

    image_path = Path(path)
    if not image_path.exists():
        return None

    vector, tensor, width, height = _preprocess(image_path)
    cnn = _predict_cnn(tensor)
    fallback = None

    if cnn:
        emotion = cnn["emotion"]
        label = f"FER CNN {cnn['label']}"
        confidence = cnn["confidence"]
        summary = f"FER-2013 CNN 模型判断用户表情为{cnn['label']}。"
    else:
        model = _load_fallback_model()
        if model is None:
            return EmotionSignals(
                label="未加载表情模型",
                emotion="neutral",
                confidence=0.0,
                summary="未找到 FER-2013 CNN/MLP 模型，表情图片未参与模型推理。",
                details={"filename": image_path.name, "model_path": str(MODEL_PATH), "source": "missing_model"},
                intent_scores=_emotion_intent_scores("neutral", 0.0),
            )
        probabilities = _predict_proba(model, vector)
        angry_probability = float(probabilities[1]) if probabilities.size > 1 else float(probabilities[0])
        other_probability = float(1.0 - angry_probability)
        fallback = {
            "angry_probability": round(angry_probability, 4),
            "other_probability": round(other_probability, 4),
            "model": FALLBACK_MODEL_PATH.name,
        }
        if angry_probability >= 0.5:
            label = "FER MLP 愤怒倾向"
            emotion = "angry"
            confidence = angry_probability
            summary = "CNN 不可用，FER-2013 MLP 判断用户表情存在愤怒/不满倾向。"
        else:
            label = "FER MLP 非愤怒表情"
            emotion = "neutral"
            confidence = other_probability
            summary = "CNN 不可用，FER-2013 MLP 未检测到明显愤怒倾向。"

    return EmotionSignals(
        label=label,
        emotion=emotion,
        confidence=round(confidence, 4),
        summary=summary,
        details={
            "filename": image_path.name,
            "width": width,
            "height": height,
            "model": MODEL_PATH.name if cnn else (FALLBACK_MODEL_PATH.name if fallback else ""),
            "task": "fer2013_expression",
            "source": "fer2013_cnn" if cnn else "fer2013_mlp_fallback",
            "cnn": cnn,
            "fallback": fallback,
        },
        intent_scores={key: round(value, 6) for key, value in _emotion_intent_scores(emotion, confidence).items()},
    )


def _load_model():
    global _EMOTION_MODEL, _EMOTION_THRESHOLD
    if _EMOTION_MODEL is not None:
        return _EMOTION_MODEL
    if not MODEL_PATH.exists():
        return None
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    architecture = checkpoint.get("architecture", "legacy")
    num_classes = int(checkpoint.get("num_classes", 7))
    model = EmotionCNN(num_classes=num_classes, architecture=architecture)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _EMOTION_THRESHOLD = float(checkpoint.get("decision_threshold", 0.5))
    _EMOTION_MODEL = model
    return _EMOTION_MODEL


def _load_fallback_model():
    global _FALLBACK_MODEL
    if _FALLBACK_MODEL is not None:
        return _FALLBACK_MODEL
    if not FALLBACK_MODEL_PATH.exists():
        return None
    _FALLBACK_MODEL = joblib.load(FALLBACK_MODEL_PATH)
    return _FALLBACK_MODEL


def _preprocess(path: Path) -> tuple[np.ndarray, torch.Tensor, int, int]:
    with Image.open(path) as image:
        gray = image.convert("L")
        width, height = gray.size
        resized = gray.resize((48, 48))
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        vector = arr.reshape(1, -1)
        tensor = torch.tensor(arr[None, None, :, :], dtype=torch.float32)
    return vector, tensor, width, height


def _predict_cnn(tensor: torch.Tensor) -> dict | None:
    model = _load_model()
    if model is None:
        return None

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    index = int(np.argmax(probabilities))
    if probabilities.size == 2:
        angry_probability = float(probabilities[1])
        if angry_probability >= _EMOTION_THRESHOLD:
            index = 1
            emotion, label = ("angry", "愤怒")
        else:
            index = 0
            emotion, label = ("neutral", "非愤怒")
    else:
        emotion, label = FER_LABELS.get(index, ("neutral", str(index)))
    return {
        "class_index": index,
        "label": label,
        "emotion": emotion,
        "confidence": round(float(probabilities[index]), 4),
        "probabilities": {str(i): round(float(value), 4) for i, value in enumerate(probabilities)},
        "model": MODEL_PATH.name,
        "decision_threshold": round(float(_EMOTION_THRESHOLD), 4) if probabilities.size == 2 else None,
    }


def _predict_proba(model, vector: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(vector)[0], dtype=np.float32)

    pred = int(model.predict(vector)[0])
    probabilities = np.zeros(2, dtype=np.float32)
    probabilities[pred] = 1.0
    return probabilities


def _emotion_intent_scores(emotion: str, confidence: float) -> dict[str, float]:
    scores = {
        "consult": 0.20,
        "complaint": 0.20,
        "refund": 0.20,
        "repair": 0.20,
        "howto": 0.20,
    }
    strength = max(0.35, float(confidence))
    if emotion in {"angry", "sad"}:
        scores["complaint"] += 0.52 * strength
        scores["refund"] += 0.20 * strength
    elif emotion == "anxious":
        scores["repair"] += 0.32 * strength
        scores["howto"] += 0.24 * strength
    elif emotion == "happy":
        scores["consult"] += 0.22 * strength
    total = sum(scores.values()) or 1.0
    return {label: value / total for label, value in scores.items()}
