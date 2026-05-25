from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "librispeech_lstm.pt"
FALLBACK_MODEL_PATH = ROOT / "models" / "audio_mfcc_classifier.joblib"
_LSTM_MODEL = None
_LSTM_LABELS: list[str] = []


class AudioLSTM(nn.Module):
    def __init__(self, n_mfcc: int, hidden_size: int, num_classes: int, num_layers: int = 1):
        super().__init__()
        dropout = 0.2 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(n_mfcc, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.classifier(self.dropout(hidden[-1]))


@dataclass
class AudioSignals:
    label: str
    confidence: float
    summary: str
    details: dict
    intent_scores: dict[str, float]


def analyze_audio(path: str | Path | None) -> AudioSignals | None:
    if not path:
        return None

    audio_path = Path(path)
    if not audio_path.exists():
        return None

    audio, sr = librosa.load(audio_path, sr=16000, duration=8.0)
    if audio.size == 0:
        return AudioSignals(
            label="空音频",
            confidence=0.0,
            summary="上传的音频无法读取有效声音。",
            details={"sample_rate": sr, "duration_sec": 0.0},
            intent_scores=infer_audio_intent_scores(rms=0.0, zcr=0.0, centroid=0.0, confidence=0.0),
        )

    duration = float(librosa.get_duration(y=audio, sr=sr))
    rms = float(np.mean(librosa.feature.rms(y=audio)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    stats_features = extract_stats(audio, sr)
    sequence_features = extract_mfcc_sequence(audio, sr)

    label = "有效语音"
    confidence = 0.72
    source = "mfcc_feature_extraction"
    model_summary = "已提取 MFCC 序列、RMS、过零率、谱质心等声学特征。"
    lstm_result = _predict_lstm(sequence_features)
    fallback_result = None

    if lstm_result:
        label = f"LSTM 声学类别 {lstm_result['label']}"
        confidence = lstm_result["confidence"]
        source = "librispeech_lstm"
        model_summary = "已调用 LibriSpeech LSTM 语音分支。"
    elif FALLBACK_MODEL_PATH.exists():
        model = joblib.load(FALLBACK_MODEL_PATH)
        pred = model.predict([stats_features])[0]
        label = f"声学类别 {pred}"
        confidence = infer_confidence(model, stats_features)
        source = "mfcc_classifier_fallback"
        fallback_result = {
            "label": str(pred),
            "confidence": round(float(confidence), 4),
            "model": FALLBACK_MODEL_PATH.name,
        }
        model_summary = "LSTM 模型不可用，已使用 MFCC 统计分类器兜底。"

    loudness_summary = "音量较低，建议用户重新录制。" if rms < 0.01 else "音量有效，可参与声学分支分析。"
    intent_scores = infer_audio_intent_scores(rms=rms, zcr=zcr, centroid=centroid, confidence=confidence)
    return AudioSignals(
        label=label,
        confidence=round(float(confidence), 4),
        summary=f"{model_summary}{loudness_summary}",
        details={
            "duration_sec": round(duration, 2),
            "sample_rate": sr,
            "rms": round(rms, 5),
            "zero_crossing_rate": round(zcr, 5),
            "spectral_centroid": round(centroid, 2),
            "filename": audio_path.name,
            "model": MODEL_PATH.name if lstm_result else (FALLBACK_MODEL_PATH.name if fallback_result else ""),
            "source": source,
            "lstm": lstm_result,
            "fallback": fallback_result,
        },
        intent_scores={label: round(score, 6) for label, score in intent_scores.items()},
    )


def extract_stats(audio: np.ndarray, sr: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    feats = np.vstack([mfcc, delta, zcr, centroid])
    return np.concatenate([feats.mean(axis=1), feats.std(axis=1), feats.min(axis=1), feats.max(axis=1)])


def extract_mfcc_sequence(audio: np.ndarray, sr: int, n_mfcc: int = 20, max_frames: int = 120) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
    if len(mfcc) < max_frames:
        pad = np.zeros((max_frames - len(mfcc), n_mfcc), dtype=np.float32)
        mfcc = np.vstack([mfcc, pad])
    return mfcc[:max_frames].astype(np.float32)


def _predict_lstm(features: np.ndarray) -> dict | None:
    model = _load_lstm_model()
    if model is None:
        return None

    with torch.no_grad():
        logits = model(torch.tensor(features[None, :, :], dtype=torch.float32))[0]
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()
    index = int(np.argmax(probabilities))
    label = _LSTM_LABELS[index] if index < len(_LSTM_LABELS) else str(index)
    return {
        "label": label,
        "confidence": round(float(probabilities[index]), 4),
        "probabilities": {
            _LSTM_LABELS[i] if i < len(_LSTM_LABELS) else str(i): round(float(value), 4)
            for i, value in enumerate(probabilities)
        },
        "model": MODEL_PATH.name,
    }


def _load_lstm_model():
    global _LSTM_MODEL, _LSTM_LABELS
    if _LSTM_MODEL is not None:
        return _LSTM_MODEL
    if not MODEL_PATH.exists():
        return None

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    label_to_id = checkpoint.get("label_to_id", {})
    _LSTM_LABELS = [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]
    state = checkpoint["model_state_dict"]
    hidden_size = int(checkpoint.get("hidden_size") or state["classifier.weight"].shape[1])
    num_layers = int(checkpoint.get("num_layers") or len([key for key in state if key.startswith("lstm.weight_ih_l")]))
    model = AudioLSTM(n_mfcc=20, hidden_size=hidden_size, num_classes=len(_LSTM_LABELS), num_layers=num_layers)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _LSTM_MODEL = model
    return _LSTM_MODEL


def infer_audio_intent_scores(rms: float, zcr: float, centroid: float, confidence: float) -> dict[str, float]:
    scores = {
        "consult": 0.20,
        "complaint": 0.20,
        "refund": 0.20,
        "repair": 0.20,
        "howto": 0.20,
    }
    if rms < 0.01:
        scores["howto"] += 0.22
        scores["repair"] += 0.12
    else:
        scores["consult"] += 0.08

    if zcr > 0.11 or centroid > 2300:
        scores["complaint"] += 0.15 * max(0.5, confidence)
        scores["repair"] += 0.10
    elif centroid < 1200:
        scores["refund"] += 0.08

    total = sum(scores.values()) or 1.0
    return {label: value / total for label, value in scores.items()}


def infer_confidence(model, features: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([features])[0]
        return float(np.max(probabilities))

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function([features])).reshape(-1)
        if scores.size == 1:
            return float(1 / (1 + np.exp(-abs(scores[0]))))
        scores = scores - scores.max()
        exp = np.exp(scores)
        return float(np.max(exp / exp.sum()))

    return 0.72
