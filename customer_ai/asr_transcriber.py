from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import librosa


DEFAULT_MODEL = "openai/whisper-tiny"
_ASR_PIPELINE = None
_ASR_MODEL_NAME = None


@dataclass
class TranscriptionResult:
    text: str
    confidence: float | None
    summary: str
    details: dict


def transcribe_audio(path: str | Path | None) -> TranscriptionResult | None:
    if not path:
        return None

    audio_path = Path(path)
    if not audio_path.exists():
        return None

    model_name = os.environ.get("CUSTOMER_AI_ASR_MODEL", DEFAULT_MODEL)
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True, duration=20.0)
        pipe = _load_pipeline(model_name)
        result = pipe({"array": audio, "sampling_rate": sr})
        text = str(result.get("text", "")).strip()
        summary = "已使用 ASR 模型自动转写语音。" if text else "ASR 模型未返回有效文本。"
        return TranscriptionResult(
            text=text,
            confidence=None,
            summary=summary,
            details={
                "filename": audio_path.name,
                "model": model_name,
                "sample_rate": sr,
                "duration_sec": round(float(librosa.get_duration(y=audio, sr=sr)), 2),
                "source": "transformers_asr",
            },
        )
    except Exception as exc:  # pragma: no cover - depends on local model/network availability.
        return TranscriptionResult(
            text="",
            confidence=None,
            summary="ASR 自动转写未完成，请使用手动语音转写文本。",
            details={
                "filename": audio_path.name,
                "model": model_name,
                "source": "transformers_asr",
                "error": str(exc),
            },
        )


def _load_pipeline(model_name: str):
    global _ASR_PIPELINE, _ASR_MODEL_NAME
    if _ASR_PIPELINE is not None and _ASR_MODEL_NAME == model_name:
        return _ASR_PIPELINE

    from transformers import pipeline

    _ASR_PIPELINE = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=-1,
    )
    _ASR_MODEL_NAME = model_name
    return _ASR_PIPELINE
