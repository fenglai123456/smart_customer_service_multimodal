from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from customer_ai.audio_analyzer import analyze_audio
from customer_ai.emotion_analyzer import analyze_emotion_image


DEMO_ROOT = ROOT / "demo_assets" / "media"
AUDIO_OUT = DEMO_ROOT / "audio"
EMOTION_OUT = DEMO_ROOT / "emotion_faces"
REPORT_PATH = ROOT / "reports" / "course_demo_media_predictions.json"

AUDIO_CASES = [
    (
        "voice_01_public_librispeech.flac",
        ROOT
        / "datasets"
        / "audio"
        / "librispeech"
        / "LibriSpeech"
        / "dev-clean"
        / "1272"
        / "128104"
        / "1272-128104-0000.flac",
        "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL",
    ),
    (
        "voice_02_public_librispeech.flac",
        ROOT
        / "datasets"
        / "audio"
        / "librispeech"
        / "LibriSpeech"
        / "dev-clean"
        / "1462"
        / "170138"
        / "1462-170138-0001.flac",
        "HUGH'S WRITTEN A DELIGHTFUL PART FOR HER AND SHE'S QUITE INEXPRESSIBLE",
    ),
]

EMOTION_TARGETS = [
    ("angry", "愤怒/不满表情"),
    ("neutral", "非愤怒表情"),
]


def copy_audio_samples() -> list[dict]:
    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for filename, source, transcript in AUDIO_CASES:
        target = AUDIO_OUT / filename
        shutil.copyfile(source, target)
        result = analyze_audio(target)
        rows.append(
            {
                "file": str(target),
                "source": str(source),
                "transcript": transcript,
                "result": result.__dict__ if result else None,
            }
        )
    return rows


def export_emotion_samples() -> list[dict]:
    csv_path = ROOT / "datasets" / "emotion" / "fer2013" / "fer2013.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    EMOTION_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    found = {name: False for name, _ in EMOTION_TARGETS}
    with csv_path.open("r", encoding="utf-8", newline="") as reader:
        for row_index, item in enumerate(csv.DictReader(reader)):
            emotion_id = int(item["emotion"])
            pixels = np.asarray([int(value) for value in item["pixels"].split()], dtype=np.uint8).reshape(48, 48)
            for name, description in EMOTION_TARGETS:
                if found[name]:
                    continue
                image = Image.fromarray(pixels, mode="L").resize((192, 192), Image.Resampling.NEAREST)
                candidate = EMOTION_OUT / f"_candidate_{name}.png"
                image.save(candidate)
                result = analyze_emotion_image(candidate)
                if result is None or result.emotion != name or result.confidence < 0.70:
                    candidate.unlink(missing_ok=True)
                    continue
                target = EMOTION_OUT / f"{name}_fer2013_demo.png"
                if target.exists():
                    target.unlink()
                candidate.rename(target)
                result = analyze_emotion_image(target)
                rows.append(
                    {
                        "file": str(target),
                        "source": "FER-2013 fer2013.csv",
                        "fer2013_row_index": row_index,
                        "fer2013_emotion_id": emotion_id,
                        "expected_demo_emotion": description,
                        "result": result.__dict__ if result else None,
                    }
                )
                found[name] = True
            if all(found.values()):
                break

    missing = [name for name, done in found.items() if not done]
    if missing:
        raise RuntimeError(f"Missing FER-2013 demo samples: {', '.join(missing)}")
    return rows


def main() -> None:
    report = {
        "task": "course demo audio and emotion media preparation",
        "audio": copy_audio_samples(),
        "emotion_faces": export_emotion_samples(),
        "note": "Audio samples are copied from the local LibriSpeech dev-clean public dataset. Emotion images are exported from local FER-2013 CSV for course demo uploads.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
