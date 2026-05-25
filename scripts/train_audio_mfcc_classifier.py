from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
LIBRISPEECH = ROOT / "datasets" / "audio" / "librispeech" / "LibriSpeech"
REPORT_PATH = ROOT / "reports" / "audio_mfcc_classifier_metrics.json"
MODEL_PATH = ROOT / "models" / "audio_mfcc_classifier.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a stable MFCC single-modality audio classifier.")
    parser.add_argument("--subset", default="dev-clean", choices=["dev-clean", "train-clean-100"])
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--top_k_speakers", type=int, default=3)
    parser.add_argument("--model", choices=["svm", "rf"], default="svm")
    args = parser.parse_args()

    items = collect_items(LIBRISPEECH / args.subset, args.max_samples, args.top_k_speakers)
    x = np.stack([extract_stats(path) for path, _ in items])
    y = np.array([label for _, label in items])
    train_x, test_x, train_y, test_y = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    if args.model == "svm":
        model = Pipeline([("scaler", StandardScaler()), ("clf", SVC(C=10, gamma="scale"))])
    else:
        model = RandomForestClassifier(n_estimators=240, random_state=42, class_weight="balanced")
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    metrics = {
        "model": f"MFCC statistics + {args.model.upper()}",
        "task": "LibriSpeech acoustic speaker-proxy classification",
        "note": "Used as the stable single-modality audio branch metric; LSTM script is also provided separately.",
        "subset": args.subset,
        "samples": len(items),
        "accuracy": round(float(accuracy_score(test_y, pred_y)), 4),
        "classification_report": classification_report(test_y, pred_y, output_dict=True, zero_division=0),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def collect_items(subset_dir: Path, max_samples: int, top_k_speakers: int) -> list[tuple[Path, str]]:
    flacs = sorted(subset_dir.rglob("*.flac"))
    speaker_counts = Counter(path.parts[-3] for path in flacs)
    speakers = [speaker for speaker, _ in speaker_counts.most_common(top_k_speakers)]
    by_speaker = defaultdict(list)
    for path in flacs:
        speaker = path.parts[-3]
        if speaker in speakers:
            by_speaker[speaker].append((path, speaker))
    per_speaker = max(1, max_samples // len(speakers))
    items = []
    for speaker in speakers:
        items.extend(by_speaker[speaker][:per_speaker])
    return items


def extract_stats(path: Path, sr: int = 16000) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, duration=4.0)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    feats = np.vstack([mfcc, delta, zcr, centroid])
    return np.concatenate([feats.mean(axis=1), feats.std(axis=1), feats.min(axis=1), feats.max(axis=1)])


if __name__ == "__main__":
    main()
