from __future__ import annotations

import json
import sys
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from customer_ai.audio_analyzer import analyze_audio
from customer_ai.emotion_analyzer import analyze_emotion_image
from customer_ai.fusion_model import FEATURE_DIM, FusionMLP, build_fusion_features, normalize_scores
from customer_ai.image_analyzer import analyze_image
from customer_ai.intent_engine import IntentEngine


DATA_PATH = ROOT / "data" / "demo" / "multimodal_samples.csv"
STRICT_DATA_PATH = ROOT / "datasets" / "multimodal" / "customer_cases.csv"
MODEL_PATH = ROOT / "models" / "fusion_intent_mlp.pt"
REPORT_PATH = ROOT / "reports" / "fusion_model_metrics.json"
EMOTIONS = ["neutral", "happy", "anxious", "angry", "sad"]
IMAGE_SIGNALS = ["normal", "error_screenshot", "low_info"]
INTENT_LABELS = ["consult", "complaint", "refund", "repair", "howto"]
STRICT_REQUIRED_COLUMNS = [
    "case_id",
    "label",
    "text",
    "voice_text",
    "audio_path",
    "emotion_label",
    "emotion_image_path",
    "screenshot_path",
    "source_dataset",
    "source_note",
]
FORBIDDEN_STRICT_SOURCE_MARKERS = ["demo", "synthetic", "template", "generated", "fake"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fully connected feature-level fusion model.")
    parser.add_argument(
        "--strict_sources",
        action="store_true",
        help="Require real multimodal cases from screenshot-allowed sources instead of demo rows.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--min_cases_per_label", type=int, default=10)
    args = parser.parse_args()

    data_path = STRICT_DATA_PATH if args.strict_sources else DATA_PATH
    if args.strict_sources and not data_path.exists():
        raise SystemExit(
            "Strict fusion training requires datasets/multimodal/customer_cases.csv. "
            "The demo CSV is not allowed for strict compliance."
        )
    df = pd.read_csv(data_path)
    if args.strict_sources:
        validate_strict_fusion_data(df, min_cases_per_label=args.min_cases_per_label)
    engine = IntentEngine()
    labels = INTENT_LABELS if args.strict_sources else sorted(df["label"].astype(str).unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    rows = df.to_dict(orient="records")
    x = np.stack([build_features(engine, row, strict_sources=args.strict_sources) for row in rows])
    y = np.asarray([label_to_id[str(row["label"])] for row in rows], dtype=np.int64)

    train_x, test_x, train_y, test_y = train_test_split(
        x,
        y,
        test_size=0.25 if args.strict_sources else 0.5,
        random_state=42,
        stratify=y,
    )
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_x), torch.tensor(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = FusionMLP(input_dim=FEATURE_DIM, hidden_dim=48, num_classes=len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(args.epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(test_x, dtype=torch.float32))
        pred_ids = logits.argmax(dim=1).numpy()

    test_labels = [labels[index] for index in test_y]
    pred_labels = [labels[index] for index in pred_ids]
    metrics = {
        "model": "PyTorch fully-connected feature fusion",
        "task": "multimodal customer-service intent fusion",
        "strict_sources": bool(args.strict_sources),
        "data": str(data_path),
        "source_contract": "real multimodal cases from strict sources" if args.strict_sources else "demo samples",
        "source_note": (
            "Strict mode requires real rows with existing audio_path and screenshot_path; "
            "demo/template/synthetic rows are rejected."
            if args.strict_sources
            else "Demo mode uses small handcrafted rows and synthetic modality scores."
        ),
        "samples": int(len(df)),
        "feature_count": int(x.shape[1]),
        "epochs": args.epochs,
        "accuracy": round(float(accuracy_score(test_labels, pred_labels)), 4),
        "classification_report": classification_report(
            test_labels,
            pred_labels,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": labels,
            "label_to_id": label_to_id,
            "input_dim": FEATURE_DIM,
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    REPORT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def validate_strict_fusion_data(df: pd.DataFrame, min_cases_per_label: int) -> None:
    missing_columns = [column for column in STRICT_REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise SystemExit(f"Strict fusion CSV is missing required columns: {', '.join(missing_columns)}")
    if df.empty:
        raise SystemExit(
            "Strict fusion training requires real multimodal cases. "
            "datasets/multimodal/customer_cases.csv currently has no rows."
        )

    label_counts = Counter(clean_value(row.get("label")) for _, row in df.iterrows())
    invalid_labels = sorted(label for label in label_counts if label not in INTENT_LABELS)
    if invalid_labels:
        raise SystemExit(f"Strict fusion CSV has labels outside the five required intents: {', '.join(invalid_labels)}")
    insufficient = [label for label in INTENT_LABELS if label_counts[label] < min_cases_per_label]
    if insufficient:
        details = ", ".join(f"{label}={label_counts[label]}" for label in INTENT_LABELS)
        raise SystemExit(
            "Strict fusion training requires at least "
            f"{min_cases_per_label} real multimodal cases per label. Current counts: {details}"
        )

    problems: list[str] = []
    for index, row in df.iterrows():
        row_id = clean_value(row.get("case_id")) or f"row_{index + 1}"
        source_text = f"{clean_value(row.get('source_dataset'))} {clean_value(row.get('source_note'))}".strip().lower()
        if not source_text:
            problems.append(f"{row_id}: source_dataset/source_note is required")
        if any(marker in source_text for marker in FORBIDDEN_STRICT_SOURCE_MARKERS):
            problems.append(f"{row_id}: source notes look like demo/synthetic/template data")
        if not (clean_value(row.get("text")) or clean_value(row.get("voice_text"))):
            problems.append(f"{row_id}: text or voice_text is required")
        if not existing_path(row.get("audio_path")):
            problems.append(f"{row_id}: audio_path must point to an existing real audio file")
        if not existing_path(row.get("screenshot_path")):
            problems.append(f"{row_id}: screenshot_path must point to an existing customer-service screenshot")
        emotion_label = clean_value(row.get("emotion_label"))
        emotion_image = existing_path(row.get("emotion_image_path"))
        if not emotion_label and not emotion_image:
            problems.append(f"{row_id}: emotion_label or emotion_image_path is required")

    if problems:
        preview = "\n".join(problems[:20])
        suffix = "" if len(problems) <= 20 else f"\n... and {len(problems) - 20} more"
        raise SystemExit(f"Strict fusion CSV is not valid real multimodal training data:\n{preview}{suffix}")


def build_features(engine: IntentEngine, row: dict, strict_sources: bool) -> np.ndarray:
    if strict_sources:
        return build_strict_features(engine, row)
    return build_demo_features(engine, row)


def build_demo_features(engine: IntentEngine, row: dict) -> np.ndarray:
    text = clean_value(row.get("text"))
    voice = clean_value(row.get("voice_text"))
    emotion = clean_value(row.get("emotion"))
    image_signal = clean_value(row.get("image_signal"))
    merged_text = f"{text} {voice}".strip()
    baseline_scores = engine._model_scores(merged_text)
    keyword_scores = engine._keyword_scores(merged_text)
    text_scores = normalize_scores(
        {label: baseline_scores.get(label, 0.0) * 0.78 + keyword_scores.get(label, 0.0) * 0.22 for label in engine.labels}
    )
    audio_scores = synthetic_audio_scores(voice)
    emotion_scores = synthetic_emotion_scores(emotion)
    image_scores = synthetic_image_scores(image_signal)
    numeric = [
        min(len(merged_text) / 80.0, 1.0),
        1.0 if text else 0.0,
        1.0 if voice else 0.0,
        0.72,
        0.75 if emotion != "neutral" else 0.2,
        0.82 if image_signal == "error_screenshot" else 0.62,
    ]
    return build_fusion_features(text_scores, audio_scores, emotion_scores, image_scores, numeric)


def build_strict_features(engine: IntentEngine, row: dict) -> np.ndarray:
    text = clean_value(row.get("text"))
    voice = clean_value(row.get("voice_text"))
    merged_text = f"{text} {voice}".strip()
    baseline_scores = engine._model_scores(merged_text)
    keyword_scores = engine._keyword_scores(merged_text)
    text_scores = normalize_scores(
        {label: baseline_scores.get(label, 0.0) * 0.78 + keyword_scores.get(label, 0.0) * 0.22 for label in engine.labels}
    )

    audio = analyze_audio(existing_path(row.get("audio_path")))
    emotion_image = analyze_emotion_image(existing_path(row.get("emotion_image_path")))
    image = analyze_image(existing_path(row.get("screenshot_path")))
    emotion_label = clean_value(row.get("emotion_label")) or (emotion_image.emotion if emotion_image else "neutral")

    audio_scores = normalize_scores(audio.intent_scores if audio else {})
    emotion_scores = normalize_scores(
        emotion_image.intent_scores if emotion_image else synthetic_emotion_scores(emotion_label)
    )
    image_scores = normalize_scores(image.intent_scores if image else {})
    numeric = [
        min(len(merged_text) / 80.0, 1.0),
        1.0 if text else 0.0,
        1.0 if voice else 0.0,
        float(audio.confidence) if audio else 0.0,
        float(emotion_image.confidence) if emotion_image else (0.75 if emotion_label != "neutral" else 0.2),
        float(image.confidence) if image else 0.0,
    ]
    return build_fusion_features(text_scores, audio_scores, emotion_scores, image_scores, numeric)


def clean_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def existing_path(value) -> Path | None:
    raw = clean_value(value)
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path if path.exists() else None


def synthetic_audio_scores(voice_text: str) -> dict[str, float]:
    scores = {label: 0.20 for label in ["consult", "complaint", "refund", "repair", "howto"]}
    if any(word in voice_text for word in ["生气", "人工", "没人"]):
        scores["complaint"] += 0.28
    if any(word in voice_text for word in ["取消", "退款", "不到账"]):
        scores["refund"] += 0.24
    if any(word in voice_text for word in ["错误", "着急", "截图"]):
        scores["repair"] += 0.24
    if any(word in voice_text for word in ["教", "不会", "操作"]):
        scores["howto"] += 0.24
    if any(word in voice_text for word in ["查", "咨询", "活动"]):
        scores["consult"] += 0.24
    return normalize_scores(scores)


def synthetic_emotion_scores(emotion: str) -> dict[str, float]:
    scores = {label: 0.20 for label in ["consult", "complaint", "refund", "repair", "howto"]}
    if emotion in {"angry", "sad"}:
        scores["complaint"] += 0.34
        scores["refund"] += 0.16
    elif emotion == "anxious":
        scores["repair"] += 0.30
        scores["howto"] += 0.16
    elif emotion == "happy":
        scores["consult"] += 0.18
    return normalize_scores(scores)


def synthetic_image_scores(image_signal: str) -> dict[str, float]:
    scores = {label: 0.20 for label in ["consult", "complaint", "refund", "repair", "howto"]}
    if image_signal == "error_screenshot":
        scores["repair"] += 0.42
        scores["howto"] += 0.12
    elif image_signal == "low_info":
        scores["howto"] += 0.24
        scores["repair"] += 0.12
    else:
        scores["consult"] += 0.08
    return normalize_scores(scores)


if __name__ == "__main__":
    main()
