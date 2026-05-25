from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "emotion" / "fer2013" / "fer2013.csv"
REPORT_PATH = ROOT / "reports" / "emotion_mlp_metrics.json"
MODEL_PATH = ROOT / "models" / "fer2013_mlp.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MLP baseline on FER-2013 raw pixels.")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--binary",
        choices=["none", "happy_vs_other", "negative_vs_other", "angry_vs_other"],
        default="none",
    )
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH, nrows=args.max_samples or None)
    x = np.stack([np.fromstring(pixels, sep=" ", dtype=np.float32) for pixels in df["pixels"]]) / 255.0
    y = df["emotion"].astype(int).to_numpy()
    task = "FER-2013 7-class facial expression classification"
    if args.binary == "happy_vs_other":
        y = (y == 3).astype(int)
        task = "FER-2013 happy-vs-other customer emotion classification"
    elif args.binary == "negative_vs_other":
        y = np.isin(y, [0, 1, 2, 4]).astype(int)
        task = "FER-2013 negative-vs-other customer emotion classification"
    elif args.binary == "angry_vs_other":
        y = (y == 0).astype(int)
        task = "FER-2013 angry-vs-other customer emotion classification"

    train_x, test_x, train_y, test_y = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=40,
                    random_state=42,
                    early_stopping=True,
                ),
            ),
        ]
    )
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    metrics = {
        "model": "raw pixels + MLP",
        "task": task,
        "mode": "sampled" if args.max_samples else "full",
        "samples": int(len(df)),
        "accuracy": round(float(accuracy_score(test_y, pred_y)), 4),
        "classification_report": classification_report(test_y, pred_y, output_dict=True, zero_division=0),
    }
    import joblib

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
