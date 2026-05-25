from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "demo" / "text_intent_samples.csv"
MODEL_PATH = ROOT / "models" / "text_intent_tfidf_lr.joblib"
REPORT_PATH = ROOT / "reports" / "text_intent_metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train text intent baseline model.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="CSV with text,label columns.")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain text,label columns")

    train_x, test_x, train_y, test_y = train_test_split(
        df["text"].astype(str),
        df["label"].astype(str),
        test_size=0.28,
        random_state=42,
        stratify=df["label"].astype(str),
    )
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
            ("clf", LogisticRegression(max_iter=800, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    metrics = {
        "model": "TF-IDF(char 2-4gram) + LogisticRegression",
        "data": str(data_path),
        "train_size": int(len(train_x)),
        "test_size": int(len(test_x)),
        "accuracy": round(float(accuracy_score(test_y, pred_y)), 4),
        "classification_report": classification_report(test_y, pred_y, output_dict=True, zero_division=0),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
