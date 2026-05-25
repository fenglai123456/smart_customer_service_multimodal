from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_text_intent_augmented import build_dataset as build_augmented_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "demo" / "text_intent_samples.csv"
STRICT_DATA = ROOT / "datasets" / "text" / "strict_text_intents.csv"
SQUAD_TRAIN = ROOT / "datasets" / "text" / "squad" / "train-v2.0.json"
SQUAD_DEV = ROOT / "datasets" / "text" / "squad" / "dev-v2.0.json"
REPORT_PATH = ROOT / "reports" / "text_bert_metrics.json"
MODEL_DIR = ROOT / "models" / "text_bert_classifier"


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_to_id, max_length: int):
        self.texts = list(texts)
        self.labels = [label_to_id[label] for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        item = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": item["input_ids"].squeeze(0),
            "attention_mask": item["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a BERT-style text intent classifier.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--model_name", default="bert-base-chinese")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--augmented", action="store_true", help="Use generated customer-service intent samples.")
    parser.add_argument(
        "--strict_sources",
        action="store_true",
        help="Train only on screenshot-allowed text sources: SQuAD + public customer-service dialogue subset.",
    )
    parser.add_argument("--freeze_encoder", action="store_true", help="Train only the classification head.")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Use a tiny random BERT model for a quick local run.")
    args = parser.parse_args()
    set_seed(args.seed)

    if args.smoke:
        args.model_name = "hf-internal-testing/tiny-random-bert"
        args.epochs = min(args.epochs, 1)
        args.max_samples = args.max_samples or 30

    if args.strict_sources:
        validate_strict_text_sources()
        df = pd.read_csv(STRICT_DATA)
    elif args.augmented:
        texts, labels_for_texts = build_augmented_dataset()
        df = pd.DataFrame({"text": texts, "label": labels_for_texts})
    else:
        df = pd.read_csv(args.data)

    if args.max_samples:
        df = df.groupby("label", group_keys=False).head(max(1, args.max_samples // df["label"].nunique()))

    labels = sorted(df["label"].astype(str).unique())
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    train_x, test_x, train_y, test_y = train_test_split(
        df["text"].astype(str),
        df["label"].astype(str),
        test_size=0.28,
        random_state=42,
        stratify=df["label"].astype(str),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )
    if args.freeze_encoder:
        for name, parameter in model.named_parameters():
            if not name.startswith("classifier"):
                parameter.requires_grad = False

    train_loader = DataLoader(
        TextDataset(train_x, train_y, tokenizer, label_to_id, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TextDataset(test_x, test_y, tokenizer, label_to_id, args.max_length),
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1
        print(f"epoch={epoch + 1} loss={running_loss / max(1, batches):.4f}")

    y_true, y_pred = evaluate(model, test_loader, device, id_to_label)
    metrics = {
        "model": args.model_name,
        "task": "BERT text intent classification",
        "mode": "smoke" if args.smoke else "full",
        "data": "strict_squad_plus_abcd" if args.strict_sources else ("augmented" if args.augmented else str(args.data)),
        "strict_sources": bool(args.strict_sources),
        "source_contract": "SQuAD v2.0 + ABCD public customer-dialogue subset" if args.strict_sources else "demo/augmented",
        "train_size": len(train_x),
        "test_size": len(test_x),
        "epochs": args.epochs,
        "freeze_encoder": args.freeze_encoder,
        "learning_rate": args.learning_rate,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    REPORT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_strict_text_sources() -> None:
    missing = [path for path in [SQUAD_TRAIN, SQUAD_DEV, STRICT_DATA] if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise SystemExit(
            "Strict text training requires SQuAD and the mapped ABCD customer-dialogue subset. "
            f"Missing:\n{joined}\nRun scripts\\prepare_abcd_customer_intents.py after downloading ABCD."
        )


def evaluate(model, loader, device, id_to_label):
    model.eval()
    truth, pred = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.cpu()
            truth.extend(id_to_label[int(item)] for item in labels)
            pred.extend(id_to_label[int(item)] for item in logits.argmax(dim=1))
    return truth, pred


if __name__ == "__main__":
    main()
