from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import Counter
from pathlib import Path

from prepare_abcd_customer_intents import build_customer_text, map_abcd_to_project_intent


ROOT = Path(__file__).resolve().parents[1]
ABCD_PATH = ROOT / "datasets" / "text" / "customer_dialogue_public" / "abcd" / "data" / "abcd_v1.1.json.gz"
SQUAD_TRAIN = ROOT / "datasets" / "text" / "squad" / "train-v2.0.json"
SQUAD_DEV = ROOT / "datasets" / "text" / "squad" / "dev-v2.0.json"
OUT_CSV = ROOT / "datasets" / "text" / "strict_text_intents.csv"
OUT_META = ROOT / "reports" / "strict_text_intents_metadata.json"

LABELS = {"consult", "complaint", "refund", "repair", "howto"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the strict text training CSV from SQuAD + public customer-service dialogue data."
    )
    parser.add_argument("--abcd", default=str(ABCD_PATH))
    parser.add_argument("--squad_train", default=str(SQUAD_TRAIN))
    parser.add_argument("--squad_dev", default=str(SQUAD_DEV))
    parser.add_argument("--output", default=str(OUT_CSV))
    parser.add_argument("--metadata", default=str(OUT_META))
    parser.add_argument(
        "--max_squad_questions",
        type=int,
        default=1200,
        help="Balanced preprocessing cap for SQuAD consult-style questions; use 0 to include all questions.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    abcd_path = Path(args.abcd)
    squad_train = Path(args.squad_train)
    squad_dev = Path(args.squad_dev)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata)
    for path, label in [
        (abcd_path, "ABCD public customer-dialogue subset"),
        (squad_train, "SQuAD train-v2.0.json"),
        (squad_dev, "SQuAD dev-v2.0.json"),
    ]:
        if not path.exists():
            raise SystemExit(f"Missing {label}: {path}")

    rows = build_abcd_rows(abcd_path)
    squad_rows = build_squad_rows(squad_train, "train") + build_squad_rows(squad_dev, "dev")
    if args.max_squad_questions and len(squad_rows) > args.max_squad_questions:
        rng = random.Random(args.seed)
        squad_rows = rng.sample(squad_rows, args.max_squad_questions)
    rows.extend(squad_rows)

    label_counts = Counter(row["label"] for row in rows)
    missing_labels = LABELS - set(label_counts)
    if missing_labels:
        raise SystemExit(f"Strict text dataset is missing labels: {sorted(missing_labels)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["text", "label", "source_dataset", "source_split", "source_id", "flow", "subflow"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "source_contract": "SQuAD v2.0 + ABCD public customer-dialogue subset",
        "output_csv": str(output_path),
        "rows": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "source_counts": dict(sorted(Counter(row["source_dataset"] for row in rows).items())),
        "max_squad_questions": args.max_squad_questions,
        "note": (
            "SQuAD questions are used as public QA/consult text. ABCD conversations provide "
            "the customer-service intent distribution mapped to the five screenshot-required intents."
        ),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def build_abcd_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    for split, conversations in data.items():
        for convo in conversations:
            flow = str(convo["scenario"]["flow"])
            subflow = str(convo["scenario"]["subflow"])
            label = map_abcd_to_project_intent(flow, subflow)
            if not label:
                continue
            text = build_customer_text(convo)
            if not text:
                continue
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "source_dataset": "ABCD",
                    "source_split": split,
                    "source_id": str(convo["convo_id"]),
                    "flow": flow,
                    "subflow": subflow,
                }
            )
    return rows


def build_squad_rows(path: Path, split: str) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for article in data.get("data", []):
        for paragraph in article.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                question = str(qa.get("question", "")).strip()
                if not question:
                    continue
                rows.append(
                    {
                        "text": question,
                        "label": "consult",
                        "source_dataset": "SQuAD",
                        "source_split": split,
                        "source_id": str(qa.get("id", "")),
                        "flow": "qa",
                        "subflow": "question_answering",
                    }
                )
    return rows


if __name__ == "__main__":
    main()
