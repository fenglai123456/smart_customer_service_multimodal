from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ABCD_PATH = ROOT / "datasets" / "text" / "customer_dialogue_public" / "abcd" / "data" / "abcd_v1.1.json.gz"
SQUAD_TRAIN = ROOT / "datasets" / "text" / "squad" / "train-v2.0.json"
SQUAD_DEV = ROOT / "datasets" / "text" / "squad" / "dev-v2.0.json"
OUT_CSV = ROOT / "datasets" / "text" / "customer_dialogue_public" / "abcd_customer_intents.csv"
OUT_META = ROOT / "reports" / "abcd_customer_intents_metadata.json"


LABELS = {"consult", "complaint", "refund", "repair", "howto"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a strict customer-dialogue intent dataset from the public ABCD dataset."
    )
    parser.add_argument("--input", default=str(ABCD_PATH))
    parser.add_argument("--output", default=str(OUT_CSV))
    parser.add_argument("--metadata", default=str(OUT_META))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata)
    require_file(input_path, "ABCD customer-dialogue public subset")
    require_file(SQUAD_TRAIN, "SQuAD train-v2.0.json")
    require_file(SQUAD_DEV, "SQuAD dev-v2.0.json")

    with gzip.open(input_path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)

    rows: list[dict[str, str]] = []
    skipped: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for split, conversations in data.items():
        for convo in conversations:
            flow = str(convo["scenario"]["flow"])
            subflow = str(convo["scenario"]["subflow"])
            label = map_abcd_to_project_intent(flow, subflow)
            if label is None:
                skipped[f"{flow}/{subflow}"] += 1
                continue
            text = build_customer_text(convo)
            if not text:
                skipped[f"{flow}/{subflow}"] += 1
                continue
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "source_dataset": "ABCD",
                    "source_split": split,
                    "convo_id": str(convo["convo_id"]),
                    "flow": flow,
                    "subflow": subflow,
                }
            )
            source_counts[f"{flow}/{subflow}->{label}"] += 1

    if not rows:
        raise SystemExit("No ABCD conversations could be mapped to the project intent labels.")
    labels = {row["label"] for row in rows}
    if labels != LABELS:
        missing = sorted(LABELS - labels)
        raise SystemExit(f"Prepared dataset is missing required labels: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text", "label", "source_dataset", "source_split", "convo_id", "flow", "subflow"],
        )
        writer.writeheader()
        writer.writerows(rows)

    label_counts = Counter(row["label"] for row in rows)
    split_counts = Counter(row["source_split"] for row in rows)
    metadata = {
        "source_contract": "SQuAD v2.0 + ABCD public customer-dialogue subset",
        "squad_train": str(SQUAD_TRAIN),
        "squad_dev": str(SQUAD_DEV),
        "abcd_source": str(input_path),
        "output_csv": str(output_path),
        "rows": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "mapping_counts": dict(sorted(source_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "note": (
            "ABCD has 55 customer-service subflows. This script maps those public subflows "
            "to the five project intents required by the screenshots."
        ),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def build_customer_text(convo: dict) -> str:
    customer_turns = [
        str(text).strip()
        for speaker, text in convo.get("original", [])
        if str(speaker).lower() == "customer" and str(text).strip()
    ]
    return " ".join(customer_turns)


def map_abcd_to_project_intent(flow: str, subflow: str) -> str | None:
    if flow in {"single_item_query", "storewide_query"}:
        return "consult"
    if flow == "troubleshoot_site":
        return "repair"
    if flow == "product_defect":
        return "refund"
    if flow == "account_access":
        return "howto"
    if flow == "manage_account":
        return "howto" if subflow.startswith("manage_") else "consult"
    if flow == "order_issue":
        if subflow == "manage_cancel":
            return "refund"
        if subflow.startswith("manage_"):
            return "howto"
        return "complaint" if subflow in {"status_mystery_fee", "status_quantity"} else "consult"
    if flow == "purchase_dispute":
        if subflow.startswith("mistimed_billing"):
            return "refund"
        return "complaint"
    if flow == "shipping_issue":
        if subflow in {"missing", "cost"}:
            return "complaint"
        if subflow == "manage":
            return "howto"
        return "consult"
    if flow == "subscription_inquiry":
        if subflow == "manage_dispute_bill":
            return "complaint"
        if subflow.startswith("manage_"):
            return "howto"
        return "consult"
    return None


if __name__ == "__main__":
    main()
