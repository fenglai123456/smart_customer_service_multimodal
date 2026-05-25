from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ABCD_IMAGES = ROOT / "datasets" / "text" / "customer_dialogue_public" / "abcd" / "data" / "images"
OUT_DIR = ROOT / "datasets" / "images" / "customer_screenshots"
MANIFEST = OUT_DIR / "manifest.csv"
REPORT = ROOT / "reports" / "customer_screenshot_seed_manifest.json"


SEEDS = [
    {
        "source_file": "agent_dashboard.png",
        "label": "refund",
        "reason": "ABCD agent dashboard includes an explicit Offer Refund action in a customer-service workflow.",
    },
    {
        "source_file": "action_example.png",
        "label": "order",
        "reason": "ABCD action example is a customer-service action/order-management interface example.",
    },
    {
        "source_file": "customer_site.png",
        "label": "normal",
        "reason": "ABCD customer site screenshot shows the normal customer-facing shopping/support context.",
    },
    {
        "source_file": "faq_screenshot.png",
        "label": "howto",
        "reason": "ABCD FAQ/policy screenshot is suitable for operation guidance/how-to intent examples.",
    },
]


def main() -> None:
    rows = []
    for seed in SEEDS:
        source = ABCD_IMAGES / seed["source_file"]
        if not source.exists():
            raise SystemExit(f"Missing ABCD seed image: {source}")
        target_dir = OUT_DIR / seed["label"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"abcd_{seed['source_file']}"
        shutil.copy2(source, target)
        rows.append(
            {
                "path": str(target),
                "label": seed["label"],
                "source_dataset": "ABCD",
                "source_url": "https://github.com/asappresearch/abcd",
                "source_file": str(source),
                "usage": "public_customer_screenshot_seed",
                "reason": seed["reason"],
            }
        )

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "label", "source_dataset", "source_url", "source_file", "usage", "reason"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source": "ABCD public customer-service dataset images",
        "source_url": "https://github.com/asappresearch/abcd",
        "manifest": str(MANIFEST),
        "seed_count": len(rows),
        "label_counts": {label: sum(1 for row in rows if row["label"] == label) for label in sorted({r["label"] for r in rows})},
        "strict_note": (
            "These are real public customer-service dataset screenshots, but they are seed samples only. "
            "They are not enough for strict 82% image-model training."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
