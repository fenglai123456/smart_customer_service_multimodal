from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "datasets" / "text" / "squad"
FILES = {
    "train-v2.0.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
    "dev-v2.0.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
}


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        target = TARGET_DIR / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"SKIP {target}")
            continue
        print(f"Downloading {url}")
        urlretrieve(url, target)
        print(f"SAVED {target} ({target.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
