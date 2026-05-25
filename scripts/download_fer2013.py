from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "datasets" / "emotion" / "fer2013" / "fer2013.csv"
URL = "https://sourceforge.net/projects/emotion-detector/files/fer2013.csv/download"


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and TARGET.stat().st_size > 100_000_000:
        print(f"SKIP {TARGET} ({TARGET.stat().st_size / 1024 / 1024:.2f} MB)")
        return

    print(f"Downloading {URL}")
    urlretrieve(URL, TARGET)
    print(f"SAVED {TARGET} ({TARGET.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
