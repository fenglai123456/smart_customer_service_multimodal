from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from collect_public_customer_page_screenshots import SOURCES, slug


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "datasets" / "images" / "customer_screenshots"
MANIFEST = OUT_DIR / "public_article_tile_manifest.json"
TMP_DIR = ROOT / "datasets" / "images" / "customer_screenshots" / "_tmp_full_pages"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture public customer-service source pages and crop article/body tiles for screenshot intent training."
    )
    parser.add_argument("--tiles_per_source", type=int, default=5)
    parser.add_argument("--tile_height", type=int, default=900)
    parser.add_argument("--viewport", default="1280,900")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--recapture", action="store_true")
    args = parser.parse_args()

    npx = shutil.which("npx.cmd") or shutil.which("npx")
    if not npx:
        raise SystemExit("npx is required to run Playwright screenshots.")

    jobs = [
        (npx, source_index, label, title, url, args.tiles_per_source, args.tile_height, args.viewport, args.recapture)
        for source_index, (label, title, url) in enumerate(SOURCES, start=1)
    ]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(capture_and_tile, job) for job in jobs]
        for future in as_completed(futures):
            source_rows = future.result()
            rows.extend(source_rows)
            ok = sum(1 for row in source_rows if row["status"] in {"ok", "existing"})
            title = source_rows[0]["title"] if source_rows else "unknown"
            print(f"{title}: tiles={ok}/{len(source_rows)}", flush=True)

    rows.sort(key=lambda row: (row["label"], row["source_index"], row.get("tile_index", 0)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "manifest": str(MANIFEST),
        "saved": sum(1 for row in rows if row["status"] in {"ok", "existing"}),
        "total_rows": len(rows),
        "label_counts": label_counts(rows),
        "strict_note": (
            "Tiles are cropped from full-page screenshots of traceable public customer-service source pages. "
            "Multiple tiles from one page must be grouped by source during evaluation and must not be described "
            "as independent real customer cases."
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def capture_and_tile(job: tuple[str, int, str, str, str, int, int, str, bool]) -> list[dict]:
    npx, source_index, label, title, url, tiles_per_source, tile_height, viewport, recapture = job
    target_dir = OUT_DIR / label
    target_dir.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    full_page = TMP_DIR / f"public_tile_{source_index:03d}_{slug(title)}_full.png"

    rows: list[dict] = []
    if recapture or not full_page.exists():
        command = [
            npx,
            "--yes",
            "playwright",
            "screenshot",
            "--browser",
            "chromium",
            "--viewport-size",
            viewport,
            "--full-page",
            "--wait-for-timeout",
            "1600",
            "--timeout",
            "35000",
            "--ignore-https-errors",
            "--user-agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            url,
            str(full_page),
        ]
        result = subprocess.run(command, cwd=ROOT, capture_output=True)
        if result.returncode != 0 or not full_page.exists():
            return [
                base_row(source_index, label, title, url)
                | {
                    "status": "failed_capture",
                    "stdout": result.stdout.decode("utf-8", errors="replace").strip()[-1000:],
                    "stderr": result.stderr.decode("utf-8", errors="replace").strip()[-1000:],
                }
            ]

    try:
        with Image.open(full_page) as image:
            page = image.convert("RGB")
            starts = tile_starts(page.height, tile_height, tiles_per_source)
            for tile_index, top in enumerate(starts, start=1):
                output = target_dir / f"public_tile_{source_index:03d}_{tile_index:02d}_{slug(title)}.png"
                row = base_row(source_index, label, title, url) | {
                    "tile_index": tile_index,
                    "top": top,
                    "height": tile_height,
                    "full_page": str(full_page),
                    "file": str(output),
                    "usage": "public_customer_service_article_tile",
                    "strict_note": (
                        "Public page body tile. Same-page tiles are grouped by source id during model evaluation."
                    ),
                }
                if output.exists():
                    row["status"] = "existing"
                else:
                    bottom = min(page.height, top + tile_height)
                    if bottom - top < min(320, tile_height):
                        row["status"] = "skipped_short_tile"
                    else:
                        page.crop((0, top, page.width, bottom)).save(output)
                        row["status"] = "ok"
                rows.append(row)
    except Exception as exc:
        rows.append(base_row(source_index, label, title, url) | {"status": "failed_crop", "error": str(exc)})
    return rows


def tile_starts(page_height: int, tile_height: int, count: int) -> list[int]:
    if page_height <= tile_height:
        return [0]
    # Skip the very top navigation/header when possible, then sample the body at regular intervals.
    start_min = min(260, max(0, page_height - tile_height))
    start_max = max(start_min, page_height - tile_height)
    if count <= 1 or start_max == start_min:
        return [start_min]
    step = (start_max - start_min) / max(1, count - 1)
    return sorted({int(round(start_min + index * step)) for index in range(count)})


def base_row(source_index: int, label: str, title: str, url: str) -> dict:
    return {"source_index": source_index, "label": label, "title": title, "url": url}


def label_counts(rows: list[dict]) -> dict[str, int]:
    counts = {label: 0 for label in sorted({row["label"] for row in rows})}
    for row in rows:
        if row["status"] in {"ok", "existing"}:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
    return counts


if __name__ == "__main__":
    main()
