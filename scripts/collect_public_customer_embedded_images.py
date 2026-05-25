from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from PIL import Image, ImageStat

from collect_public_customer_page_screenshots import SOURCES, slug


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "datasets" / "images" / "customer_screenshots"
MANIFEST = OUT_DIR / "public_embedded_image_manifest.json"
IMAGE_URL_PATTERN = re.compile(
    r"""(?:src|data-src|data-lazy-src|href)=["']([^"']+\.(?:png|jpg|jpeg|webp)(?:\?[^"']*)?)["']""",
    flags=re.IGNORECASE,
)
SRCSET_PATTERN = re.compile(r"""(?:srcset|data-srcset)=["']([^"']+)["']""", flags=re.IGNORECASE)
ABSOLUTE_IMAGE_PATTERN = re.compile(r"""https?://[^"' <>)]+\.(?:png|jpg|jpeg|webp)(?:\?[^"' <>)]+)?""", flags=re.IGNORECASE)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download public embedded UI/business images from traceable customer-service source pages."
    )
    parser.add_argument("--max_per_source", type=int, default=6)
    parser.add_argument("--min_width", type=int, default=320)
    parser.add_argument("--min_height", type=int, default=160)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    jobs = [
        (source_index, label, title, url, args.max_per_source, args.min_width, args.min_height)
        for source_index, (label, title, url) in enumerate(SOURCES, start=1)
    ]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_source, job) for job in jobs]
        for future in as_completed(futures):
            source_rows = future.result()
            rows.extend(source_rows)
            ok = sum(1 for row in source_rows if row["status"] in {"ok", "existing"})
            title = source_rows[0]["title"] if source_rows else "unknown"
            print(f"{title}: saved={ok} candidates={len(source_rows)}", flush=True)

    rows.sort(key=lambda row: (row["label"], row["source_index"], row.get("image_index", 0), row.get("image_url", "")))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "manifest": str(MANIFEST),
        "saved": sum(1 for row in rows if row["status"] in {"ok", "existing"}),
        "total_rows": len(rows),
        "label_counts": label_counts(rows),
        "strict_note": (
            "Images are downloaded from public source pages already used by the screenshot route. "
            "They are traceable public embedded images, not generated screenshots. They should still be "
            "reviewed because some public pages use marketing images rather than business UI screenshots."
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def process_source(job: tuple[int, str, str, str, int, int, int]) -> list[dict]:
    source_index, label, title, page_url, max_per_source, min_width, min_height = job
    rows: list[dict] = []
    try:
        response = requests.get(page_url, headers=HEADERS, timeout=25)
        response.raise_for_status()
    except Exception as exc:
        return [
            base_row(source_index, label, title, page_url)
            | {"status": "failed_page_fetch", "error": str(exc)}
        ]

    image_urls = extract_image_urls(response.text, page_url)
    saved = 0
    for image_index, image_url in enumerate(image_urls, start=1):
        if saved >= max_per_source:
            break
        row = base_row(source_index, label, title, page_url) | {
            "image_index": image_index,
            "image_url": image_url,
            "usage": "public_customer_service_embedded_image",
            "strict_note": (
                "Public image embedded in the listed source page. This is traceable public data, "
                "but it must not be described as a private real customer case."
            ),
        }
        try:
            image_response = requests.get(image_url, headers=HEADERS | {"Referer": page_url}, timeout=25)
            image_response.raise_for_status()
            image = Image.open(BytesIO(image_response.content)).convert("RGB")
            quality = inspect_image(image)
            row |= quality
            if quality["width"] < min_width or quality["height"] < min_height:
                row["status"] = "skipped_small"
            elif quality["edge_score"] < 7.0 and quality["contrast"] < 18.0:
                row["status"] = "skipped_low_visual_detail"
            elif looks_like_logo_or_icon(image_url, quality):
                row["status"] = "skipped_logo_or_icon"
            else:
                target = target_path(source_index, label, title, image_index)
                if target.exists():
                    row["status"] = "existing"
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    image.save(target)
                    row["status"] = "ok"
                row["file"] = str(target)
                saved += 1
        except Exception as exc:
            row["status"] = "failed_image_fetch"
            row["error"] = str(exc)
        rows.append(row)
    if not rows:
        rows.append(base_row(source_index, label, title, page_url) | {"status": "no_image_candidates"})
    return rows


def extract_image_urls(html: str, page_url: str) -> list[str]:
    urls: list[str] = []
    for match in IMAGE_URL_PATTERN.finditer(html):
        urls.append(match.group(1))
    for match in SRCSET_PATTERN.finditer(html):
        for candidate in match.group(1).split(","):
            url = candidate.strip().split(" ")[0]
            if url:
                urls.append(url)
    urls.extend(match.group(0) for match in ABSOLUTE_IMAGE_PATTERN.finditer(html))

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        if not raw or raw.startswith("data:"):
            continue
        url = urljoin(page_url, raw.replace("\\/", "/"))
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        clean = parsed._replace(fragment="").geturl()
        lower_path = parsed.path.lower()
        if any(token in lower_path for token in ["/icon", "logo", "favicon", "sprite", "avatar"]):
            continue
        if clean not in seen:
            seen.add(clean)
            normalized.append(clean)
    return normalized


def inspect_image(image: Image.Image) -> dict:
    gray = image.convert("L")
    arr = np.asarray(gray.resize((128, 128)), dtype=np.float32)
    edge_score = float(np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean())
    stat = ImageStat.Stat(gray)
    return {
        "width": int(image.width),
        "height": int(image.height),
        "contrast": round(float(stat.stddev[0]), 4),
        "edge_score": round(edge_score, 4),
    }


def looks_like_logo_or_icon(image_url: str, quality: dict) -> bool:
    path = urlparse(image_url).path.lower()
    width = int(quality["width"])
    height = int(quality["height"])
    if max(width, height) <= 256:
        return True
    if "logo" in path or "icon" in path or "favicon" in path:
        return True
    ratio = width / max(1, height)
    return ratio > 8.0 or ratio < 0.12


def target_path(source_index: int, label: str, title: str, image_index: int) -> Path:
    return OUT_DIR / label / f"public_asset_{source_index:03d}_{image_index:02d}_{slug(title)}.png"


def base_row(source_index: int, label: str, title: str, page_url: str) -> dict:
    return {
        "source_index": source_index,
        "label": label,
        "title": title,
        "page_url": page_url,
    }


def label_counts(rows: list[dict]) -> dict[str, int]:
    counts = {label: 0 for label in sorted({row["label"] for row in rows})}
    for row in rows:
        if row["status"] in {"ok", "existing"}:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
    return counts


if __name__ == "__main__":
    main()
