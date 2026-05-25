from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "demo_assets" / "customer_screenshots" / "test"
OUTPUT_PATH = ROOT / "demo_assets" / "customer_screenshots" / "demo_test_contact_sheet.png"

LABELS = [
    ("fault", "故障报修"),
    ("order", "订单咨询"),
    ("refund", "退款售后"),
    ("howto", "操作指引"),
    ("normal", "普通咨询"),
]


def load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    thumb_w, thumb_h = 260, 170
    label_h = 44
    padding = 18
    cols = 3
    rows = len(LABELS)
    width = padding * 2 + cols * thumb_w + (cols - 1) * padding
    height = padding * 2 + rows * (label_h + thumb_h) + (rows - 1) * padding

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    caption_font = load_font(18)

    y = padding
    for label, label_cn in LABELS:
        paths = sorted((DATA_DIR / label).glob("*.png"))[:cols]
        draw.text((padding, y + 8), f"{label_cn} / {label}", fill=(20, 24, 31), font=title_font)

        y_img = y + label_h
        for index, path in enumerate(paths):
            x = padding + index * (thumb_w + padding)
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_w, thumb_h - 30), Image.LANCZOS)

            frame = Image.new("RGB", (thumb_w, thumb_h), (245, 247, 250))
            frame.paste(image, ((thumb_w - image.width) // 2, 8))
            canvas.paste(frame, (x, y_img))
            draw.rectangle(
                [x, y_img, x + thumb_w - 1, y_img + thumb_h - 1],
                outline=(210, 216, 224),
                width=1,
            )
            draw.text((x + 10, y_img + thumb_h - 25), path.name, fill=(70, 78, 92), font=caption_font)

        y += label_h + thumb_h + padding

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_PATH)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
