from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "demo_assets" / "customer_screenshots"
REPORT_PATH = ROOT / "reports" / "demo_screenshot_dataset.json"

LABELS = ["fault", "order", "refund", "howto", "normal"]
DISPLAY = {
    "fault": "故障报修",
    "order": "订单咨询",
    "refund": "退款售后",
    "howto": "操作指引",
    "normal": "普通咨询",
}
PALETTES = {
    "fault": {"main": (220, 38, 38), "soft": (254, 226, 226), "dark": (127, 29, 29)},
    "order": {"main": (37, 99, 235), "soft": (219, 234, 254), "dark": (30, 64, 175)},
    "refund": {"main": (217, 119, 6), "soft": (254, 243, 199), "dark": (146, 64, 14)},
    "howto": {"main": (13, 148, 136), "soft": (204, 251, 241), "dark": (15, 118, 110)},
    "normal": {"main": (75, 85, 99), "soft": (243, 244, 246), "dark": (31, 41, 55)},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo-only customer-service screenshots for course presentation.")
    parser.add_argument("--train_per_label", type=int, default=160)
    parser.add_argument("--test_per_label", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    random.seed(args.seed)
    rows = []
    for split, count in [("train", args.train_per_label), ("test", args.test_per_label)]:
        for label in LABELS:
            target_dir = OUT_DIR / split / label
            target_dir.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                image = make_screenshot(label, seed=args.seed + index + (0 if split == "train" else 10000))
                path = target_dir / f"{label}_{index + 1:04d}.png"
                image.save(path)
                rows.append({"split": split, "label": label, "path": str(path)})

    report = {
        "dataset": str(OUT_DIR),
        "labels": LABELS,
        "train_per_label": args.train_per_label,
        "test_per_label": args.test_per_label,
        "total": len(rows),
        "strict_sources": False,
        "usage": "course_demo_only",
        "note": (
            "These screenshots are generated demo assets for presentation testing. "
            "They must not be counted as strict public-data compliance."
        ),
        "rows": rows[:20],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def make_screenshot(label: str, seed: int) -> Image.Image:
    rng = random.Random(f"{label}-{seed}")
    width, height = 1280, 720
    palette = PALETTES[label]
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    fonts = load_fonts()

    draw_header(draw, width, palette, fonts, label)
    draw_sidebar(draw, height, palette, fonts, active=label)
    draw_card(draw, (250, 100, 1210, 655), fill=(255, 255, 255), outline=(226, 232, 240))

    if label == "fault":
        draw_fault(draw, rng, palette, fonts)
    elif label == "order":
        draw_order(draw, rng, palette, fonts)
    elif label == "refund":
        draw_refund(draw, rng, palette, fonts)
    elif label == "howto":
        draw_howto(draw, rng, palette, fonts)
    else:
        draw_normal(draw, rng, palette, fonts)

    add_noise_lines(draw, rng)
    return image


def load_fonts() -> dict[str, ImageFont.ImageFont]:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)

    def font(size: int) -> ImageFont.ImageFont:
        if font_path:
            return ImageFont.truetype(str(font_path), size)
        return ImageFont.load_default()

    return {
        "xs": font(18),
        "sm": font(22),
        "md": font(28),
        "lg": font(36),
        "xl": font(48),
    }


def draw_header(draw: ImageDraw.ImageDraw, width: int, palette: dict, fonts: dict, label: str) -> None:
    draw.rectangle((0, 0, width, 72), fill=(17, 24, 39))
    draw.text((34, 18), "智能客服工作台", fill=(255, 255, 255), font=fonts["lg"])
    draw.rounded_rectangle((920, 18, 1210, 54), radius=12, fill=palette["main"])
    draw.text((946, 23), f"当前场景：{DISPLAY[label]}", fill=(255, 255, 255), font=fonts["sm"])


def draw_sidebar(draw: ImageDraw.ImageDraw, height: int, palette: dict, fonts: dict, active: str) -> None:
    draw.rectangle((0, 72, 220, height), fill=(241, 245, 249))
    items = [("order", "订单查询"), ("refund", "退款售后"), ("fault", "故障报修"), ("howto", "操作指南"), ("normal", "咨询会话")]
    y = 112
    for key, text in items:
        fill = palette["soft"] if key == active else (255, 255, 255)
        outline = palette["main"] if key == active else (226, 232, 240)
        draw.rounded_rectangle((24, y, 196, y + 52), radius=12, fill=fill, outline=outline, width=2)
        draw.text((48, y + 13), text, fill=palette["dark"] if key == active else (71, 85, 105), font=fonts["sm"])
        y += 72


def draw_fault(draw: ImageDraw.ImageDraw, rng: random.Random, palette: dict, fonts: dict) -> None:
    draw.text((285, 125), "页面异常 / 支付失败 / 系统报错", fill=palette["dark"], font=fonts["xl"])
    draw_card(draw, (285, 200, 1175, 330), fill=palette["soft"], outline=palette["main"])
    code = rng.choice(["ERR-503", "PAY-402", "UPLOAD-500", "AUTH-401"])
    draw.text((325, 228), "系统检测到异常", fill=palette["dark"], font=fonts["lg"])
    draw.text((325, 278), f"错误码：{code}    建议：提交故障报修并转人工排查", fill=(127, 29, 29), font=fonts["md"])
    draw_progress(draw, 315, 385, ["提交问题", "上传截图", "技术排查", "结果反馈"], active=2, palette=palette, fonts=fonts)
    draw_table(draw, (300, 475), ["模块", "状态", "说明"], [["支付接口", "失败", "回调超时"], ["订单同步", "异常", "状态未更新"], ["图片上传", "失败", "返回 500"]], fonts, palette)


def draw_order(draw: ImageDraw.ImageDraw, rng: random.Random, palette: dict, fonts: dict) -> None:
    order_no = f"DD{rng.randint(20260000, 20269999)}"
    draw.text((285, 125), "订单详情 / 物流状态 / 交易信息", fill=palette["dark"], font=fonts["xl"])
    draw_card(draw, (285, 195, 1175, 310), fill=palette["soft"], outline=palette["main"])
    draw.text((325, 222), f"订单号：{order_no}", fill=palette["dark"], font=fonts["lg"])
    draw.text((325, 270), f"收货状态：{rng.choice(['已发货', '运输中', '待揽收', '派送中'])}", fill=(30, 64, 175), font=fonts["md"])
    draw_progress(draw, 315, 365, ["已下单", "已付款", "仓库处理", "物流运输", "签收"], active=rng.randint(2, 4), palette=palette, fonts=fonts)
    rows = [["商品名称", "数量", "金额"], ["智能耳机", "1", "299.00"], ["保护壳", "2", "58.00"], ["运费", "1", "0.00"]]
    draw_table(draw, (300, 465), rows[0], rows[1:], fonts, palette)


def draw_refund(draw: ImageDraw.ImageDraw, rng: random.Random, palette: dict, fonts: dict) -> None:
    draw.text((285, 125), "退款申请 / 售后审核 / 原路退回", fill=palette["dark"], font=fonts["xl"])
    draw_card(draw, (285, 195, 1175, 335), fill=palette["soft"], outline=palette["main"])
    amount = rng.choice(["¥59.90", "¥128.00", "¥299.00", "¥468.50"])
    status = rng.choice(["退款审核中", "等待商家处理", "退款失败需人工", "已提交售后"])
    draw.text((325, 225), f"退款金额：{amount}", fill=palette["dark"], font=fonts["lg"])
    draw.rounded_rectangle((855, 220, 1090, 270), radius=18, fill=palette["main"])
    draw.text((885, 230), status, fill=(255, 255, 255), font=fonts["sm"])
    draw_progress(draw, 315, 380, ["申请退款", "审核中", "退回账户", "完成"], active=rng.randint(1, 2), palette=palette, fonts=fonts)
    draw_table(draw, (300, 475), ["售后项", "状态", "说明"], [["退款原因", "已填写", "商品未收到"], ["凭证截图", "已上传", "订单/支付截图"], ["处理方式", "待确认", "原路退回"]], fonts, palette)


def draw_howto(draw: ImageDraw.ImageDraw, rng: random.Random, palette: dict, fonts: dict) -> None:
    draw.text((285, 125), "操作指引 / 步骤说明 / 新手帮助", fill=palette["dark"], font=fonts["xl"])
    steps = rng.sample(["进入个人中心", "点击订单管理", "选择修改地址", "上传问题截图", "保存并提交"], 4)
    y = 205
    for index, step in enumerate(steps, start=1):
        draw_card(draw, (305, y, 1120, y + 72), fill=(255, 255, 255), outline=palette["main"])
        draw.ellipse((325, y + 16, 365, y + 56), fill=palette["main"])
        draw.text((338, y + 20), str(index), fill=(255, 255, 255), font=fonts["sm"])
        draw.text((390, y + 19), step, fill=palette["dark"], font=fonts["md"])
        y += 92
    draw.rounded_rectangle((835, 560, 1120, 615), radius=16, fill=palette["soft"], outline=palette["main"], width=2)
    draw.text((865, 573), "需要帮助？查看操作教程", fill=palette["dark"], font=fonts["sm"])


def draw_normal(draw: ImageDraw.ImageDraw, rng: random.Random, palette: dict, fonts: dict) -> None:
    draw.text((285, 125), "客服咨询 / 常见问题 / 在线会话", fill=palette["dark"], font=fonts["xl"])
    bubbles = [
        ("用户", rng.choice(["活动什么时候结束？", "可以开发票吗？", "会员权益怎么查看？"])),
        ("客服", "您好，请问需要咨询哪一类问题？"),
        ("用户", "我想了解一下服务说明和办理流程。"),
    ]
    y = 205
    for speaker, text in bubbles:
        x1, x2 = (315, 830) if speaker == "用户" else (610, 1135)
        fill = palette["soft"] if speaker == "用户" else (255, 255, 255)
        draw_card(draw, (x1, y, x2, y + 78), fill=fill, outline=(203, 213, 225))
        draw.text((x1 + 24, y + 22), f"{speaker}：{text}", fill=palette["dark"], font=fonts["sm"])
        y += 100
    draw_table(draw, (300, 520), ["咨询主题", "状态", "建议入口"], [["会员权益", "可咨询", "账户中心"], ["发票申请", "可办理", "订单详情"], ["活动规则", "可查看", "帮助中心"]], fonts, palette)


def draw_progress(draw: ImageDraw.ImageDraw, x: int, y: int, steps: list[str], active: int, palette: dict, fonts: dict) -> None:
    gap = 190
    for index, step in enumerate(steps):
        cx = x + index * gap
        fill = palette["main"] if index <= active else (203, 213, 225)
        draw.line((cx + 18, y + 18, cx + gap - 18, y + 18), fill=(203, 213, 225), width=4)
        draw.ellipse((cx, y, cx + 36, y + 36), fill=fill)
        draw.text((cx + 50, y + 4), step, fill=palette["dark"], font=fonts["xs"])


def draw_table(draw: ImageDraw.ImageDraw, origin: tuple[int, int], header: list[str], rows: list[list[str]], fonts: dict, palette: dict) -> None:
    x, y = origin
    col_w = [260, 180, 360]
    row_h = 46
    draw.rounded_rectangle((x, y, x + sum(col_w), y + row_h * (len(rows) + 1)), radius=12, fill=(255, 255, 255), outline=(203, 213, 225))
    draw.rectangle((x, y, x + sum(col_w), y + row_h), fill=palette["soft"])
    cursor = x
    for index, text in enumerate(header):
        draw.text((cursor + 16, y + 10), text, fill=palette["dark"], font=fonts["xs"])
        cursor += col_w[index]
    for row_index, row in enumerate(rows, start=1):
        cursor = x
        for col_index, text in enumerate(row):
            draw.text((cursor + 16, y + row_h * row_index + 10), text, fill=(51, 65, 85), font=fonts["xs"])
            cursor += col_w[col_index]


def draw_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int], outline: tuple[int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=2)


def add_noise_lines(draw: ImageDraw.ImageDraw, rng: random.Random) -> None:
    for _ in range(18):
        x = rng.randint(250, 1180)
        y = rng.randint(110, 650)
        color = rng.choice([(226, 232, 240), (241, 245, 249), (229, 231, 235)])
        draw.line((x, y, min(1210, x + rng.randint(20, 80)), y), fill=color, width=1)


if __name__ == "__main__":
    main()
