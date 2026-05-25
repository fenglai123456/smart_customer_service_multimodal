from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "reports" / "report_assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for item in candidates:
        path = Path(item)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F_TITLE = font(42, True)
F_SUB = font(24)
F_BOX = font(25, True)
F_BODY = font(20)
F_SMALL = font(18)


def multiline_center(draw: ImageDraw.ImageDraw, box, text: str, fill, face, line_gap=8):
    x1, y1, x2, y2 = box
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=face)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + line_gap * (len(lines) - 1)
    y = y1 + ((y2 - y1) - total_h) / 2
    for line, w, h in zip(lines, widths, heights):
        draw.text((x1 + ((x2 - x1) - w) / 2, y), line, font=face, fill=fill)
        y += h + line_gap


def arrow(draw: ImageDraw.ImageDraw, start, end, color=(68, 84, 106), width=4):
    draw.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 >= x1 else -1
        points = [(x2, y2), (x2 - 18 * direction, y2 - 10), (x2 - 18 * direction, y2 + 10)]
    else:
        direction = 1 if y2 >= y1 else -1
        points = [(x2, y2), (x2 - 10, y2 - 18 * direction), (x2 + 10, y2 - 18 * direction)]
    draw.polygon(points, fill=color)


def rounded_box(draw, xy, fill, outline=(196, 204, 216), radius=22, width=3):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def make_requirement_flow():
    img = Image.new("RGB", (1700, 940), "#F7F9FC")
    d = ImageDraw.Draw(img)
    d.text((70, 52), "附录图A 需求分析与系统流程图", font=F_TITLE, fill="#1F2937")
    d.text((72, 110), "智能客服多模态接待场景：从用户输入到服务推荐的完整处理链路", font=F_SUB, fill="#516070")

    boxes = [
        ((80, 220, 360, 370), "多模态输入\n文字诉求\n语音/音频\n表情图片\n业务截图", "#E8F2FF"),
        ((460, 220, 740, 370), "数据预处理\n文本规范化\n音频特征\n图像缩放\n截图OCR规则", "#F0F9FF"),
        ((840, 220, 1120, 370), "单模态识别\n文本意图\n语音声学\n情绪状态\n截图类别", "#ECFDF5"),
        ((1220, 220, 1500, 370), "融合决策\n26维特征拼接\n全连接MLP\nSoftmax五分类", "#FFF7ED"),
        ((1220, 545, 1500, 695), "智能体输出\n最终意图\n推荐服务\n智能回复\n处理步骤", "#FDF2F8"),
        ((840, 545, 1120, 695), "人工介入判断\n低置信度\n投诉/负向情绪\n复杂售后问题", "#FEF2F2"),
        ((460, 545, 740, 695), "前端展示\n文件预览\n五类得分\n融合依据\n模态摘要", "#F5F3FF"),
        ((80, 545, 360, 695), "课程测试\n五个完整用例\n五类意图覆盖\n端到端验证", "#F1F5F9"),
    ]
    for xy, text, fill in boxes:
        rounded_box(d, xy, fill)
        multiline_center(d, xy, text, "#1F2937", F_BOX, line_gap=7)

    arrow(d, (360, 295), (460, 295))
    arrow(d, (740, 295), (840, 295))
    arrow(d, (1120, 295), (1220, 295))
    arrow(d, (1360, 370), (1360, 545))
    arrow(d, (1220, 620), (1120, 620))
    arrow(d, (840, 620), (740, 620))
    arrow(d, (460, 620), (360, 620))

    d.text((80, 800), "对应需求：识别咨询、投诉、退款、故障报修、不会操作五类客服诉求，并根据情绪和置信度给出服务推荐。", font=F_BODY, fill="#374151")
    d.text((80, 838), "对应实现：Flask Web + 文本/语音/表情/截图分支 + 多模态特征拼接 + 全连接融合。", font=F_BODY, fill="#374151")
    img.save(ASSET_DIR / "appendix_requirement_flow.png", quality=95)


def make_model_architecture():
    img = Image.new("RGB", (1700, 980), "#F8FAFC")
    d = ImageDraw.Draw(img)
    d.text((70, 52), "附录图B 多模态模型结构示意图", font=F_TITLE, fill="#111827")
    d.text((72, 110), "四个模态分支输出同一标签空间的五类得分，再与6个辅助数值特征拼接为26维向量", font=F_SUB, fill="#4B5563")

    top_boxes = [
        ((70, 220, 310, 380), "文本分支\nTF-IDF/BERT\n五类意图得分", "#DBEAFE"),
        ((380, 220, 620, 380), "语音分支\nMFCC/LSTM\n声学意图得分", "#D1FAE5"),
        ((690, 220, 930, 380), "表情分支\nFER CNN/MLP\n情绪辅助得分", "#FCE7F3"),
        ((1000, 220, 1240, 380), "截图分支\ndemo CNN + OCR\n业务类别得分", "#FEF3C7"),
        ((1310, 220, 1550, 380), "辅助特征\n文本长度/上传状态\n各分支置信度", "#EDE9FE"),
    ]
    for xy, text, fill in top_boxes:
        rounded_box(d, xy, fill)
        multiline_center(d, xy, text, "#111827", F_BOX)

    concat = (440, 510, 1260, 635)
    rounded_box(d, concat, "#FFFFFF", outline="#94A3B8", radius=18)
    multiline_center(d, concat, "特征拼接 Concat\ntext_scores:5 + audio_scores:5 + emotion_scores:5 + image_scores:5 + numeric_features:6 = 26维", "#111827", F_BOX)

    mlp_boxes = [
        ((455, 760, 690, 860), "Linear\n26 -> 48", "#E0F2FE"),
        ((735, 760, 970, 860), "ReLU + Dropout\n0.1", "#ECFDF5"),
        ((1015, 760, 1250, 860), "Linear\n48 -> 24", "#FEF3C7"),
        ((1295, 760, 1530, 860), "Linear + Softmax\n24 -> 5", "#FFE4E6"),
    ]
    for xy, text, fill in mlp_boxes:
        rounded_box(d, xy, fill)
        multiline_center(d, xy, text, "#111827", F_BOX)

    for xy, _, _ in top_boxes:
        arrow(d, ((xy[0] + xy[2]) // 2, xy[3]), ((concat[0] + concat[2]) // 2, concat[1]))
    arrow(d, ((concat[0] + concat[2]) // 2, concat[3]), (570, 760))
    arrow(d, (690, 810), (735, 810))
    arrow(d, (970, 810), (1015, 810))
    arrow(d, (1250, 810), (1295, 810))
    d.text((460, 906), "输出标签：咨询 consult、投诉 complaint、退款 refund、故障报修 repair、不会操作 howto", font=F_BODY, fill="#374151")
    img.save(ASSET_DIR / "appendix_model_architecture.png", quality=95)


def make_test_summary():
    img = Image.new("RGB", (1700, 960), "#FFFFFF")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 1700, 160), fill="#F1F5F9")
    d.text((70, 48), "附录图C 五类演示测试结果汇总", font=F_TITLE, fill="#0F172A")
    d.text((72, 108), "course_demo_test_data 五个完整测试文件夹均通过端到端识别", font=F_SUB, fill="#475569")

    rows = [
        ("故障报修", "repair", "repair", "fault", "fault", 0.9924),
        ("订单咨询", "consult", "consult", "order", "order", 0.7193),
        ("退款售后", "refund", "refund", "refund", "refund", 0.5596),
        ("操作指引", "howto", "howto", "howto", "howto", 0.7159),
        ("投诉", "complaint", "complaint", "normal", "normal", 0.9939),
    ]
    headers = ["场景", "期望意图", "预测意图", "期望截图", "预测截图", "融合置信度", "结论"]
    x0, y0 = 70, 220
    col_w = [210, 220, 220, 210, 210, 250, 170]
    row_h = 72
    x = x0
    for header, w in zip(headers, col_w):
        d.rectangle((x, y0, x + w, y0 + row_h), fill="#DBEAFE", outline="#CBD5E1", width=2)
        multiline_center(d, (x, y0, x + w, y0 + row_h), header, "#0F172A", F_BODY)
        x += w
    for r, row in enumerate(rows, 1):
        x = x0
        fill = "#FFFFFF" if r % 2 else "#F8FAFC"
        values = [row[0], row[1], row[2], row[3], row[4], f"{row[5] * 100:.2f}%", "通过"]
        for value, w in zip(values, col_w):
            d.rectangle((x, y0 + r * row_h, x + w, y0 + (r + 1) * row_h), fill=fill, outline="#CBD5E1", width=2)
            color = "#047857" if value == "通过" else "#111827"
            multiline_center(d, (x, y0 + r * row_h, x + w, y0 + (r + 1) * row_h), value, color, F_BODY)
            x += w

    d.rounded_rectangle((70, 680, 500, 830), radius=22, fill="#ECFDF5", outline="#A7F3D0", width=3)
    multiline_center(d, (70, 680, 500, 830), "端到端样例\n5/5 通过\n准确率 100%", "#065F46", F_BOX)
    d.rounded_rectangle((620, 680, 1050, 830), radius=22, fill="#EFF6FF", outline="#BFDBFE", width=3)
    multiline_center(d, (620, 680, 1050, 830), "业务截图五分类\n5/5 通过\n截图类别全部正确", "#1D4ED8", F_BOX)
    d.rounded_rectangle((1170, 680, 1600, 830), radius=22, fill="#FFF7ED", outline="#FED7AA", width=3)
    multiline_center(d, (1170, 680, 1600, 830), "说明\n课程demo测试结果\n非严格泛化指标", "#9A3412", F_BOX)

    img.save(ASSET_DIR / "appendix_test_summary.png", quality=95)


def main():
    make_requirement_flow()
    make_model_architecture()
    make_test_summary()
    print(json.dumps({
        "created": [
            str(ASSET_DIR / "appendix_requirement_flow.png"),
            str(ASSET_DIR / "appendix_model_architecture.png"),
            str(ASSET_DIR / "appendix_test_summary.png"),
        ]
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
