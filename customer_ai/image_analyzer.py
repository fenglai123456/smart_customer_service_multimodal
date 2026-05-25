from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat
import torch
from torch import nn
from torchvision import models


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "customer_image_binary_cnn.pt"
SCREENSHOT_INTENT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "customer_screenshot_intent_cnn.pt"
DEMO_SCREENSHOT_INTENT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "demo_screenshot_intent_cnn.pt"
COCO_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "coco_cnn.pt"
_IMAGE_MODEL = None
_SCREENSHOT_INTENT_MODEL = None
_SCREENSHOT_INTENT_LABELS: list[str] = []
_SCREENSHOT_INTENT_IMAGE_SIZE = 96
_SCREENSHOT_INTENT_ARCHITECTURE = "screenshot_intent_cnn_v1"
_SCREENSHOT_INTENT_MODEL_NAME = ""
_COCO_MODEL = None
_COCO_LABELS: list[str] = []
_COCO_IMAGE_SIZE = 64
_COCO_ARCHITECTURE = "legacy"
_OCR_ENGINE = None
_OCR_ERROR = ""

SCREENSHOT_INTENT_TO_APP_INTENT = {
    "fault": "repair",
    "order": "consult",
    "refund": "refund",
    "howto": "howto",
    "normal": "consult",
}

SCREENSHOT_KEYWORDS = {
    "fault": (
        "fault",
        "error",
        "failed",
        "failure",
        "bug",
        "crash",
        "timeout",
        "exception",
        "payment failed",
        "故障",
        "报错",
        "异常",
        "失败",
        "无法支付",
        "支付失败",
        "崩溃",
    ),
    "order": (
        "order",
        "tracking",
        "shipping",
        "delivery",
        "shipment",
        "logistics",
        "订单",
        "物流",
        "配送",
        "发货",
        "快递",
        "运单",
    ),
    "refund": (
        "refund",
        "return",
        "after_sale",
        "after-sale",
        "credit memo",
        "退款",
        "退货",
        "售后",
        "返款",
        "退回",
        "退费",
    ),
    "howto": (
        "howto",
        "how_to",
        "guide",
        "tutorial",
        "setup",
        "manual",
        "steps",
        "操作",
        "步骤",
        "教程",
        "设置",
        "不会",
        "指引",
        "指南",
    ),
    "normal": (
        "normal",
        "general",
        "consult",
        "inquiry",
        "faq",
        "普通",
        "咨询",
        "服务",
        "客服",
        "问题",
    ),
}


class BinaryCNN(nn.Module):
    def __init__(self, image_size: int = 64):
        super().__init__()
        pooled = image_size // 4
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * pooled * pooled, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class CocoCNN(nn.Module):
    def __init__(self, num_classes: int, image_size: int = 64, architecture: str = "legacy"):
        super().__init__()
        pooled = image_size // 4
        if architecture == "enhanced_cnn_v2":
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.10),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * pooled * pooled, 256),
                nn.ReLU(),
                nn.Dropout(0.35),
                nn.Linear(256, num_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * pooled * pooled, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )

    def forward(self, x):
        return self.net(x)


class CocoResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = models.resnet18(weights=None)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x):
        return self.net(x)


class ScreenshotIntentCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ScreenshotIntentResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = models.resnet18(weights=None)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x):
        return self.net(x)


class DemoScreenshotCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)),
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 192),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class ImageSignals:
    label: str
    confidence: float
    summary: str
    details: dict
    intent_scores: dict[str, float]


def analyze_image(path: str | Path | None, context_text: str = "") -> ImageSignals | None:
    if not path:
        return None

    image_path = Path(path)
    if not image_path.exists():
        return None

    with Image.open(image_path) as image:
        img = image.convert("RGB")
        width, height = img.size
        stat = ImageStat.Stat(img)
        mean_rgb = stat.mean
        gray = img.convert("L")
        gray_arr = np.asarray(gray, dtype=np.float32)
        brightness = float(gray_arr.mean())
        contrast = float(gray_arr.std())

        small = gray.resize((96, 96))
        arr = np.asarray(small, dtype=np.float32)
        horizontal_edges = float(np.abs(np.diff(arr, axis=0)).mean())
        vertical_edges = float(np.abs(np.diff(arr, axis=1)).mean())
        edge_score = horizontal_edges + vertical_edges

    # The screenshot branch is layered on purpose:
    # 1) cheap visual heuristics keep the system usable without any model file;
    # 2) demo/strict CNN predicts the screenshot class;
    # 3) optional OCR and filename keywords provide an explainable override for
    #    classroom screenshots whose text directly names the business state.
    heuristic = _heuristic_result(brightness, contrast, edge_score)
    screenshot_intent = _predict_screenshot_intent(image_path)
    ocr = _extract_optional_ocr_text(image_path)
    keyword_intent = _predict_screenshot_intent_from_keywords(
        image_path=image_path,
        context_text=context_text,
        ocr_text=ocr.get("text", ""),
    )
    screenshot_decision = _combine_screenshot_intent(screenshot_intent, keyword_intent)
    cnn = _predict_customer_image(image_path)
    coco = _predict_coco_image(image_path)
    label, confidence, summary = heuristic["label"], heuristic["confidence"], heuristic["summary"]

    if screenshot_decision:
        label = screenshot_decision["label"]
        confidence = max(heuristic["confidence"], screenshot_decision["confidence"])
        summary = screenshot_decision["summary"]
    elif cnn:
        if cnn["class_name"] == "customer_screenshot":
            if edge_score > 28 or contrast > 34:
                label = "疑似故障截图"
                summary = "客服截图 CNN 识别为界面/截图类图片，结合边缘和对比度特征，可能包含页面异常或订单信息。"
            else:
                label = "客服截图/界面图片"
                summary = "客服截图 CNN 识别为界面/截图类图片，可作为故障报修或操作引导的辅助证据。"
            confidence = max(heuristic["confidence"], cnn["confidence"])
        elif heuristic["label"] in {"光线较暗图片", "信息较少图片"}:
            confidence = max(heuristic["confidence"], cnn["confidence"] * 0.75)
        else:
            label = "普通客服图片"
            confidence = max(heuristic["confidence"], cnn["confidence"])
            summary = "客服截图 CNN 倾向判断为普通自然图片，系统将其作为辅助线索参与融合。"

    return ImageSignals(
        label=label,
        confidence=round(float(confidence), 4),
        summary=summary,
        details={
            "width": width,
            "height": height,
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "edge_score": round(edge_score, 2),
            "mean_rgb": [round(v, 2) for v in mean_rgb],
            "heuristic_label": heuristic["label"],
            "heuristic_confidence": heuristic["confidence"],
            "screenshot_intent_cnn": screenshot_intent,
            "screenshot_keyword_rules": keyword_intent,
            "screenshot_demo_decision": screenshot_decision,
            "ocr": ocr,
            "cnn": cnn,
            "coco_cnn": coco,
        },
        intent_scores={
            key: round(value, 6)
            for key, value in _image_intent_scores(
                label=label,
                confidence=confidence,
                coco=coco,
                screenshot_intent=screenshot_decision,
            ).items()
        },
    )


def _heuristic_result(brightness: float, contrast: float, edge_score: float) -> dict:
    if edge_score > 34 and contrast > 42:
        return {
            "label": "疑似故障截图",
            "confidence": 0.82,
            "summary": "图片边缘和对比度较高，接近页面报错、订单截图或系统异常截图。",
        }
    if brightness < 75:
        return {
            "label": "光线较暗图片",
            "confidence": 0.68,
            "summary": "图片整体偏暗，可能需要用户补充更清晰截图。",
        }
    if contrast < 22:
        return {
            "label": "信息较少图片",
            "confidence": 0.61,
            "summary": "图片对比度较低，包含的可识别线索可能有限。",
        }
    return {
        "label": "普通客服图片",
        "confidence": 0.64,
        "summary": "图片质量可用于辅助判断，建议结合文本或语音描述确认意图。",
    }


def _predict_customer_image(path: Path) -> dict | None:
    model = _load_image_model()
    if model is None:
        return None

    with Image.open(path) as image:
        tensor = _preprocess_for_cnn(image, image_size=64, normalize_imagenet=False)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    class_index = int(np.argmax(probabilities))
    class_names = ["natural_image", "customer_screenshot"]
    return {
        "class_index": class_index,
        "class_name": class_names[class_index],
        "label": "普通自然图片" if class_index == 0 else "客服截图",
        "confidence": round(float(probabilities[class_index]), 4),
        "probabilities": {
            "natural_image": round(float(probabilities[0]), 4),
            "customer_screenshot": round(float(probabilities[1]), 4),
        },
        "model": "customer_image_binary_cnn.pt",
        "source": "pytorch_cnn",
    }


def _predict_screenshot_intent(path: Path) -> dict | None:
    model = _load_screenshot_intent_model()
    if model is None:
        return None

    with Image.open(path) as image:
        tensor = _preprocess_for_cnn(
            image,
            image_size=_SCREENSHOT_INTENT_IMAGE_SIZE,
            normalize_imagenet=_SCREENSHOT_INTENT_ARCHITECTURE == "resnet18_transfer",
        )

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    class_index = int(np.argmax(probabilities))
    class_name = _SCREENSHOT_INTENT_LABELS[class_index] if class_index < len(_SCREENSHOT_INTENT_LABELS) else str(class_index)
    label, summary = _screenshot_display(class_name)
    return {
        "class_index": class_index,
        "class_name": class_name,
        "label": label,
        "confidence": round(float(probabilities[class_index]), 4),
        "probabilities": {
            _SCREENSHOT_INTENT_LABELS[index] if index < len(_SCREENSHOT_INTENT_LABELS) else str(index): round(float(value), 4)
            for index, value in enumerate(probabilities)
        },
        "model": _SCREENSHOT_INTENT_MODEL_NAME or "customer_screenshot_intent_cnn.pt",
        "source": "pytorch_cnn",
        "summary": summary,
    }


def _extract_optional_ocr_text(path: Path) -> dict:
    global _OCR_ENGINE, _OCR_ERROR
    if _OCR_ENGINE == "unavailable":
        return {"text": "", "source": "ocr_unavailable", "error": _OCR_ERROR}

    try:
        if _OCR_ENGINE is None:
            import pytesseract  # type: ignore

            _OCR_ENGINE = pytesseract
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            last_error = ""
            for lang in ("chi_sim+eng", "eng"):
                try:
                    text = _OCR_ENGINE.image_to_string(rgb, lang=lang)
                    return {"text": str(text).strip(), "source": f"pytesseract:{lang}", "error": ""}
                except Exception as exc:
                    last_error = str(exc)
            raise RuntimeError(last_error)
    except Exception as exc:  # pragma: no cover - OCR is optional in local demo installs
        _OCR_ENGINE = "unavailable"
        _OCR_ERROR = str(exc)
        return {"text": "", "source": "ocr_unavailable", "error": _OCR_ERROR}


def _predict_screenshot_intent_from_keywords(
    image_path: Path,
    context_text: str = "",
    ocr_text: str = "",
) -> dict | None:
    """Classify business screenshots from filename/context/OCR keywords.

    This rule layer is not presented as a strict real-world accuracy model. It
    exists so the course demo can transparently show why a screenshot is treated
    as fault/order/refund/howto/normal when the uploaded image contains obvious
    business terms.
    """
    sources = {
        "filename": image_path.name.lower(),
        "context": str(context_text).lower(),
        "ocr": str(ocr_text).lower(),
    }
    counts: dict[str, float] = {}
    matched: dict[str, list[str]] = {}
    for class_name, keywords in SCREENSHOT_KEYWORDS.items():
        hits: list[str] = []
        for keyword in keywords:
            key = keyword.lower()
            if any(key in value for value in sources.values()):
                hits.append(keyword)
        if hits:
            matched[class_name] = hits
            counts[class_name] = float(len(hits))

    if not counts:
        return None

    class_name = max(counts, key=lambda item: (counts[item], item == "normal"))
    total = sum(counts.values()) or 1.0
    probabilities = {
        label: round(counts.get(label, 0.0) / total, 4)
        for label in SCREENSHOT_KEYWORDS
    }
    confidence = min(0.98, 0.70 + 0.08 * counts[class_name])
    label, summary = _screenshot_display(class_name)
    return {
        "class_index": list(SCREENSHOT_KEYWORDS.keys()).index(class_name),
        "class_name": class_name,
        "label": label,
        "confidence": round(float(confidence), 4),
        "probabilities": probabilities,
        "model": "ocr_keyword_rules",
        "source": "filename_path_ocr_keyword_rules",
        "summary": f"OCR/文件名关键词命中 {class_name} 类演示截图线索，作为课程演示规则层参与融合。",
        "matched_keywords": matched,
        "source_text": {key: value[:300] for key, value in sources.items() if value},
    }


def _combine_screenshot_intent(cnn: dict | None, keyword: dict | None) -> dict | None:
    """Merge CNN and OCR/keyword screenshot decisions for demo robustness."""
    if not cnn and not keyword:
        return None
    if not keyword:
        decision = dict(cnn or {})
        decision["decision_source"] = "demo_cnn" if _is_demo_screenshot_model(decision) else "strict_cnn"
        return decision
    if not cnn:
        decision = dict(keyword)
        decision["decision_source"] = "keyword_rules"
        return decision

    cnn_class = str(cnn.get("class_name", ""))
    keyword_class = str(keyword.get("class_name", ""))
    decision = dict(cnn)
    decision["cnn_class_name"] = cnn_class
    decision["keyword_class_name"] = keyword_class
    decision["keyword_rules"] = keyword

    if _is_demo_screenshot_model(cnn):
        # In demo mode an explicit OCR/filename hit can override the small CNN;
        # the decision_source field is exposed on the frontend and in reports.
        if keyword_class and keyword_class != cnn_class and float(keyword.get("confidence", 0.0)) >= 0.78:
            decision.update({key: value for key, value in keyword.items() if key != "probabilities"})
            decision["cnn_class_name"] = cnn_class
            decision["keyword_class_name"] = keyword_class
            decision["keyword_rules"] = keyword
            decision["decision_source"] = "demo_cnn_keyword_override"
            decision["summary"] = (
                f"Demo CNN 输出 {cnn_class}，OCR/文件名关键词指向 {keyword_class}，"
                "演示模式优先采用可解释关键词线索。"
            )
        else:
            decision["decision_source"] = "demo_cnn_keyword_confirmed" if keyword_class == cnn_class else "demo_cnn"
    else:
        decision["decision_source"] = "strict_cnn"
        decision["keyword_rules"] = keyword

    decision["confidence"] = round(max(float(cnn.get("confidence", 0.0)), float(keyword.get("confidence", 0.0))), 4)
    return decision


def _is_demo_screenshot_model(screenshot_intent: dict | None) -> bool:
    if not screenshot_intent:
        return False
    model = str(screenshot_intent.get("model", "")).lower()
    source = str(screenshot_intent.get("source", "")).lower()
    decision_source = str(screenshot_intent.get("decision_source", "")).lower()
    return "demo" in model or "keyword" in source or "keyword" in decision_source


def _screenshot_display(class_name: str) -> tuple[str, str]:
    display = {
        "fault": ("故障截图", "截图 CNN/规则识别为故障或异常页面，优先作为故障报修线索参与融合。"),
        "order": ("订单截图", "截图 CNN/规则识别为订单或物流页面，优先作为订单咨询线索参与融合。"),
        "refund": ("退款截图", "截图 CNN/规则识别为退款或售后页面，优先作为退款诉求线索参与融合。"),
        "howto": ("操作引导截图", "截图 CNN/规则识别为帮助文档或操作指引页面，优先作为不会操作线索参与融合。"),
        "normal": ("普通客服页面截图", "截图 CNN/规则识别为普通客服相关页面，作为一般咨询线索参与融合。"),
    }
    return display.get(class_name, (class_name, "截图意图识别分支已输出图片侧意图线索。"))


def _predict_coco_image(path: Path) -> dict | None:
    model = _load_coco_model()
    if model is None:
        return None

    with Image.open(path) as image:
        tensor = _preprocess_for_cnn(
            image,
            image_size=_COCO_IMAGE_SIZE,
            normalize_imagenet=_COCO_ARCHITECTURE == "resnet18_transfer",
        )

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    class_index = int(np.argmax(probabilities))
    label = _COCO_LABELS[class_index] if class_index < len(_COCO_LABELS) else str(class_index)
    return {
        "class_index": class_index,
        "label": label,
        "confidence": round(float(probabilities[class_index]), 4),
        "probabilities": {
            _COCO_LABELS[index] if index < len(_COCO_LABELS) else str(index): round(float(value), 4)
            for index, value in enumerate(probabilities)
        },
        "model": "coco_cnn.pt",
        "source": "coco_pytorch_cnn",
    }


def _load_image_model():
    global _IMAGE_MODEL
    if _IMAGE_MODEL is not None:
        return _IMAGE_MODEL
    if not MODEL_PATH.exists():
        return None

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = BinaryCNN(image_size=64)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _IMAGE_MODEL = model
    return _IMAGE_MODEL


def _load_screenshot_intent_model():
    global _SCREENSHOT_INTENT_MODEL, _SCREENSHOT_INTENT_LABELS, _SCREENSHOT_INTENT_IMAGE_SIZE, _SCREENSHOT_INTENT_ARCHITECTURE, _SCREENSHOT_INTENT_MODEL_NAME
    if _SCREENSHOT_INTENT_MODEL is not None:
        return _SCREENSHOT_INTENT_MODEL
    # Prefer the lightweight demo model shipped with the GitHub repo. If it is
    # absent, fall back to the stricter training artifact when available locally.
    model_path = DEMO_SCREENSHOT_INTENT_MODEL_PATH if DEMO_SCREENSHOT_INTENT_MODEL_PATH.exists() else SCREENSHOT_INTENT_MODEL_PATH
    if not model_path.exists():
        return None

    _SCREENSHOT_INTENT_MODEL_NAME = model_path.name
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    label_to_id = checkpoint.get("label_to_id", {})
    _SCREENSHOT_INTENT_LABELS = [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]
    _SCREENSHOT_INTENT_IMAGE_SIZE = int(checkpoint.get("image_size", 96))
    _SCREENSHOT_INTENT_ARCHITECTURE = str(checkpoint.get("architecture", "screenshot_intent_cnn_v1"))
    if _SCREENSHOT_INTENT_ARCHITECTURE == "demo_screenshot_cnn_v1":
        model = DemoScreenshotCNN(num_classes=len(_SCREENSHOT_INTENT_LABELS))
    elif _SCREENSHOT_INTENT_ARCHITECTURE == "resnet18_transfer":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(_SCREENSHOT_INTENT_LABELS))
    else:
        model = ScreenshotIntentCNN(num_classes=len(_SCREENSHOT_INTENT_LABELS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _SCREENSHOT_INTENT_MODEL = model
    return _SCREENSHOT_INTENT_MODEL


def _load_coco_model():
    global _COCO_MODEL, _COCO_LABELS, _COCO_IMAGE_SIZE, _COCO_ARCHITECTURE
    if _COCO_MODEL is not None:
        return _COCO_MODEL
    if not COCO_MODEL_PATH.exists():
        return None

    checkpoint = torch.load(COCO_MODEL_PATH, map_location="cpu", weights_only=False)
    label_to_id = checkpoint.get("label_to_id", {})
    _COCO_LABELS = [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]
    architecture = checkpoint.get("architecture", "legacy")
    _COCO_ARCHITECTURE = str(architecture)
    _COCO_IMAGE_SIZE = int(checkpoint.get("image_size", 64))
    if architecture == "resnet18_transfer":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(_COCO_LABELS))
    else:
        model = CocoCNN(
            num_classes=len(_COCO_LABELS),
            image_size=_COCO_IMAGE_SIZE,
            architecture=architecture,
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _COCO_MODEL = model
    return _COCO_MODEL


def _preprocess_for_cnn(image: Image.Image, image_size: int = 64, normalize_imagenet: bool = False) -> torch.Tensor:
    resized = image.convert("RGB").resize((image_size, image_size))
    arr = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    if normalize_imagenet:
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        arr = (arr - mean) / std
    return torch.tensor(arr[None, :, :, :], dtype=torch.float32)


def _image_intent_scores(
    label: str,
    confidence: float,
    coco: dict | None,
    screenshot_intent: dict | None = None,
) -> dict[str, float]:
    scores = {
        "consult": 0.20,
        "complaint": 0.20,
        "refund": 0.20,
        "repair": 0.20,
        "howto": 0.20,
    }
    strength = max(0.35, float(confidence))

    if screenshot_intent:
        class_name = str(screenshot_intent.get("class_name", ""))
        mapped_intent = SCREENSHOT_INTENT_TO_APP_INTENT.get(class_name)
        if mapped_intent:
            # Demo screenshots are intentionally stronger evidence than generic
            # image hints because each demo case is built around a known business
            # page state: fault/order/refund/howto/normal.
            if _is_demo_screenshot_model(screenshot_intent):
                scores[mapped_intent] += 2.4 * strength
            else:
                scores[mapped_intent] += 0.72 * strength
            if class_name == "fault":
                scores["complaint"] += 0.18 * strength
            elif class_name == "refund":
                scores["complaint"] += 0.10 * strength
    elif "故障" in label:
        scores["repair"] += 0.55 * strength
    elif "截图" in label or "界面" in label:
        scores["repair"] += 0.28 * strength
        scores["howto"] += 0.20 * strength
    elif "较暗" in label or "信息较少" in label:
        scores["howto"] += 0.24 * strength
        scores["repair"] += 0.16 * strength
    else:
        scores["consult"] += 0.16 * strength

    if coco:
        coco_label = str(coco.get("label", ""))
        coco_confidence = float(coco.get("confidence", 0.0))
        if coco_label in {"person", "chair"}:
            scores["consult"] += 0.06 * coco_confidence
        elif coco_label in {"car"}:
            scores["repair"] += 0.06 * coco_confidence

    total = sum(scores.values()) or 1.0
    return {intent: value / total for intent, value in scores.items()}
