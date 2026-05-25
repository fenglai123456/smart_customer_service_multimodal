from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "text_intent_augmented_metrics.json"
MODEL_PATH = ROOT / "models" / "text_intent_augmented.joblib"


INTENTS = {
    "consult": ["咨询", "请问", "了解", "查询", "多少钱", "库存", "物流", "活动", "会员", "营业时间"],
    "complaint": ["投诉", "不满意", "服务差", "态度差", "没人处理", "生气", "差评", "升级处理", "反馈", "等太久"],
    "refund": ["退款", "退货", "取消订单", "钱退回", "赔付", "扣费", "到账", "售后退款", "退回账户", "申请退"],
    "repair": ["报修", "故障", "打不开", "报错", "系统异常", "闪退", "支付失败", "无法使用", "维修", "截图错误"],
    "howto": ["不会操作", "怎么", "如何", "步骤", "入口", "找不到", "绑定", "修改", "设置", "教我"],
}

OBJECTS = ["订单", "商品", "页面", "账号", "会员", "发票", "优惠券", "门店", "小程序", "支付"]
TAILS = ["麻烦帮我处理", "请尽快回复", "我现在需要解决", "能不能告诉我", "谢谢", "需要人工协助", "请给我步骤", "一直没有结果"]
EXTRA_SAMPLES = [
    ("我不会操作这个功能", "howto"),
    ("请告诉我具体步骤", "howto"),
    ("这个功能怎么用", "howto"),
    ("在哪里可以修改信息", "howto"),
    ("找不到入口请教我", "howto"),
    ("麻烦给我操作指引", "howto"),
    ("我想咨询订单物流", "consult"),
    ("请问营业时间是什么", "consult"),
    ("这个活动规则是什么", "consult"),
]


def main() -> None:
    texts, labels = build_dataset()
    train_x, test_x, train_y, test_y = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=4.0, random_state=42)),
        ]
    )
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    metrics = {
        "model": "TF-IDF(char 2-5gram) + LogisticRegression",
        "task": "augmented customer-service text intent classification",
        "samples": len(texts),
        "train_size": len(train_x),
        "test_size": len(test_x),
        "accuracy": round(float(accuracy_score(test_y, pred_y)), 4),
        "classification_report": classification_report(test_y, pred_y, output_dict=True, zero_division=0),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_dataset() -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for label, keywords in INTENTS.items():
        for keyword in keywords:
            for obj in OBJECTS:
                for tail in TAILS[:5]:
                    texts.append(f"{obj}{keyword}，{tail}")
                    labels.append(label)
        for i, keyword in enumerate(keywords):
            texts.append(f"{keyword}{OBJECTS[i % len(OBJECTS)]}")
            labels.append(label)
    for text, label in EXTRA_SAMPLES:
        for _ in range(20):
            texts.append(text)
            labels.append(label)
    return texts, labels


if __name__ == "__main__":
    main()
