from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "FINAL_EXPERIMENT_REPORT.md"

REPORTS = [
    {
        "name": "文本 TF-IDF 基线",
        "path": ROOT / "reports" / "text_intent_metrics.json",
        "task_definition": "客服文本五分类，使用轻量 demo 文本样本。",
        "split": "stratified train_test_split，约 70%/30%。",
        "interpretation": "能说明轻量文本基线可用，但样本只有 30 条左右，泛化结论有限。",
        "risk": "中",
    },
    {
        "name": "文本增强单模态",
        "path": ROOT / "reports" / "text_intent_augmented_metrics.json",
        "task_definition": "程序生成的客服意图五分类增强样本。",
        "split": "stratified train_test_split，75%/25%。",
        "interpretation": "1.0 来自模板化增强语料，适合作为演示模型，不宜直接声称真实客服文本泛化达到 100%。",
        "risk": "中",
    },
    {
        "name": "文本 BERT",
        "path": ROOT / "reports" / "text_bert_metrics.json",
        "task_definition": "BERT 文本意图五分类 smoke 训练。",
        "split": "stratified train_test_split，21 条训练、9 条测试。",
        "interpretation": "使用 tiny-random-bert 且只跑 1 个 epoch，准确率低是预期结果；该项证明训练链路，不作为最终模型。",
        "risk": "高",
    },
    {
        "name": "语音 LSTM",
        "path": ROOT / "reports" / "audio_lstm_metrics.json",
        "task_definition": "LibriSpeech 说话人代理分类，不是客服语音意图识别。",
        "split": "stratified train_test_split，80%/20%。",
        "interpretation": "低分说明小样本 LSTM 声学分类尚不稳定；Web 端语音意图仍依赖 ASR 转写文本。",
        "risk": "高",
    },
    {
        "name": "语音 MFCC 单模态",
        "path": ROOT / "reports" / "audio_mfcc_classifier_metrics.json",
        "task_definition": "LibriSpeech 说话人代理分类，MFCC 统计特征分类。",
        "split": "stratified train_test_split，80%/20%。",
        "interpretation": "结果较高，说明声学特征分支可区分说话人/音频模式；它不是客服意图准确率。",
        "risk": "中",
    },
    {
        "name": "表情 CNN",
        "path": ROOT / "reports" / "emotion_cnn_metrics.json",
        "task_definition": "FER-2013 原始七分类表情识别。",
        "split": "stratified train_test_split，80%/20%。",
        "interpretation": "采样 1400、训练 2 epoch，准确率低，说明七分类 CNN 未充分训练；保留为弱项和改进方向。",
        "risk": "高",
    },
    {
        "name": "表情 MLP",
        "path": ROOT / "reports" / "emotion_mlp_metrics.json",
        "task_definition": "FER-2013 angry-vs-other 二分类，用于客服负向情绪检测。",
        "split": "stratified train_test_split，80%/20%。",
        "interpretation": "达到 82%+ 的是二分类场景化任务，不等同于 FER-2013 原始七分类；Web 端已接入该模型作为愤怒倾向信号。",
        "risk": "中",
    },
    {
        "name": "图片 CNN",
        "path": ROOT / "reports" / "image_cnn_metrics.json",
        "task_definition": "COCO top-category 小样本图像多分类。",
        "split": "stratified train_test_split，80%/20%。",
        "interpretation": "150 张样本、2 epoch，准确率低，说明 COCO 通用多分类模型不足；不用于当前 Web 端主流程。",
        "risk": "高",
    },
    {
        "name": "客服截图图片单模态",
        "path": ROOT / "reports" / "customer_image_binary_metrics.json",
        "task_definition": "COCO 自然图片 vs 合成客服截图二分类。",
        "split": "stratified train_test_split，80%/20%。",
        "interpretation": "1.0 来自自然图像与合成截图的强分布差异，说明模型能识别演示截图形态，但不能证明真实业务截图泛化 100%。",
        "risk": "中",
    },
    {
        "name": "多模态规则融合",
        "path": ROOT / "reports" / "multimodal_demo_metrics.json",
        "task_definition": "10 条人工设计多模态 demo 样例上的规则加权融合验证。",
        "split": "无训练/测试划分；逐条功能验证。",
        "interpretation": "1.0 只表示预设 demo 场景全部跑通，应定位为功能验收，不作为模型泛化指标。",
        "risk": "高",
    },
    {
        "name": "多模态训练融合",
        "path": ROOT / "reports" / "fusion_model_metrics.json",
        "task_definition": "10 条 demo 样本构造 15 维融合特征后训练 LogisticRegression。",
        "split": "stratified train_test_split，50%/50%。",
        "interpretation": "样本只有 10 条，测试集每类 1 条，0.6 说明训练式融合还不稳定；当前 Web 主流程使用规则融合更稳。",
        "risk": "高",
    },
]


def main() -> None:
    rows = []
    for item in REPORTS:
        data = load_json(item["path"])
        rows.append(build_row(item, data))

    lines = [
        "# 智能客服多模态接待场景实验总报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 口径说明",
        "",
        "本报告区分“Web 演示已接入能力”和“训练脚本实验结果”。准确率只在对应任务定义、样本规模和划分方式下成立，不能跨任务解释。尤其是 1.0 准确率项目，需要结合数据来源判断，不能直接说成真实业务泛化 100%。",
        "",
        "当前 Web 演示主线使用轻量文本模型、关键词规则、情绪标签、音频声学摘要、FER 表情图片模型和客服截图 CNN 模型完成闭环；BERT、LSTM、COCO 通用 CNN 和训练式融合模型主要用于证明实验链路可运行和后续扩展。",
        "",
        "## 指标总览",
        "",
        "| 模块 | 任务定义 | 模型/方法 | 样本规模 | 训练/测试划分 | 准确率 | 风险 | 解释 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {task_definition} | {model} | {sample_summary} | {split_summary} | {accuracy} | {risk} | {interpretation} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## 重点解释",
            "",
            "- `多模态规则融合 = 1.0`：该结果来自 `data/demo/multimodal_samples.csv` 的 10 条人工设计样例，每类 2 条，没有独立测试集。它适合证明“用户输入 -> 多模态融合 -> 回复/转人工”的演示链路跑通，不适合当作泛化指标。",
            "- `客服截图图片单模态 = 1.0`：任务是 COCO 自然图片和合成客服截图二分类，两个类别视觉差异较大，因此容易得到满分。答辩时应说明它是截图形态识别能力，不是所有真实业务截图的 100% 准确率。",
            "- `文本增强单模态 = 1.0`：增强语料由模板组合生成，训练集和测试集分布非常接近。可以作为演示系统的稳定文本模型，但真实客服口语、错别字、长上下文还需要额外数据验证。",
            "- `BERT = 0.1111`：当前使用 `hf-internal-testing/tiny-random-bert` 且 smoke 训练 1 个 epoch，只证明脚本可执行。若要用作最终文本模型，应换成 `bert-base-chinese` 或中文客服领域模型并增加训练轮数。",
            "- `语音 LSTM = 0.3906`：LibriSpeech 没有客服意图标签，此处是说话人代理分类；低分说明小样本 LSTM 声学分支效果一般。Web 端语音意图应主要依靠 ASR 转写后的文本分类。",
            "- `表情 CNN = 0.2321` 和 `图片 CNN = 0.3667`：这两个是公开数据集原始/通用任务的小样本快速训练结果，难度更高、训练不足，应作为后续改进方向保留。",
            "- `多模态训练融合 = 0.6`：只有 10 条样本，测试集每类 1 条，结果波动很大。当前系统采用规则融合更适合课堂演示，训练式融合需要扩充样本后再讨论效果。",
            "",
            "## 答辩建议",
            "",
            "- 可以强调“系统已实现可运行多模态客服闭环”，而不是笼统宣称所有模型都达到高精度。",
            "- 遇到 1.0 指标时主动解释数据规模和任务边界，老师通常会更认可这种诚实口径。",
            "- 把弱项包装成清晰的下一步：BERT 换中文预训练模型，语音接真实 ASR，表情七分类增加训练轮数，截图模型换真实业务截图数据。",
        ]
    )
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_row(item: dict[str, Any], data: dict[str, Any] | None) -> dict[str, str]:
    if data is None:
        return {
            "name": item["name"],
            "task_definition": item["task_definition"],
            "model": "未运行",
            "sample_summary": "未运行",
            "split_summary": item["split"],
            "accuracy": "未运行",
            "risk": item["risk"],
            "interpretation": item["interpretation"],
        }

    return {
        "name": item["name"],
        "task_definition": item["task_definition"],
        "model": str(data.get("model", "")),
        "sample_summary": sample_summary(data),
        "split_summary": split_summary(data, item["split"]),
        "accuracy": format_accuracy(data.get("accuracy")),
        "risk": item["risk"],
        "interpretation": item["interpretation"],
    }


def sample_summary(data: dict[str, Any]) -> str:
    sample = data.get("samples") or data.get("sample_size")
    train_size = data.get("train_size")
    test_size = data.get("test_size")
    inferred_test = infer_test_size(data)
    parts = []
    if sample is not None:
        parts.append(f"总样本 {sample}")
    if train_size is not None:
        parts.append(f"训练 {train_size}")
    if test_size is not None:
        parts.append(f"测试 {test_size}")
    elif inferred_test is not None:
        parts.append(f"测试约 {inferred_test:g}")
    if data.get("epochs") is not None:
        parts.append(f"epochs {data['epochs']}")
    if data.get("mode"):
        parts.append(f"mode={data['mode']}")
    return "；".join(parts) if parts else "未记录"


def split_summary(data: dict[str, Any], fallback: str) -> str:
    if data.get("train_size") is not None and data.get("test_size") is not None:
        total = int(data["train_size"]) + int(data["test_size"])
        return f"训练 {data['train_size']} / 测试 {data['test_size']}（约 {data['train_size'] / total:.0%}/{data['test_size'] / total:.0%}）"
    return fallback


def infer_test_size(data: dict[str, Any]) -> float | None:
    report = data.get("classification_report")
    if not isinstance(report, dict):
        return None
    supports = []
    for key, value in report.items():
        if key in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if isinstance(value, dict) and "support" in value:
            supports.append(float(value["support"]))
    return sum(supports) if supports else None


def format_accuracy(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
