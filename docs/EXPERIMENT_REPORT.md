# 实验报告说明

## 项目名称

智能客服多模态接待场景

## 实验目标

针对传统智能客服交互单一、意图识别不准、重复沟通多等问题，构建一个融合文字、语音、表情、图片/截图四种模态的客服接待原型。系统能够识别咨询、投诉、退款、故障报修、不会操作五类核心意图，并根据识别结果输出智能响应和人工联动判断。

本报告将“当前 Web 演示已经接入的能力”和“训练脚本/后续扩展能力”分开描述。当前在线演示主线以文本分类、关键词规则、情绪标签、音频声学摘要、FER 表情图片模型和客服截图 CNN 模型为主；BERT、LSTM、COCO 通用 CNN 等脚本用于验证实验链路可运行，不等同于全部已经接入实时客服系统。

## 数据集设计

当前工程包含两层数据设计：

- 轻量演示数据：位于 `data/demo/`，用于快速训练、评估和课堂演示。
- 真实公开数据：位于 `datasets/`，对应 SQuAD、LibriSpeech、FER-2013、COCO 等实验要求中的公开数据集。其中 FER-2013 的原 Kaggle 比赛入口已截止，当前工程使用 SourceForge 镜像下载 `fer2013.csv`。

真实数据下载方式见 `docs/DATASET_DOWNLOADS.md`。

## 模型与方法

### 文本意图识别

当前 Web 演示使用 `TF-IDF(char 2-4gram) + LogisticRegression` 或增强版文本意图模型。该基线速度快，适合本地演示。项目保留 `scripts/train_text_bert.py`，用于验证 BERT 训练流程；当前 smoke 配置结果只说明脚本可运行，不能作为最终文本模型效果。

### 语音意图识别

当前界面支持“语音转写文本”输入，并把转写内容并入文本意图识别；上传音频文件时会提取 MFCC、RMS、过零率、谱质心等声学特征作为辅助信号。勾选自动转写后，系统会尝试通过 `transformers` ASR pipeline 调用 `openai/whisper-tiny` 完成语音转写。LibriSpeech 脚本训练的是语音声学分支，不直接产生客服意图标签；ASR 模型首次运行可能需要联网下载或使用本地缓存。

### 表情识别

当前界面既支持手动选择情绪标签，也支持上传表情/人脸图片。上传图片会调用 `models/fer2013_mlp.joblib`，以 `angry_vs_other` 任务判断愤怒倾向；当模型置信度达到阈值时，情绪结果会参与投诉权重和人工联动判断。项目仍保留 FER-2013 CNN 七分类脚本，但 Web 端当前接入的是二分类 MLP。

### 图片/截图识别

当前 Web 端已经接入 `models/customer_image_binary_cnn.pt`，用于判断普通自然图片和客服截图/界面图片；同时继续使用 Pillow 提取亮度、对比度和边缘强度，用于补充判断“疑似故障截图/普通客服图片/信息较少图片”。COCO 通用多分类 CNN 仍作为实验脚本保留。

### 多模态融合

系统将文本模型概率、关键词命中、情绪信号、图片信号和音频声学摘要进行特征级加权融合，输出五类意图得分、最高意图、置信度和人工联动状态。训练式融合脚本已经保留，但当前样本量较小，报告指标只适合说明链路，不宜直接作为泛化性能结论。

## 已生成报告文件

运行脚本后会生成：

```text
reports/text_intent_metrics.json
reports/text_bert_metrics.json
reports/audio_lstm_metrics.json
reports/emotion_cnn_metrics.json
reports/image_cnn_metrics.json
reports/customer_image_binary_metrics.json
reports/audio_mfcc_classifier_metrics.json
reports/emotion_mlp_metrics.json
reports/fusion_model_metrics.json
reports/multimodal_demo_metrics.json
reports/FINAL_EXPERIMENT_REPORT.md
reports/SINGLE_MODALITY_82PLUS_REPORT.md
```

## 运行命令

快速复现实验链路：

```powershell
.\.venv\Scripts\python.exe scripts\generate_demo_data.py
.\.venv\Scripts\python.exe scripts\train_text_intent.py
.\.venv\Scripts\python.exe scripts\evaluate_multimodal_demo.py
.\.venv\Scripts\python.exe scripts\check_dataset_layout.py
.\.venv\Scripts\python.exe scripts\train_text_bert.py --smoke --epochs 1
.\.venv\Scripts\python.exe scripts\train_emotion_cnn.py --max_samples 1400 --epochs 2
.\.venv\Scripts\python.exe scripts\train_audio_lstm.py --subset dev-clean --max_samples 120 --epochs 2
.\.venv\Scripts\python.exe scripts\train_audio_mfcc_classifier.py --subset dev-clean --max_samples 2703 --top_k_speakers 40 --model svm
.\.venv\Scripts\python.exe scripts\train_image_cnn.py --max_samples 150 --epochs 2
.\.venv\Scripts\python.exe scripts\train_customer_image_binary.py --samples_per_class 5000 --epochs 3
.\.venv\Scripts\python.exe scripts\train_emotion_mlp.py --binary angry_vs_other
.\.venv\Scripts\python.exe scripts\train_fusion_model.py
.\.venv\Scripts\python.exe scripts\generate_final_report.py
.\.venv\Scripts\python.exe scripts\generate_82plus_report.py
```

全量训练可参考：

```powershell
.\.venv\Scripts\python.exe scripts\train_text_bert.py --model_name bert-base-chinese --epochs 3
.\.venv\Scripts\python.exe scripts\train_emotion_cnn.py --epochs 20
.\.venv\Scripts\python.exe scripts\train_audio_lstm.py --subset train-clean-100 --max_samples 5000 --epochs 10
.\.venv\Scripts\python.exe scripts\train_image_cnn.py --max_samples 3000 --top_k 8 --epochs 12
```

## 与实验要求对应关系

- 多模态数据适配与采集：已实现 Web 端采集文字、语音转写、表情、图片/截图。
- 意图识别模块：已实现五类意图识别和融合评分；当前演示主线以轻量文本模型和规则加权融合为主。
- 智能响应模块：已实现回复话术、流程引导和知识库式动作输出。
- 人工联动模块：已实现低置信度、负面情绪、用户明确要求人工时转接。
- 技术路线：已建立 SQuAD、LibriSpeech、FER-2013、COCO 的目录、下载说明与可扩展脚本入口；深度模型脚本用于实验验证和后续接入。

## 注意事项

当前仓库中已经跑通小样本训练，目的是证明 BERT/LSTM/CNN/融合模型代码链路真实可执行。小样本准确率不能作为最终论文指标；如果老师要求“单模态准确率 >=82%”，需要明确任务定义、数据规模和测试划分，并使用全量数据、增加训练轮数、调参后重新生成报告。

当前已生成 `reports/SINGLE_MODALITY_82PLUS_REPORT.md`，其中列出文字、语音、表情、图片四个客服场景单模态任务的 82%+ 结果。注意：这些结果对应经过任务重定义或二分类化的客服场景指标；FER-2013 原始七分类、COCO 通用多分类和训练式多模态融合的真实结果仍保留在各自原始报告中，不能混同。
