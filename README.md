# 智能客服多模态接待场景

这是一个用于课程汇报的多模态意图感知人机交互智能体。系统面向智能客服接待任务，支持文字诉求、语音转写、音频文件、表情图片和业务截图五类输入，并输出用户意图、融合置信度、推荐服务、智能回复、处理步骤和是否转人工。

> 说明：本项目以选修课大作业和课堂演示为目标。报告中的准确率主要来自课程 demo 测试集，用于证明系统链路可运行，不作为真实公开业务数据集上的严格泛化指标。

## 核心功能

- 五类意图识别：咨询、投诉、退款、故障报修、不会操作。
- 多模态输入：文本、语音转写、音频声学特征、表情图片、业务截图。
- 截图识别：demo CNN + OCR/文件名关键词规则，稳定识别 fault/order/refund/howto/normal 五类业务截图。
- 多模态融合：文本5维 + 语音5维 + 表情5维 + 截图5维 + 辅助数值6维，拼接为 26 维特征后输入全连接 MLP。
- 智能客服输出：推荐服务、回复话术、处理步骤、转人工判断、融合依据展示。
- 课程演示数据：`course_demo_test_data/` 中提供 5 个完整测试文件夹，每个包含 case.json、业务截图、表情图片和音频文件。

## 技术路线

1. 文本分支：TF-IDF/LogisticRegression + BERT 分支 + 关键词规则增强。
2. 语音分支：音频文件提取 MFCC/RMS/过零率/谱质心等声学特征，优先调用 LSTM 模型，缺失时使用 MFCC 统计分类器兜底。
3. 表情分支：FER-2013 CNN/MLP 识别用户表情或情绪倾向，用于提高投诉、退款、故障等负向场景权重。
4. 截图分支：业务截图 CNN 输出页面类别，再结合 OCR/关键词规则生成更稳定的 demo 截图判断。
5. 融合分支：`customer_ai/fusion_model.py` 中的 `build_fusion_features` 构造 26 维特征，`FusionMLP` 执行全连接融合并输出五类意图概率。
6. Web 智能体：Flask 提供接口和页面，前端展示文件预览、五类得分、模态摘要和融合特征。

## 项目结构

```text
smart_customer_service/
  app.py                         # Flask Web 入口和预测接口
  customer_ai/
    intent_engine.py             # 意图感知主流程
    fusion_model.py              # 26维特征拼接 + 全连接融合
    image_analyzer.py            # 业务截图/图片识别
    audio_analyzer.py            # 音频声学分析
    emotion_analyzer.py          # 表情图片识别
    text_bert_analyzer.py        # BERT 文本分支
    knowledge_base.py            # 意图标签、关键词和回复模板
  course_demo_test_data/          # 五类完整课程演示数据
  demo_assets/                    # 演示截图和音频/表情素材
  models/                         # 小模型和模型说明
  reports/                        # 指标、截图和报告素材
  scripts/                        # 训练、验证和报告生成脚本
  static/                         # 前端 CSS/JS
  templates/                      # Flask 页面模板
  tests/                          # API 单元测试
```

## 运行方式

建议使用 Python 3.10+。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

如果本机有 NVIDIA GPU，可以安装 CUDA 12.6 版本依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-cu126.txt
```

## 测试方式

运行 API 单元测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

验证五个完整课程演示样例：

```powershell
.\.venv\Scripts\python.exe scripts\verify_packaged_course_demo_data.py
```

当前课程 demo 端到端验证结果保存在：

```text
reports/packaged_course_demo_data_predictions.json
```

## 课程演示数据

`course_demo_test_data/` 包含五个完整测试文件夹：

- `01_fault_repair`：故障报修
- `02_order_consult`：订单咨询
- `03_refund`：退款售后
- `04_howto`：操作指引/不会操作
- `05_complaint`：投诉

每个文件夹都包含：

- `case.json`：文本、语音转写、期望意图、期望截图类别
- 业务截图：用于截图五分类
- 表情图片：用于情绪辅助判断
- 音频文件：用于语音/声学分支

## 模型文件说明

为了让仓库适合 GitHub 查看，本仓库上传了课程 demo 运行所需的小模型和指标文件，排除了较大的训练权重与原始数据集。

已保留的小模型示例：

- `models/fusion_intent_mlp.pt`
- `models/demo_screenshot_intent_cnn.pt`
- `models/text_intent_augmented.joblib`
- `models/fer2013_cnn.pt`
- `models/librispeech_lstm.pt`

未上传的大文件示例：

- `models/text_bert_classifier/model.safetensors`，约 409MB
- 严格截图训练过程中的 ResNet 试验权重
- COCO/LibriSpeech/FER-2013 原始下载数据

如需复现训练，可参考：

- `docs/DATASET_DOWNLOADS.md`
- `scripts/train_text_bert.py`
- `scripts/train_audio_lstm.py`
- `scripts/train_emotion_cnn.py`
- `scripts/train_demo_screenshot_intent_cnn.py`
- `scripts/train_fusion_model.py`

## 报告与截图

报告相关素材位于：

```text
reports/report_assets/
```

最终实践报告在本地生成，报告附录中已经补充：

- GitHub 代码仓库地址
- 核心代码片段与注释
- 系统运行截图
- 交互过程截图
- 五类测试汇总图
- 需求分析流程图
- 模型结构示意图

## 重要边界

本项目适合课程演示和原型验证。真实业务部署仍需要更多公开或企业授权数据、更严格的训练/测试划分、隐私合规处理，以及真实客服截图、真实语音、真实多模态联合样本上的泛化验证。
