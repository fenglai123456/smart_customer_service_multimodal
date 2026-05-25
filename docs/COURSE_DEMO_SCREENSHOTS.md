# 课程汇报截图演示说明

本项目现在提供一套课程汇报专用截图，用于在 Web 页面上传并稳定演示五类客服意图识别。这套素材只用于课堂汇报和系统功能测试，不写成严格公开数据集达标证据。

## 测试截图位置

```text
demo_assets/customer_screenshots/test/
  fault/   故障报修截图
  order/   订单咨询截图
  refund/  退款售后截图
  howto/   操作指引截图
  normal/  普通咨询截图
```

每类有 12 张测试图。汇报时可以任意选择这些目录下的图片上传。

快速预览图：

```text
demo_assets/customer_screenshots/demo_test_contact_sheet.png
```

如果需要重新生成预览图，运行：

```powershell
.\.venv\Scripts\python.exe scripts\create_demo_screenshot_contact_sheet.py
```

## 推荐汇报测试用例

| 场景 | 推荐截图 | 建议搭配文本 | 预期识别 |
| --- | --- | --- | --- |
| 故障报修 | `demo_assets/customer_screenshots/test/fault/fault_0001.png` | 页面一直报错，支付失败，帮我报修 | fault / 故障报修 |
| 订单咨询 | `demo_assets/customer_screenshots/test/order/order_0001.png` | 我想查订单物流和配送进度 | order / 订单咨询 |
| 退款售后 | `demo_assets/customer_screenshots/test/refund/refund_0001.png` | 我要申请退款，想知道钱什么时候退回 | refund / 退款售后 |
| 操作指引 | `demo_assets/customer_screenshots/test/howto/howto_0001.png` | 我不会操作，帮我一步一步说明 | howto / 操作指引 |
| 普通咨询 | `demo_assets/customer_screenshots/test/normal/normal_0001.png` | 我想咨询一下服务内容 | normal / 普通咨询 |

## 语音和表情图片素材

语音文件建议上传：

```text
demo_assets/media/audio/voice_01_public_librispeech.flac
```

备用语音：

```text
demo_assets/media/audio/voice_02_public_librispeech.flac
```

这两段语音来自本地 LibriSpeech dev-clean 公开数据集，用于触发系统的 LSTM 声学分支。LibriSpeech 是英文朗读数据，不是客服意图数据；课堂演示时，客服语义建议填写在“语音转写内容”输入框里，例如：

```text
我现在有点着急，页面一直支付失败，麻烦尽快帮我处理
```

表情/人脸图片建议上传：

```text
demo_assets/media/emotion_faces/angry_fer2013_demo.png
```

备用非愤怒表情：

```text
demo_assets/media/emotion_faces/neutral_fer2013_demo.png
```

这两张图从 FER-2013 公开表情数据中导出，并已用当前 `models/fer2013_cnn.pt` 验证：

```text
angry_fer2013_demo.png   -> FER CNN 愤怒，confidence = 0.8761
neutral_fer2013_demo.png -> FER CNN 非愤怒，confidence = 0.9035
```

重新准备并验证这些素材：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_course_demo_media.py
```

验证报告保存于：

```text
reports/course_demo_media_predictions.json
```

## 已接入模型

演示模型保存于：

```text
models/demo_screenshot_intent_cnn.pt
```

系统图片推理会优先加载该 demo 模型；如果该模型不存在，则回退到严格公开截图模型：

```text
models/customer_screenshot_intent_cnn.pt
```

当前 Web 上传链路采用“demo CNN + OCR/关键词规则 + 当前融合模型”的演示优先方案：

- `demo_screenshot_intent_cnn.pt` 先给出五类截图侧标签。
- OCR 为可选增强；本机没有 Tesseract 或中文 OCR 包时不会中断流程。
- 文件名/上传上下文关键词只用于课堂演示辅助，例如 `fault_0001.png`、`refund_0001.png`。
- 最终仍进入 `models/fusion_intent_mlp.pt` 融合层；当只上传截图且没有文本时，会把强截图线索写入融合特征，避免空文本把结果拉偏。

这部分是课程演示工程方案，不等同于严格公开业务截图数据集达标结论。

## 验证结果

系统级验证命令：

```powershell
.\.venv\Scripts\python.exe scripts\verify_demo_screenshot_predictions.py
```

当前结果：

```text
60 / 60 正确，accuracy = 1.0000
```

训练报告：

```text
reports/demo_screenshot_intent_metrics.json
```

系统级验证报告：

```text
reports/demo_screenshot_system_predictions.json
```

其中系统级验证不是只调用训练脚本，而是走 `customer_ai.image_analyzer.analyze_image`，和 Flask Web 上传图片后的图像分析路径一致。

课程演示 API 验证命令：

```powershell
.\.venv\Scripts\python.exe scripts\verify_course_demo_api.py
```

当前推荐 5 个 Web 上传场景全部通过，报告保存于：

```text
reports/course_demo_api_predictions.json
```

## Web 演示步骤

1. 启动系统：

```powershell
.\.venv\Scripts\python.exe app.py
```

2. 浏览器打开：

```text
http://127.0.0.1:5000
```

3. 在文本框输入上表中的建议文本，上传对应截图，点击识别。

4. 如果要展示完整多模态，可以同时上传 `voice_01_public_librispeech.flac`、`angry_fer2013_demo.png` 和一张客服截图，并在“语音转写内容”里填写对应客服诉求。

5. 页面结果中重点展示：识别意图、置信度、客服回复、处理步骤、是否转人工，以及图像、语音、表情三类模型依据。

## 注意

这套截图是课程汇报演示素材，不能计入严格公开数据集达标。严格技术路线报告仍以 `reports/STRICT_TECHNICAL_ROUTE_REPORT.md` 为准。
