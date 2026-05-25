# 严格多模态联合样本采集说明

截图要求里的“多模态融合”不是把任意文本、任意音频、任意表情图、任意截图拼在一起，而是同一个客服案例下的联合记录。

## 一行样本应代表什么

`datasets/multimodal/customer_cases.csv` 中每一行应代表一个真实/可追溯客服案例，至少包含：

- `label`：五类意图之一：`consult`、`complaint`、`refund`、`repair`、`howto`
- `text`：用户输入文本，或客服对话中的用户问题
- `voice_text`：语音转写文本，可为空，但 `text` 和 `voice_text` 至少有一个
- `audio_path`：真实音频文件路径，不能用无关 LibriSpeech 音频随机配对
- `emotion_label` 或 `emotion_image_path`：用户情绪标签或真实表情/情绪图片
- `screenshot_path`：同一案例对应的业务截图，例如订单页、退款页、报错页、操作页
- `source_dataset` / `source_note`：来源说明，能解释为什么这个样本可用于严格训练

## 推荐目录

```text
datasets/multimodal/
  customer_cases.csv
  audio/
  emotion_images/
  screenshots/
```

CSV 中可以写相对路径，例如：

```text
datasets/multimodal/audio/case_0001.wav
datasets/multimodal/screenshots/case_0001.png
```

## 最低数量

- 本地严格校验最低要求：每类至少 10 条真实联合样本。
- 若要可信训练并向老师说明泛化能力：建议每类 100 条以上。
- 若要冲击稳定 82%+：建议每类 300 到 500 条，并保留独立测试集。

## 不能计入严格训练

- demo 样本。
- 自己模板生成的客服文本。
- 自己合成的截图。
- 将不同来源的音频、文本、截图随机拼成一行。
- 只因为标签相同，就把 LibriSpeech 音频和客服截图配成一个案例。
- 多个视口的同一网页截图当成多个独立真实客户案例。

## 校验命令

```powershell
.\.venv\Scripts\python.exe scripts\validate_multimodal_cases.py --strict
```

校验通过后再训练严格融合模型：

```powershell
.\.venv\Scripts\python.exe scripts\train_fusion_model.py --strict_sources
```
