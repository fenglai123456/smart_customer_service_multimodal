# 模型文件说明

本目录保留课程 demo 可以直接运行的小模型，便于老师下载仓库后查看和演示。

## 已上传

- `fusion_intent_mlp.pt`：多模态 26 维特征拼接后的全连接融合模型。
- `demo_screenshot_intent_cnn.pt`：课程演示业务截图五分类模型。
- `text_intent_augmented.joblib`：增强文本意图分类器。
- `text_intent_tfidf_lr.joblib`：轻量 TF-IDF 文本意图分类器。
- `fer2013_cnn.pt` / `fer2013_mlp.joblib`：表情图片识别分支。
- `librispeech_lstm.pt` / `audio_mfcc_classifier.joblib`：语音声学分支。
- `customer_image_binary_cnn.pt`：客服截图/普通图片二分类辅助模型。

## 未上传

以下文件较大或属于严格训练过程中的中间产物，不适合直接提交到普通 GitHub 仓库：

- `text_bert_classifier/model.safetensors`
- `customer_screenshot_intent_cnn*.pt`
- `coco_cnn.pt`
- `*features*.npz`

如需复现这些模型，请查看 `scripts/` 下对应训练脚本和 `docs/DATASET_DOWNLOADS.md`。
