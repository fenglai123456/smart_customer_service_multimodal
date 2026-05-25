# 严格技术路线实现报告

生成日期：2026-05-22

## 结论

按“不造假、不走捷径、训练集不偏离截图要求”的口径，当前项目**系统链路已经实现，但严格达标仍未全部完成**。

已经真实完成并可验收的部分：

- 文本数据：SQuAD v2.0 + ABCD 公开客服对话子集已合并为 `datasets/text/strict_text_intents.csv`，共 11242 条。
- 文本模型：`bert-base-chinese` 已在严格文本集上训练，测试准确率 0.9460。
- 语音数据：LibriSpeech `dev-clean` 与 `train-clean-100` 已就位。
- 语音模型：MFCC + LSTM 已训练，当前 LibriSpeech 声学分支准确率 0.9795。注意：LibriSpeech 没有客服意图标签，只能证明截图要求中的 LSTM 语音分支。
- 表情数据：FER-2013 CSV 已就位。
- 表情模型：FER-2013 angry-vs-other CNN 已使用类别权重训练并调优阈值，当前准确率 0.8459，angry recall 0.5015。
- COCO 图片模型：使用 COCO 2017 官方图片与官方实例标注框裁剪训练 ResNet18 迁移模型，按图片来源分组留出测试，当前准确率 0.8562，已超过 82%。
- 系统：Flask 前端、文字/语音/表情/图片输入、五类意图响应、人工联动、特征级拼接全连接融合链路均已接入。

仍未严格达标的部分：

- 客服截图五分类 CNN：公开/可追溯截图样本已扩展到 1748 张，包含公开网页截图、页面内嵌公开图片、公开页面正文切片；按原始公开页面来源分组测试，当前扩展后正式结果为 0.4701，仍未达到 82%。
- 扩展前高细节子集历史最好为 0.5517；加入更多业务化公开来源后，跨来源测试变难且结果下降到 0.4701。类别权重、不冻结 ResNet、384 分辨率、跨标签完全重复图剔除均已尝试，仍未接近 82%。
- 严格融合训练集 `datasets/multimodal/customer_cases.csv` 仍为 0 行。没有同一真实案例下的文字、音频、表情/情绪、截图联合样本，因此不能训练严格融合模型。

## 截图要求对照

| 模块 | 截图要求 | 当前状态 |
| --- | --- | --- |
| 文本 | SQuAD + 客服对话公开子集，BERT | 已完成严格训练：0.9460 |
| 语音 | LibriSpeech，LSTM | 已完成声学分支训练：0.9795；不等同于客服语义意图 |
| 表情 | FER-2013，CNN | 已完成客服负向情绪二分类：0.8459，angry recall 0.5015 |
| COCO 图片 | COCO，CNN | 已完成 COCO 官方标注框裁剪训练：0.8562 |
| 客服截图 | 客服截图公开子集，CNN，识别故障/订单等信息 | 公开/可追溯数据 1748 张；当前严格五分类 0.4701，未达 82% |
| 融合 | 文字、语音、表情、图片特征拼接后输入全连接层 | 代码已实现；真实联合训练集为空 |
| 系统 | Python Flask，支持多模态输入与本地部署演示 | 已实现并通过 API 测试 |

## 当前严格数据审计

```text
[OK] text: SQuAD v2.0, ABCD public customer-dialogue subset
[OK] audio: LibriSpeech
[OK] emotion: FER-2013
[OK] image: COCO, customer screenshot public subset
[MISSING] fusion: real multimodal cases assembled from the strict sources above
```

客服截图公开子集当前计数：

- fault: 322
- order: 341
- refund: 356
- howto: 331
- normal: 398
- total: 1748

按来源形态计数：

- webpage_viewport: 793
- embedded_asset: 408
- article_tile: 543
- seed: 4

高细节筛选后计数：

- fault: 133
- order: 145
- refund: 123
- howto: 145
- normal: 209

融合样本缺口：

- `datasets/multimodal/customer_cases.csv` 当前 0 行。
- 本地严格校验最低要求每类 10 行真实联合样本；若要可信训练和答辩，建议每类 100 行以上。

## 当前严格模型审计

```text
[OK] text: accuracy=0.9460 samples=11242
[OK] audio: accuracy=0.9795 samples=728
[OK] emotion: accuracy=0.8459 samples=35887
[OK] coco_image: accuracy=0.8562 samples=1500
[NOT_READY] screenshot_intent: accuracy=0.4701 samples=661
[NOT_READY] fusion: accuracy=1.0000 samples=10
```

说明：fusion 的 1.0000 来自 `data/demo/multimodal_samples.csv`，已标记为 `strict_sources=false`，不能计入严格达标。

## 本轮新增的真实改进

- 扩展 `scripts/collect_public_customer_page_screenshots.py`，新增 WooCommerce、Adobe Commerce 等更贴近后台业务的公开文档页，包括订单页、退款/credit memo、错误排查、设置与商品操作页。
- 重新采集公开网页截图、页面内嵌图和正文切片，客服截图公开子集从 1313 张扩展到 1748 张。
- `scripts/train_customer_screenshot_intent_cnn.py` 修复默认不过滤时收集不到图片的问题，新增 `--model_path`、`--report_path`、`--class_weight`、`--mirror_aug`、`--drop_cross_label_duplicates`。
- 关闭默认水平翻转增强，因为镜像会破坏截图文字和后台 UI 方向。
- 尝试多组正规训练：扩展后 ResNet18 冻结 0.4701，384 分辨率 0.4274，跨标签完全重复图剔除 0.3491；扩展前类别权重试验 0.5287，不冻结微调 0.4598。
- 冻结 ImageNet ResNet18 特征 + 逻辑回归快速诊断在全量 1748 张公开截图上只有 0.2659，进一步说明当前公开截图视觉可分性不足。
- 刷新 `reports/strict_dataset_audit.json`、`reports/strict_model_audit.json`、`reports/customer_screenshot_subset_audit.json` 与 `reports/customer_screenshot_completion_status.json`。

## 为什么仍未达标

当前公开页面虽然来源真实可追溯，但很多截图是帮助文档页、营销页或通用后台 UI。它们在视觉上常常共享同一套导航、文档布局、按钮和表格样式，真正能区分“故障、订单、退款、操作、普通”的信息主要在页面文字语义里，而不是图像结构里。严格测试又按公开页面来源分组，模型不能靠同一页面不同视口记忆，因此泛化准确率明显低于 82%。

另外，公开页面中还存在同一素材图跨多个标签页面重复出现的问题。剔除这些跨标签完全重复图后，准确率进一步下降，说明它们不能作为可靠的五分类业务截图证据。

## 禁止计入严格达标的内容

- 模板生成文本。
- 合成客服截图。
- 没有客服业务标签的泛 UI 截图数据集。
- `data/demo/multimodal_samples.csv` 演示样本。
- 将同一公开网页的不同视口、内嵌图、正文切片当成互相独立的真实客户案例。
- 只用 COCO 分类结果替代客服截图识别。
- 随机把不同来源的文本、音频、表情图、截图拼成一个所谓“多模态案例”。
- 将随机切分或演示集结果写成严格公开数据集达标结果。

## 严格脚本入口

```powershell
.\.venv\Scripts\python.exe scripts\check_runtime_environment.py
.\.venv\Scripts\python.exe scripts\prepare_abcd_customer_intents.py
.\.venv\Scripts\python.exe scripts\prepare_strict_text_intents.py
.\.venv\Scripts\python.exe scripts\prepare_public_customer_screenshot_seeds.py
.\.venv\Scripts\python.exe scripts\collect_public_customer_page_screenshots.py
.\.venv\Scripts\python.exe scripts\collect_public_customer_embedded_images.py --max_per_source 8 --workers 8
.\.venv\Scripts\python.exe scripts\collect_public_customer_article_tiles.py --tiles_per_source 5 --workers 4
.\.venv\Scripts\python.exe scripts\audit_customer_screenshot_subsets.py
.\.venv\Scripts\python.exe scripts\prepare_multimodal_case_template.py
.\.venv\Scripts\python.exe scripts\validate_multimodal_cases.py --strict
.\.venv\Scripts\python.exe scripts\check_dataset_layout.py --strict
.\.venv\Scripts\python.exe scripts\check_strict_model_metrics.py --strict
```

严格训练入口：

```powershell
.\.venv\Scripts\python.exe scripts\train_text_bert.py --strict_sources --model_name bert-base-chinese --epochs 3 --batch_size 16
.\.venv\Scripts\python.exe scripts\train_audio_lstm.py --subset dev-clean --max_samples 1200 --top_k_speakers 8 --epochs 20 --hidden_size 128 --num_layers 2
.\.venv\Scripts\python.exe scripts\train_emotion_cnn.py --binary_angry --class_weight --epochs 24 --batch_size 256 --learning_rate 0.0006
.\.venv\Scripts\python.exe scripts\tune_emotion_binary_threshold.py
.\.venv\Scripts\python.exe scripts\train_image_cnn.py --training_unit object_crop --architecture resnet18_transfer --freeze_backbone --max_samples 1500 --top_k 3 --epochs 12 --batch_size 64 --image_size 224 --learning_rate 0.001
.\.venv\Scripts\python.exe scripts\train_customer_screenshot_intent_cnn.py --strict_sources --architecture resnet18_transfer --freeze_backbone --epochs 20 --patience 7 --batch_size 32 --image_size 224 --learning_rate 0.001 --source_kinds webpage_viewport,embedded_asset,seed --min_edge_score 12 --min_contrast 25 --min_images_per_label 50 --split_strategy source_group
.\.venv\Scripts\python.exe scripts\train_fusion_model.py --strict_sources
```

`train_fusion_model.py --strict_sources` 目前会正确失败，因为真实多模态联合样本尚未提供。

## 下一步

1. 补真实/老师提供的客服业务截图，尤其是故障报错、订单后台、退款/售后后台、操作页面真实截图；建议每类至少 200 到 500 张，并保留独立测试来源。
2. 每张截图需要有明确标签和来源说明，尽量来自真实后台、工单系统、订单系统或老师认可的公开业务系统截图，而不是帮助文档整页截图。
3. 补同一客服案例下的多模态联合样本：文字、语音或转写、情绪标签/表情图、业务截图、五类意图标签、来源说明。
4. 在补齐真实业务截图后重训 `train_customer_screenshot_intent_cnn.py --strict_sources`；补齐真实联合样本后再运行 `train_fusion_model.py --strict_sources`。
5. 重新运行 `check_dataset_layout.py --strict` 与 `check_strict_model_metrics.py --strict`，只有全部 OK 才能写“完全符合截图要求”。
