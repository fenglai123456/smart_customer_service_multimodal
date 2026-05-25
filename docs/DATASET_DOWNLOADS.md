# 数据集下载与存放目录

这些数据集对应实验要求中的公开数据来源。为了课堂演示更稳，项目已经内置轻量样例数据；真实公开数据集用于进一步训练、评估和写实验报告。

## 1. SQuAD 文本问答数据集

官方页面：https://rajpurkar.github.io/SQuAD-explorer/

下载文件：

- 训练集：https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
- 验证集：https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

保存到：

```text
smart_customer_service/datasets/text/squad/train-v2.0.json
smart_customer_service/datasets/text/squad/dev-v2.0.json
```

## 1.1 客服对话公开子集（ABCD）

截图要求写的是 `SQuAD + 客服对话公开子集`。本项目使用 ABCD（Action-Based Conversations Dataset）作为客服对话公开子集。

- GitHub：https://github.com/asappresearch/abcd
- 论文：https://arxiv.org/abs/2104.00783

保存到：

```text
smart_customer_service/datasets/text/customer_dialogue_public/abcd/
```

下载：

```powershell
git clone --depth 1 https://github.com/asappresearch/abcd.git datasets/text/customer_dialogue_public/abcd
```

转换为五类客服意图，并合并 SQuAD 问题：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_abcd_customer_intents.py
.\.venv\Scripts\python.exe scripts\prepare_strict_text_intents.py
```

严格文本训练只允许使用：

```text
smart_customer_service/datasets/text/strict_text_intents.csv
```

## 2. LibriSpeech 语音数据集

官方页面：https://www.openslr.org/12/

建议先下小型验证集，课堂项目够用；如果后续训练语音模型，再下 `train-clean-100`。

下载文件：

- 小型验证集 dev-clean：https://openslr.trmal.net/resources/12/dev-clean.tar.gz
- 训练集 train-clean-100：https://openslr.trmal.net/resources/12/train-clean-100.tar.gz
- 国内镜像 train-clean-100：https://openslr.magicdatatech.com/resources/12/train-clean-100.tar.gz

保存到：

```text
smart_customer_service/datasets/audio/librispeech/dev-clean.tar.gz
smart_customer_service/datasets/audio/librispeech/train-clean-100.tar.gz
```

## 3. FER-2013 表情数据集

原 Kaggle 比赛页面：https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

由于原 Kaggle 比赛已经截止，新账号可能无法加入比赛下载数据。项目当前推荐使用 SourceForge 镜像：

- 镜像下载：https://sourceforge.net/projects/emotion-detector/files/fer2013.csv/download
- 文件页：https://sourceforge.net/projects/emotion-detector/files/

保存到：

```text
smart_customer_service/datasets/emotion/fer2013/fer2013.csv
```

也可以在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe scripts\download_fer2013.py
```

如果使用其他 Kaggle 镜像数据集，可能是图片文件夹结构，不一定包含 `fer2013.csv`。这种情况下请保存到：

```text
smart_customer_service/datasets/emotion/fer2013/images/
```

## 4. COCO 图片分类/截图辅助数据

官方页面：https://cocodataset.org/#download

下载文件：

- 验证集图片 val2017：http://images.cocodataset.org/zips/val2017.zip
- 2017 标注 annotations：http://images.cocodataset.org/annotations/annotations_trainval2017.zip

保存到：

```text
smart_customer_service/datasets/images/coco/val2017.zip
smart_customer_service/datasets/images/coco/annotations_trainval2017.zip
```

## 5. 客服截图公开/老师提供子集

实验要求中提到“客服截图公开子集”。严格口径下，这里只能放真实公开数据、老师提供数据，或经允许采集并标注的真实客服业务截图。不能放合成截图，也不能用泛 UI 截图数据集替代。

```text
smart_customer_service/datasets/images/customer_screenshots/
```

建议目录结构：

```text
customer_screenshots/
  fault/
  refund/
  order/
  howto/
  normal/
```

每个类别至少应有真实图片，建议每类至少几十到几百张；如果要追求 82%+ 并可答辩，最好每类 500 张以上并保留独立测试集。

当前项目提供两个补样本脚本：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_public_customer_screenshot_seeds.py
.\.venv\Scripts\python.exe scripts\collect_public_customer_page_screenshots.py
```

第一条从 ABCD 官方仓库复制公开客服系统示例图；第二条从公开电商/支付/帮助中心页面截取可追溯网页截图。它们只能作为种子样本，不应被写成完整训练集。

来源清单：

```text
smart_customer_service/datasets/images/customer_screenshots/manifest.csv
smart_customer_service/datasets/images/customer_screenshots/public_web_manifest.json
```

## 6. 多模态联合样本

严格融合训练需要真实联合样本，不允许使用 `data/demo/multimodal_samples.csv` 作为达标训练集。项目提供空模板：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_multimodal_case_template.py
```

生成文件：

```text
smart_customer_service/datasets/multimodal/customer_cases.csv
```

字段包括文本、语音转写、音频路径、表情标签、表情图片路径、截图路径、五类意图标签和来源说明。

## 检查数据是否放对

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset_layout.py
.\.venv\Scripts\python.exe scripts\check_dataset_layout.py --strict
```

`--strict` 会在缺少客服截图公开子集或真实多模态联合样本时返回失败，这是预期行为，不应绕过。
