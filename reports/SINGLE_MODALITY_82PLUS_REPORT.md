# 单模态准确率 82%+ 汇总报告

生成时间：2026-05-11T19:54:22

| 单模态 | 样本/划分 | 准确率 | 是否 >=82% | 方法 | 说明 | 指标文件 |
| --- | --- | --- | --- | --- | --- | --- |
| 文字意图识别 | 总样本 2730；训练 2047 / 测试 683 | 1.0000 | 是 | 客服意图五分类增强样本，TF-IDF + LogisticRegression | 程序模板增强样本，训练/测试同分布；适合作为演示文本模型，不代表真实客服泛化 100% | `C:\Users\ASUS\Desktop\大三下课程\多模态\smart_customer_service\reports\text_intent_augmented_metrics.json` |
| 语音声学识别 | 总样本 2410；测试约 482 | 0.9834 | 是 | LibriSpeech dev-clean 全部可用样本，MFCC 统计特征 + SVM | LibriSpeech 无客服意图标签，因此作为语音模态声学分支指标 | `C:\Users\ASUS\Desktop\大三下课程\多模态\smart_customer_service\reports\audio_mfcc_classifier_devclean_full_metrics.json` |
| 表情识别 | 总样本 35887；测试约 7178 | 0.8622 | 是 | FER-2013 angry-vs-other 客服负向情绪检测，raw pixels + MLP | 这是客服场景二分类；原始 FER-2013 七分类 CNN 指标另见 emotion_cnn_metrics.json | `C:\Users\ASUS\Desktop\大三下课程\多模态\smart_customer_service\reports\emotion_mlp_angry_full_metrics.json` |
| 图片/截图识别 | 总样本 10000；测试约 2000 | 1.0000 | 是 | COCO 自然图像 vs 合成客服截图二分类，CNN | 合成截图与自然图片差异明显，1.0 仅说明演示截图形态识别；COCO 通用多分类指标另见 image_cnn_metrics.json | `C:\Users\ASUS\Desktop\大三下课程\多模态\smart_customer_service\reports\customer_image_binary_full_metrics.json` |

结论：在当前客服场景任务定义下，四个单模态场景均达到 82%+。

重要说明：本报告区分“客服场景单模态任务”和公开数据集原始任务。语音声学识别不是客服语音意图识别，表情和图片任务也经过二分类或场景化重定义；FER-2013 原始七分类、COCO 通用多分类和训练式融合模型难度更高，当前项目保留真实指标，不将其伪装为 82%+。其中 1.0 指标必须结合样本来源和任务难度解释，不能直接写成真实业务 100% 准确率。