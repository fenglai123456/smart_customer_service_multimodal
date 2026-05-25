from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "demo"


TEXT_ROWS = [
    ("请问这个订单什么时候发货", "consult"),
    ("我想了解一下会员权益和活动规则", "consult"),
    ("门店今天营业到几点", "consult"),
    ("这个商品还有库存吗", "consult"),
    ("物流停了三天请帮我查一下", "consult"),
    ("客服一直不回复我要投诉", "complaint"),
    ("服务态度太差了我非常生气", "complaint"),
    ("排队很久没人处理我要反馈", "complaint"),
    ("这次体验很糟糕我不满意", "complaint"),
    ("商品质量有问题我要投诉商家", "complaint"),
    ("我想申请退款", "refund"),
    ("订单取消后钱什么时候退回", "refund"),
    ("我要退货退款怎么处理", "refund"),
    ("多扣费了请帮我退款", "refund"),
    ("没有收到货想要赔付", "refund"),
    ("页面报错打不开", "repair"),
    ("设备坏了需要维修", "repair"),
    ("上传截图显示系统故障", "repair"),
    ("软件一直闪退怎么修复", "repair"),
    ("支付失败提示网络异常", "repair"),
    ("我不会操作这个功能", "howto"),
    ("在哪里修改收货地址", "howto"),
    ("怎么绑定银行卡", "howto"),
    ("请告诉我申请售后的步骤", "howto"),
    ("按钮找不到需要你教我", "howto"),
    ("优惠券在哪里查看", "consult"),
    ("我对处理结果不满意要求升级", "complaint"),
    ("退款一直没到账", "refund"),
    ("小程序加载失败显示白屏", "repair"),
    ("怎么开发票我找不到入口", "howto"),
]


MULTIMODAL_ROWS = [
    ("订单物流一直没有更新", "帮我查一下", "neutral", "normal", "consult"),
    ("服务太差我要投诉", "我很生气", "angry", "normal", "complaint"),
    ("我要申请退款", "订单已经取消", "sad", "normal", "refund"),
    ("支付页面报错", "截图里有错误提示", "anxious", "error_screenshot", "repair"),
    ("我不会修改地址", "能教一下吗", "neutral", "normal", "howto"),
    ("系统一直闪退", "很着急", "anxious", "error_screenshot", "repair"),
    ("客服没人理我", "要求人工", "angry", "normal", "complaint"),
    ("钱什么时候退回", "退款不到账", "sad", "normal", "refund"),
    ("这个套餐多少钱", "咨询活动", "happy", "normal", "consult"),
    ("找不到售后入口", "不会操作", "neutral", "low_info", "howto"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "text_intent_samples.csv", ["text", "label"], TEXT_ROWS)
    write_csv(
        OUT_DIR / "multimodal_samples.csv",
        ["text", "voice_text", "emotion", "image_signal", "label"],
        MULTIMODAL_ROWS,
    )
    print(f"Demo data written to {OUT_DIR}")


def write_csv(path: Path, headers: list[str], rows: list[tuple[str, ...]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as writer_file:
        writer = csv.writer(writer_file)
        writer.writerow(headers)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
