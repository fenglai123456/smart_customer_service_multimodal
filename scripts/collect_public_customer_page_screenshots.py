from __future__ import annotations

import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "datasets" / "images" / "customer_screenshots"
MANIFEST = OUT_DIR / "public_web_manifest.json"

VARIANTS = [
    ("desktop", "1280,900"),
    ("wide", "1440,1000"),
    ("laptop", "1366,768"),
    ("tablet", "900,1200"),
    ("mobile", "390,844"),
    ("small_desktop", "1024,768"),
]

SOURCES = [
    ("fault", "WooCommerce troubleshooting", "https://woocommerce.com/document/troubleshooting/"),
    ("fault", "WooCommerce self-service troubleshooting", "https://woocommerce.com/document/woocommerce-self-service-guide/"),
    ("fault", "Shopify payment troubleshooting", "https://help.shopify.com/en/manual/payments/troubleshooting"),
    ("fault", "Shopify checkout troubleshooting", "https://help.shopify.com/en/manual/checkout-settings/troubleshooting"),
    ("fault", "Stripe card declines", "https://support.stripe.com/questions/card-declines"),
    ("order", "Shopify orders", "https://help.shopify.com/en/manual/orders"),
    ("order", "Shopify manage orders", "https://help.shopify.com/en/manual/orders/manage-orders"),
    ("order", "WooCommerce managing orders", "https://woocommerce.com/document/managing-orders/"),
    ("order", "WooCommerce orders", "https://woocommerce.com/document/orders/"),
    ("order", "Stripe dashboard payments", "https://support.stripe.com/questions/viewing-payments-in-the-dashboard"),
    ("refund", "Shopify refund and cancel orders", "https://help.shopify.com/en/manual/orders/refund-cancel-order"),
    ("refund", "Shopify refunds", "https://help.shopify.com/en/manual/payments/refunds"),
    ("refund", "WooCommerce refunds", "https://woocommerce.com/document/woocommerce-refunds/"),
    ("refund", "WooCommerce automated refunds", "https://woocommerce.com/document/automated-refunds/"),
    ("refund", "Stripe refund a customer", "https://support.stripe.com/questions/refund-a-customer"),
    ("howto", "Shopify help center manual", "https://help.shopify.com/en/manual"),
    ("howto", "Shopify online store help", "https://help.shopify.com/en/manual/online-store"),
    ("howto", "WooCommerce documentation", "https://woocommerce.com/documentation/woocommerce/"),
    ("howto", "WooCommerce pages guide", "https://woocommerce.com/document/woocommerce-pages/"),
    ("howto", "Stripe support homepage", "https://support.stripe.com/"),
    ("normal", "WooCommerce homepage", "https://woocommerce.com/"),
    ("normal", "WooCommerce product page", "https://woocommerce.com/products/woocommerce/"),
    ("normal", "Shopify homepage", "https://www.shopify.com/"),
    ("normal", "Shopify pricing", "https://www.shopify.com/pricing"),
    ("normal", "Stripe homepage", "https://stripe.com/"),
    ("fault", "Shopify payments troubleshooting", "https://help.shopify.com/en/manual/payments/shopify-payments/troubleshooting"),
    ("fault", "Stripe failed payments", "https://support.stripe.com/questions/failed-payments"),
    ("fault", "Stripe payment failures docs", "https://docs.stripe.com/payments/payment-methods#payment-failures"),
    ("fault", "Stripe disputes docs", "https://docs.stripe.com/disputes"),
    ("fault", "WooCommerce system status", "https://woocommerce.com/document/understanding-the-woocommerce-system-status-report/"),
    ("order", "Shopify processing orders", "https://help.shopify.com/en/manual/orders/processing-orders"),
    ("order", "Shopify order status page", "https://help.shopify.com/en/manual/fulfillment/setup/order-status-page"),
    ("order", "Shopify edit orders", "https://help.shopify.com/en/manual/orders/edit-orders"),
    ("order", "WooCommerce order statuses", "https://woocommerce.com/document/managing-orders/order-statuses/"),
    ("order", "Stripe payments overview", "https://docs.stripe.com/payments"),
    ("refund", "Shopify returns", "https://help.shopify.com/en/manual/orders/refund-cancel-order"),
    ("refund", "WooCommerce returns warranty requests", "https://woocommerce.com/document/woocommerce-rma/"),
    ("refund", "WooCommerce refund policy", "https://woocommerce.com/document/woocommerce-refunds/"),
    ("refund", "Stripe refunds docs", "https://docs.stripe.com/refunds"),
    ("refund", "Stripe cancel a refund", "https://support.stripe.com/questions/cancel-a-refund"),
    ("howto", "Shopify add products", "https://help.shopify.com/en/manual/products/add-update-products"),
    ("howto", "Shopify themes", "https://help.shopify.com/en/manual/online-store/themes"),
    ("howto", "WooCommerce shipping zones", "https://woocommerce.com/document/setting-up-shipping-zones/"),
    ("howto", "WooCommerce payments", "https://woocommerce.com/document/payments/"),
    ("howto", "Stripe payment links", "https://docs.stripe.com/payment-links"),
    ("normal", "Shopify features", "https://www.shopify.com/online"),
    ("normal", "Shopify enterprise", "https://www.shopify.com/plus"),
    ("normal", "WooCommerce solutions", "https://woocommerce.com/solutions/"),
    ("normal", "Stripe pricing", "https://stripe.com/pricing"),
    ("normal", "Stripe docs", "https://docs.stripe.com/"),
]

SOURCES.extend(
    [
        ("fault", "PayPal payment declined", "https://www.paypal.com/us/cshelp/article/why-was-my-payment-declined-help419"),
        ("fault", "PayPal login troubleshooting", "https://www.paypal.com/us/cshelp/article/why-cant-i-log-in-to-my-paypal-account-help276"),
        ("fault", "Stripe declines documentation", "https://docs.stripe.com/declines"),
        ("fault", "Stripe API error codes", "https://docs.stripe.com/error-codes"),
        ("fault", "Square reader troubleshooting", "https://squareup.com/help/us/en/article/5068-troubleshoot-your-square-reader"),
        ("fault", "Wix payment failure troubleshooting", "https://support.wix.com/en/article/wix-stores-troubleshooting-failed-payments"),
        ("fault", "BigCommerce checkout errors", "https://support.bigcommerce.com/s/article/Troubleshooting-Checkout-Errors"),
        ("fault", "Shopify payment authorization troubleshooting", "https://help.shopify.com/en/manual/payments/payment-authorization"),
        ("fault", "WooCommerce failed pending orders", "https://woocommerce.com/document/managing-orders/order-statuses/"),
        ("fault", "ShipStation troubleshooting", "https://help.shipstation.com/hc/en-us/categories/360001875732-Troubleshooting"),
        ("order", "BigCommerce orders", "https://support.bigcommerce.com/s/article/Orders"),
        ("order", "Wix manage orders", "https://support.wix.com/en/article/wix-stores-viewing-and-managing-your-orders"),
        ("order", "Square manage orders", "https://squareup.com/help/us/en/article/5061-manage-orders-with-square"),
        ("order", "ShipStation order grid", "https://help.shipstation.com/hc/en-us/articles/360026157731-Order-Grid"),
        ("order", "Adobe Commerce orders", "https://experienceleague.adobe.com/en/docs/commerce-admin/stores-sales/order-management/orders/orders"),
        ("order", "Shopify order status overview", "https://help.shopify.com/en/manual/orders/status-tracking/order-status-overview"),
        ("order", "PayPal transaction history", "https://www.paypal.com/us/cshelp/article/how-do-i-view-my-paypal-transaction-history-help107"),
        ("order", "Stripe payment details", "https://docs.stripe.com/payments/payment-intents/verifying-status"),
        ("order", "WooCommerce order status manager", "https://woocommerce.com/document/woocommerce-order-status-manager/"),
        ("order", "BigCommerce managing orders", "https://support.bigcommerce.com/s/article/Processing-Orders"),
        ("refund", "PayPal issue a refund", "https://www.paypal.com/us/cshelp/article/how-do-i-issue-a-refund-help101"),
        ("refund", "Square refund payments", "https://squareup.com/help/us/en/article/5060-refund-payments"),
        ("refund", "BigCommerce refunding orders", "https://support.bigcommerce.com/s/article/Refunding-Orders"),
        ("refund", "Wix refunding orders", "https://support.wix.com/en/article/wix-stores-refunding-orders"),
        ("refund", "Adobe Commerce credit memos", "https://experienceleague.adobe.com/en/docs/commerce-admin/stores-sales/order-management/credit-memos/credit-memos"),
        ("refund", "Shopify returns management", "https://help.shopify.com/en/manual/orders/refund-cancel-order"),
        ("refund", "Stripe dashboard refunds", "https://docs.stripe.com/refunds"),
        ("refund", "PayPal refund status", "https://www.paypal.com/us/cshelp/article/where-is-my-refund-help123"),
        ("refund", "WooCommerce returns warranty", "https://woocommerce.com/document/woocommerce-rma/"),
        ("refund", "Square partial refunds", "https://squareup.com/help/us/en/article/6116-process-refunds"),
        ("howto", "BigCommerce add products", "https://support.bigcommerce.com/s/article/Adding-Products"),
        ("howto", "Wix add store products", "https://support.wix.com/en/article/wix-stores-adding-a-physical-product"),
        ("howto", "Square online setup", "https://squareup.com/help/us/en/article/6868-get-started-with-square-online"),
        ("howto", "ShipStation create labels", "https://help.shipstation.com/hc/en-us/articles/360025856252-Create-Labels"),
        ("howto", "Adobe Commerce create products", "https://experienceleague.adobe.com/en/docs/commerce-admin/catalog/products/products-list"),
        ("howto", "Shopify shipping setup", "https://help.shopify.com/en/manual/fulfillment/setup"),
        ("howto", "Stripe checkout quickstart", "https://docs.stripe.com/checkout/quickstart"),
        ("howto", "PayPal create invoice", "https://www.paypal.com/us/cshelp/article/how-do-i-create-and-send-an-invoice-help319"),
        ("howto", "WooCommerce setup wizard", "https://woocommerce.com/document/woocommerce-setup-wizard/"),
        ("howto", "BigCommerce shipping setup", "https://support.bigcommerce.com/s/article/Shipping"),
        ("normal", "PayPal help center", "https://www.paypal.com/us/cshelp/personal"),
        ("normal", "Square support center", "https://squareup.com/help/us/en"),
        ("normal", "BigCommerce support", "https://support.bigcommerce.com/"),
        ("normal", "Wix support", "https://support.wix.com/en/"),
        ("normal", "ShipStation help center", "https://help.shipstation.com/hc/en-us"),
        ("normal", "Adobe Commerce documentation", "https://experienceleague.adobe.com/en/docs/commerce-admin/start/guide-overview"),
        ("normal", "PayPal business", "https://www.paypal.com/us/business"),
        ("normal", "Square commerce", "https://squareup.com/us/en/commerce"),
        ("normal", "BigCommerce homepage", "https://www.bigcommerce.com/"),
        ("normal", "Wix eCommerce", "https://www.wix.com/ecommerce/website"),
    ]
)

SOURCES.extend(
    [
        ("fault", "WooCommerce JavaScript errors", "https://woocommerce.com/document/troubleshoot-javascript-errors/"),
        (
            "fault",
            "WooCommerce mobile image upload troubleshooting",
            "https://woocommerce.com/document/troubleshooting-image-upload-issues-in-the-woo-mobile-apps/",
        ),
        ("fault", "WooCommerce subscription scheduled action errors", "https://woocommerce.com/document/subscriptions/scheduled-action-errors/"),
        ("fault", "WooCommerce Square troubleshooting", "https://woocommerce.com/document/woocommerce-square/troubleshooting"),
        ("fault", "WooCommerce Viva checkout troubleshooting", "https://woocommerce.com/document/viva-wallet-standard-checkout/"),
        (
            "fault",
            "Adobe Commerce cron troubleshooting",
            "https://experienceleague.adobe.com/en/docs/experience-cloud-kcs/kbarticles/ka-29907",
        ),
        ("order", "WooCommerce Android orders", "https://woocommerce.com/document/android-orders/"),
        (
            "order",
            "WooCommerce single order page",
            "https://woocommerce.com/document/managing-orders/view-edit-or-add-an-order/",
        ),
        ("order", "WooCommerce order barcodes", "https://woocommerce.com/document/woocommerce-order-barcodes/"),
        (
            "order",
            "Adobe Commerce order processing",
            "https://experienceleague.adobe.com/en/docs/commerce-admin/stores-sales/order-management/orders/order-processing",
        ),
        (
            "order",
            "Adobe Commerce order status",
            "https://experienceleague.adobe.com/en/docs/commerce-admin/stores-sales/order-management/orders/order-status",
        ),
        (
            "order",
            "Adobe Commerce create order",
            "https://experienceleague.adobe.com/en/docs/commerce-admin/stores-sales/order-management/orders/order-create",
        ),
        ("refund", "WooCommerce self-service refunds", "https://woocommerce.com/document/self-service-refunds/"),
        ("refund", "WooCommerce account funds refunds", "https://woocommerce.com/document/account-funds/"),
        ("refund", "WooCommerce cost of goods refunded orders", "https://woocommerce.com/document/cost-of-goods-sold"),
        ("refund", "WooCommerce returns workflow", "https://woocommerce.com/document/returns-for-woocommerce/"),
        ("refund", "WooCommerce Klarna refunds", "https://woocommerce.com/document/klarna-order-management/"),
        (
            "refund",
            "WooCommerce PayPal orders and refunds",
            "https://woocommerce.com/document/woocommerce-paypal-payments/managing-orders-and-refunds/",
        ),
        (
            "refund",
            "Adobe Commerce create credit memo",
            "https://experienceleague.adobe.com/en/docs/commerce-admin/stores-sales/order-management/credit-memos/credit-memo-create",
        ),
        ("howto", "WooCommerce product CSV importer", "https://woocommerce.com/document/product-csv-importer-exporter/"),
        ("howto", "WooCommerce settings guide", "https://woocommerce.com/document/configuring-woocommerce-settings/"),
        ("howto", "WooCommerce products guide", "https://woocommerce.com/document/adding-and-managing-products/"),
        (
            "howto",
            "Adobe Commerce products list",
            "https://experienceleague.adobe.com/en/docs/commerce-admin/catalog/products/products-list",
        ),
        (
            "howto",
            "Adobe Commerce create admin user",
            "https://experienceleague.adobe.com/en/docs/commerce-admin/start/admin/admin-user-create",
        ),
        ("normal", "WooCommerce plugin documentation", "https://woocommerce.com/documentation/plugins/woocommerce/"),
        ("normal", "WooCommerce developer portal", "https://developer.woocommerce.com/"),
        ("normal", "WooCommerce analytics", "https://woocommerce.com/document/woocommerce-analytics/"),
        ("normal", "WooCommerce mobile app", "https://woocommerce.com/document/woocommerce-mobile-app/"),
    ]
)


def main() -> None:
    npx = shutil.which("npx.cmd") or shutil.which("npx")
    if not npx:
        raise SystemExit("npx is required to run Playwright screenshots.")
    jobs = []
    for source_index, (label, title, url) in enumerate(SOURCES, start=1):
        for variant_name, viewport in VARIANTS:
            jobs.append((npx, source_index, label, title, url, variant_name, viewport))

    manifest = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(capture_one, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            manifest.append(row)
            print(f"[{row['status']}] {row['label']} {row['variant']} {row['title']}: {row['url']}", flush=True)

    manifest.sort(key=lambda row: (row["label"], row["title"], row["variant"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for row in manifest if row["status"] in {"ok", "existing"})
    print(json.dumps({"manifest": str(MANIFEST), "ok": ok, "total": len(manifest)}, ensure_ascii=False, indent=2))


def capture_one(job: tuple[str, int, str, str, str, str, str]) -> dict:
    npx, source_index, label, title, url, variant_name, viewport = job
    target_dir = OUT_DIR / label
    target_dir.mkdir(parents=True, exist_ok=True)
    output = target_dir / f"public_web_{source_index:03d}_{variant_name}_{slug(title)}.png"
    if output.exists():
        return {
            "label": label,
            "title": title,
            "variant": variant_name,
            "viewport": viewport,
            "url": url,
            "file": str(output),
            "status": "existing",
            "stdout": "",
            "stderr": "",
            "usage": "public_customer_service_webpage_screenshot",
            "strict_note": (
                "Public web screenshot captured from the listed source page. Multiple viewport captures "
                "increase visual coverage but should not be described as independent real customer cases."
            ),
        }
    command = [
        npx,
        "--yes",
        "playwright",
        "screenshot",
        "--browser",
        "chromium",
        "--viewport-size",
        viewport,
        "--wait-for-timeout",
        "1200",
        "--timeout",
        "30000",
        "--ignore-https-errors",
        "--user-agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        url,
        str(output),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True)
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    return {
        "label": label,
        "title": title,
        "variant": variant_name,
        "viewport": viewport,
        "url": url,
        "file": str(output),
        "status": "ok" if result.returncode == 0 and output.exists() else "failed",
        "stdout": stdout.strip()[-1000:],
        "stderr": stderr.strip()[-1000:],
        "usage": "public_customer_service_webpage_screenshot",
        "strict_note": (
            "Public web screenshot captured from the listed source page. Multiple viewport captures "
            "increase visual coverage but should not be described as independent real customer cases."
        ),
    }


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return value[:80]


if __name__ == "__main__":
    main()
