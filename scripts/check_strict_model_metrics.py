from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "strict_model_audit.json"

MODEL_REPORTS = {
    "text": ROOT / "reports" / "text_bert_metrics.json",
    "audio": ROOT / "reports" / "audio_lstm_metrics.json",
    "emotion": ROOT / "reports" / "emotion_cnn_metrics.json",
    "coco_image": ROOT / "reports" / "image_cnn_metrics.json",
    "screenshot_intent": ROOT / "reports" / "customer_screenshot_intent_metrics.json",
    "fusion": ROOT / "reports" / "fusion_model_metrics.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit trained model reports against the strict screenshot route.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any route-critical model is missing or not credible.")
    parser.add_argument("--target_accuracy", type=float, default=0.82)
    args = parser.parse_args()

    reports = {name: read_json(path) for name, path in MODEL_REPORTS.items()}
    audit = build_audit(reports, args.target_accuracy)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print_human_report(audit)
    if args.strict and not audit["strict_model_ready"]:
        raise SystemExit(1)


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_audit(reports: dict[str, dict | None], target_accuracy: float) -> dict:
    modules = {
        "text": audit_text(reports["text"], target_accuracy),
        "audio": audit_audio(reports["audio"], target_accuracy),
        "emotion": audit_emotion(reports["emotion"], target_accuracy),
        "coco_image": audit_coco_image(reports["coco_image"], target_accuracy),
        "screenshot_intent": audit_screenshot_intent(reports["screenshot_intent"], target_accuracy),
        "fusion": audit_fusion(reports["fusion"], target_accuracy),
    }
    return {
        "target_accuracy": target_accuracy,
        "strict_model_ready": all(module["ready"] for module in modules.values()),
        "modules": modules,
        "report_path": str(REPORT_PATH),
    }


def audit_text(report: dict | None, target_accuracy: float) -> dict:
    if not report:
        return missing("BERT text report is missing.")
    ready = (
        report.get("model") == "bert-base-chinese"
        and report.get("strict_sources") is True
        and float(report.get("accuracy", 0.0)) >= target_accuracy
    )
    note = "Requires bert-base-chinese trained on strict SQuAD + ABCD data."
    return module_result(ready, report, note)


def audit_audio(report: dict | None, target_accuracy: float) -> dict:
    if not report:
        return missing("LibriSpeech LSTM report is missing.")
    ready = "LSTM" in str(report.get("model", "")) and float(report.get("accuracy", 0.0)) >= target_accuracy
    note = "LibriSpeech has no customer-service intent labels; this can only validate the requested LSTM speech branch."
    return module_result(ready, report, note)


def audit_emotion(report: dict | None, target_accuracy: float) -> dict:
    if not report:
        return missing("FER-2013 CNN report is missing.")
    task = str(report.get("task", "")).lower()
    report_by_label = report.get("classification_report", {})
    threshold_report = report.get("threshold_tuning")
    if report.get("binary_angry"):
        if threshold_report:
            accuracy = float(threshold_report.get("accuracy", 0.0))
            angry_recall = float(threshold_report.get("angry_recall", 0.0))
            ready = accuracy >= target_accuracy and angry_recall >= 0.5
            note = "Binary angry-vs-other CNN uses a tuned decision threshold saved in the checkpoint."
            result = module_result(ready, report, note)
            result["accuracy"] = threshold_report.get("accuracy")
            result["threshold"] = threshold_report.get("threshold")
            result["angry_recall"] = threshold_report.get("angry_recall")
            return result
        angry_recall = float(report_by_label.get("1", {}).get("recall", 0.0))
        ready = float(report.get("accuracy", 0.0)) >= target_accuracy and angry_recall >= 0.5
        note = "Binary angry-vs-other accuracy alone is not enough; angry recall must be non-trivial."
    else:
        ready = "fer-2013" in task and float(report.get("accuracy", 0.0)) >= target_accuracy
        note = "Seven-class FER-2013 CNN is preferred for the screenshot route."
    return module_result(ready, report, note)


def audit_coco_image(report: dict | None, target_accuracy: float) -> dict:
    if not report:
        return missing("COCO CNN report is missing.")
    ready = "coco" in str(report.get("task", "")).lower() and float(report.get("accuracy", 0.0)) >= target_accuracy
    note = "COCO CNN is a required image branch, but it does not replace customer screenshot intent recognition."
    return module_result(ready, report, note)


def audit_screenshot_intent(report: dict | None, target_accuracy: float) -> dict:
    if not report:
        return missing("Five-class customer screenshot intent report is missing.")
    accuracy = float(report.get("best_accuracy", report.get("accuracy", 0.0)) or 0.0)
    ready = (
        report.get("strict_sources") is True
        and "five-class" in str(report.get("task", ""))
        and accuracy >= target_accuracy
    )
    note = "Uses grouped evaluation by public page/source id; this is the honest metric for screenshot intent generalization."
    result = module_result(ready, report, note)
    result["accuracy"] = report.get("best_accuracy", report.get("accuracy"))
    result["final_accuracy"] = report.get("accuracy")
    result["best_epoch"] = report.get("best_epoch")
    return result


def audit_fusion(report: dict | None, target_accuracy: float) -> dict:
    if not report:
        return missing("Fusion model report is missing.")
    ready = report.get("strict_sources") is True and int(report.get("samples", 0)) >= 50 and float(report.get("accuracy", 0.0)) >= target_accuracy
    note = "Strict fusion must be trained on real multimodal rows, not demo samples."
    return module_result(ready, report, note)


def missing(note: str) -> dict:
    return {"ready": False, "accuracy": None, "note": note, "report": None}


def module_result(ready: bool, report: dict, note: str) -> dict:
    samples = report.get("samples")
    if samples is None and "train_size" in report and "test_size" in report:
        samples = int(report.get("train_size", 0)) + int(report.get("test_size", 0))
    return {
        "ready": ready,
        "accuracy": report.get("accuracy"),
        "samples": samples,
        "task": report.get("task"),
        "strict_sources": report.get("strict_sources"),
        "note": note,
        "report": report,
    }


def print_human_report(audit: dict) -> None:
    print("Strict model audit")
    print("=" * 20)
    for module, info in audit["modules"].items():
        status = "OK" if info["ready"] else "NOT_READY"
        accuracy = info["accuracy"]
        accuracy_text = "n/a" if accuracy is None else f"{float(accuracy):.4f}"
        print(f"[{status}] {module}: accuracy={accuracy_text} samples={info.get('samples')}")
    print()
    print(f"strict_model_ready={audit['strict_model_ready']}")
    print(f"json_report={audit['report_path']}")


if __name__ == "__main__":
    main()
