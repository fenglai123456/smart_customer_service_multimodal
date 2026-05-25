from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from customer_ai.asr_transcriber import transcribe_audio
from customer_ai.audio_analyzer import analyze_audio
from customer_ai.emotion_analyzer import analyze_emotion_image
from customer_ai.image_analyzer import analyze_image
from customer_ai.intent_engine import IntentEngine


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DEMO_DATA_DIR = BASE_DIR / "course_demo_test_data"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
engine = IntentEngine()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat(timespec="seconds")})


@app.get("/api/demo-cases")
def demo_cases():
    return jsonify({"cases": load_demo_cases()})


@app.get("/api/demo-cases/<case_id>/file/<file_kind>")
def demo_case_file(case_id: str, file_kind: str):
    case = find_demo_case(case_id)
    if not case:
        return jsonify({"error": "demo case not found", "case_id": case_id}), 404

    field_by_kind = {
        "screenshot": "screenshot_file",
        "emotion_image": "emotion_image_file",
        "audio": "audio_file",
    }
    field_name = field_by_kind.get(file_kind)
    if not field_name:
        return jsonify({"error": "unsupported demo file kind", "file_kind": file_kind}), 404

    case_dir = DEMO_DATA_DIR / case.get("folder", case["case_id"])
    filename = Path(case.get(field_name, "")).name
    if not filename or not (case_dir / filename).is_file():
        return jsonify({"error": "demo file not found", "file_kind": file_kind}), 404

    return send_from_directory(case_dir, filename)


@app.post("/api/demo-cases/<case_id>/predict")
def predict_demo_case(case_id: str):
    """Run a packaged course-demo sample without requiring manual file upload."""
    case = find_demo_case(case_id)
    if not case:
        return jsonify({"error": "demo case not found", "case_id": case_id}), 404

    case_dir = DEMO_DATA_DIR / case.get("folder", case["case_id"])
    image_path = case_dir / case["screenshot_file"]
    emotion_image_path = case_dir / case["emotion_image_file"]
    audio_path = case_dir / case["audio_file"]
    if not image_path.exists() or not emotion_image_path.exists() or not audio_path.exists():
        return jsonify({"error": "demo case files are incomplete", "case_id": case_id}), 400

    text = case.get("text", "")
    voice_text = case.get("voice_text", "")
    emotion = case.get("emotion", "neutral")
    image_signals = analyze_image(image_path, context_text=image_path.name)
    audio_signals = analyze_audio(audio_path)
    emotion_signals = analyze_emotion_image(emotion_image_path)
    prediction = engine.predict(
        text=text,
        voice_text=voice_text,
        emotion=emotion,
        image=image_signals,
        audio=audio_signals,
        emotion_image=emotion_signals,
        asr=None,
    )
    payload = {
        "ticket_id": f"CS-{datetime.now():%Y%m%d}-{uuid4().hex[:6].upper()}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": {
            "text": text,
            "voice_text": voice_text,
            "emotion": emotion,
            "image_filename": image_path.name,
            "image_stored_filename": "",
            "emotion_image_filename": emotion_image_path.name,
            "emotion_image_stored_filename": "",
            "audio_filename": audio_path.name,
            "audio_stored_filename": "",
            "auto_transcribe": False,
            "demo_case_id": case["case_id"],
        },
        "demo_case": case,
        "prediction": prediction.__dict__,
    }
    log_interaction(payload)
    return jsonify(payload)


@app.post("/api/predict")
def predict():
    """Receive multimodal form data and return the fused customer-service intent.

    This is the main Web demo endpoint used in the report. It accepts text,
    optional ASR text, an audio file, an emotion image, and a business
    screenshot, then delegates each modality to the corresponding analyzer
    before calling the central intent engine.
    """
    text = request.form.get("text", "")
    voice_text = request.form.get("voice_text", "")
    emotion = request.form.get("emotion", "neutral")
    auto_transcribe = request.form.get("auto_transcribe") == "on"
    saved_image = save_upload(
        request.files.get("image"),
        allowed_suffixes={".png", ".jpg", ".jpeg", ".webp", ".bmp"},
        fallback_suffix=".png",
    )
    saved_emotion_image = save_upload(
        request.files.get("emotion_image"),
        allowed_suffixes={".png", ".jpg", ".jpeg", ".webp", ".bmp"},
        fallback_suffix=".png",
    )
    saved_audio = save_upload(
        request.files.get("audio"),
        allowed_suffixes={".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"},
        fallback_suffix=".wav",
    )
    saved_image_path = saved_image["path"] if saved_image else None
    saved_emotion_image_path = saved_emotion_image["path"] if saved_emotion_image else None
    saved_audio_path = saved_audio["path"] if saved_audio else None

    # Each uploaded artifact is analyzed independently first; the resulting
    # structured signals are later aligned into the same five-intent label space.
    image_signals = analyze_image(
        saved_image_path,
        context_text=saved_image["original_filename"] if saved_image else "",
    )
    audio_signals = analyze_audio(saved_audio_path)
    emotion_signals = analyze_emotion_image(saved_emotion_image_path)
    asr_result = transcribe_audio(saved_audio_path) if auto_transcribe and saved_audio_path else None
    asr_payload = asr_result.__dict__ if asr_result else None
    if asr_result and asr_result.text:
        voice_text = f"{voice_text} {asr_result.text}".strip()

    # The engine performs text/voice/emotion/image score fusion and returns both
    # the business decision and the technical explanation shown on the frontend.
    prediction = engine.predict(
        text=text,
        voice_text=voice_text,
        emotion=emotion,
        image=image_signals,
        audio=audio_signals,
        emotion_image=emotion_signals,
        asr=asr_payload,
    )
    payload = {
        "ticket_id": f"CS-{datetime.now():%Y%m%d}-{uuid4().hex[:6].upper()}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": {
            "text": text,
            "voice_text": voice_text,
            "emotion": emotion,
            "image_filename": saved_image["original_filename"] if saved_image else "",
            "image_stored_filename": saved_image["stored_filename"] if saved_image else "",
            "emotion_image_filename": saved_emotion_image["original_filename"] if saved_emotion_image else "",
            "emotion_image_stored_filename": saved_emotion_image["stored_filename"] if saved_emotion_image else "",
            "audio_filename": saved_audio["original_filename"] if saved_audio else "",
            "audio_stored_filename": saved_audio["stored_filename"] if saved_audio else "",
            "auto_transcribe": auto_transcribe,
        },
        "prediction": prediction.__dict__,
    }
    log_interaction(payload)
    return jsonify(payload)


def load_demo_cases() -> list[dict]:
    if not DEMO_DATA_DIR.exists():
        return []

    cases = []
    for case_file in sorted(DEMO_DATA_DIR.glob("*/case.json")):
        case = json.loads(case_file.read_text(encoding="utf-8"))
        case_dir = case_file.parent
        cases.append(
            {
                **case,
                "folder": case_dir.name,
                "files": {
                    "screenshot": case.get("screenshot_file", ""),
                    "emotion_image": case.get("emotion_image_file", ""),
                    "audio": case.get("audio_file", ""),
                },
                "file_urls": {
                    "screenshot": f"/api/demo-cases/{case['case_id']}/file/screenshot",
                    "emotion_image": f"/api/demo-cases/{case['case_id']}/file/emotion_image",
                    "audio": f"/api/demo-cases/{case['case_id']}/file/audio",
                },
            }
        )
    return cases


def find_demo_case(case_id: str) -> dict | None:
    normalized = secure_filename(case_id)
    for case in load_demo_cases():
        if case.get("case_id") == normalized or case.get("folder") == normalized:
            return case
    return None


def save_upload(file_storage, allowed_suffixes: set[str], fallback_suffix: str):
    if not file_storage or not file_storage.filename:
        return None

    # Store uploaded files under random names so repeated classroom demos do not
    # overwrite each other, while preserving the original filename for display
    # and keyword-assisted screenshot recognition.
    original_filename = file_storage.filename
    filename = secure_filename(original_filename) or f"upload{fallback_suffix}"
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_suffixes:
        suffix = fallback_suffix

    target = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    file_storage.save(target)
    return {
        "path": target,
        "original_filename": original_filename,
        "stored_filename": target.name,
    }


def log_interaction(payload: dict) -> None:
    log_file = BASE_DIR / "data" / "interaction_log.jsonl"
    log_file.parent.mkdir(exist_ok=True)
    with log_file.open("a", encoding="utf-8") as writer:
        writer.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
