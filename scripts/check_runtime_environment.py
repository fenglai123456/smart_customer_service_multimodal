from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "runtime_environment.json"


def main() -> None:
    report: dict[str, object] = {"ready": True, "problems": []}
    problems: list[str] = []

    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
        if not torch.cuda.is_available():
            problems.append("PyTorch cannot see CUDA; image/audio training will run on CPU.")
    except Exception as exc:  # pragma: no cover - diagnostic script
        report["torch_error"] = str(exc)
        problems.append("PyTorch import failed.")

    try:
        import torchvision

        report["torchvision"] = {"version": torchvision.__version__}
    except Exception as exc:  # pragma: no cover - diagnostic script
        report["torchvision_error"] = str(exc)
        problems.append("torchvision import failed; ResNet-based image models cannot run.")

    try:
        import torchaudio

        report["torchaudio"] = {"version": torchaudio.__version__}
    except Exception as exc:  # pragma: no cover - diagnostic script
        report["torchaudio_error"] = str(exc)
        problems.append("torchaudio import failed; optional audio tooling may be unavailable.")

    report["ready"] = not problems
    report["problems"] = problems
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
