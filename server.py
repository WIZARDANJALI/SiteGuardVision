"""
SiteGuardVision - Flask server.

Routes:
  GET  /                      -> home page with upload form
  POST /analyze               -> runs inference, renders analysis page
  GET  /health                -> tiny health check for debugging
"""

from __future__ import annotations

import uuid
from pathlib import Path

from flask import (
    Flask,
    abort,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

from infer_engine import analyze_image

ROOT_DIR = Path(__file__).resolve().parent
INCOMING_DIR = ROOT_DIR / "static" / "incoming"
SAMPLES_DIR = ROOT_DIR / "samples"
INCOMING_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024


def _is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _list_samples() -> list[str]:
    if not SAMPLES_DIR.exists():
        return []
    return sorted([p.name for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in ALLOWED_EXTENSIONS])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def home():
    return render_template("home.html", samples=_list_samples(), error=None)


@app.post("/analyze")
def analyze():
    uploaded = request.files.get("image")
    sample_name = request.form.get("sample")
    image_path: Path | None = None

    try:
        if uploaded and uploaded.filename:
            if not _is_allowed(uploaded.filename):
                return render_template(
                    "home.html",
                    samples=_list_samples(),
                    error="Unsupported file type. Please upload JPG, PNG, WEBP, or BMP.",
                )
            safe_name = secure_filename(uploaded.filename)
            stored_name = f"{uuid.uuid4().hex[:10]}_{safe_name}"
            image_path = INCOMING_DIR / stored_name
            uploaded.save(image_path)

        elif sample_name:
            safe_sample = secure_filename(sample_name)
            candidate = SAMPLES_DIR / safe_sample
            if not candidate.exists():
                abort(404, description="Sample image not found.")
            image_path = candidate

        else:
            return render_template(
                "home.html",
                samples=_list_samples(),
                error="Please choose a sample image or upload a file.",
            )

        conf = float(request.form.get("conf", 0.25))
        imgsz = int(request.form.get("imgsz", 960))

        result = analyze_image(image_path, conf=conf, imgsz=imgsz)

        return render_template(
            "analysis.html",
            result=result,
            source_name=image_path.name,
            conf=conf,
            imgsz=imgsz,
        )

    except Exception as exc:  # noqa: BLE001
        return render_template(
            "home.html",
            samples=_list_samples(),
            error=f"Processing failed: {type(exc).__name__}: {exc}",
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)