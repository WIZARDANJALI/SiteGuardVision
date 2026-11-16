"""
SiteGuardVision - inference engine.
Loads the PPE YOLO model once and exposes a single function to analyse one image.
"""

from __future__ import annotations
import uuid
from pathlib import Path
from typing import Any
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "yolo_ppe_core.pt"
RENDERED_DIR = ROOT_DIR / "static" / "rendered"
RENDERED_DIR.mkdir(parents=True, exist_ok=True)

_model: YOLO | None = None

def get_model() -> YOLO:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
    return _model

def _normalise(label: str) -> str:
    return label.strip().lower()

def analyze_image(image_path: str | Path, conf: float = 0.25, imgsz: int = 960) -> dict[str, Any]:
    """
    Run PPE inference on a single image with custom logic for vests vs helmets.
    """
    result: dict[str, Any] = {
        "success": False, "annotated_rel": None, "annotated_abs": None,
        "counts": {}, "person_count": 0, "hardhat_count": 0,
        "no_hardhat_count": 0, "vest_count": 0, "no_vest_count": 0,
        "potential_no_hardhat": [], "potential_no_vest": [],
        "inference_ms": 0.0, "image_size": [0, 0], "error": None,
    }

    image_path = Path(image_path)
    if not image_path.exists():
        result["error"] = f"Image not found: {image_path}"
        return result

    try:
        model = get_model()

        # 1. Run inference at a very low threshold (0.15) to catch everything.
        # We will filter it ourselves below.
        predictions = model.predict(
            source=str(image_path),
            conf=0.15, 
            imgsz=int(imgsz),
            device="cpu",
            save=False,
            verbose=False,
        )

        if not predictions:
            result["error"] = "Model returned no prediction objects."
            return result

        pred = predictions[0]
        result["inference_ms"] = float(pred.speed.get("inference", 0.0)) if hasattr(pred, "speed") else 0.0

        # 2. Custom Filtering Logic
        names = pred.names if hasattr(pred, "names") else {}
        counts: dict[str, int] = {}
        potential_no_hardhat: list[dict[str, Any]] = []
        potential_no_vest: list[dict[str, Any]] = []

        # We define custom thresholds for different items
        HELMET_CONF = conf # The user's slider controls helmets
        VEST_CONF = 0.10  # Vests need to be 55% confident to count (stops random guesses)

        boxes = getattr(pred, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.tolist()
            confs = boxes.conf.tolist()
            clss = boxes.cls.tolist()

            for bbox, c, cls_id in zip(xyxy, confs, clss):
                label = names.get(int(cls_id), str(int(cls_id)))
                norm_label = _normalise(label)
                
                                # Filter based on class
                if "no-safety vest" in norm_label or "no-safety-vest" in norm_label:
                    # Only trust a MISSING vest if the AI is 50% sure. 
                    # (This stops it from falsely accusing people in blue shirts).
                    if c < 0.25: 
                        continue 
                elif "safety vest" in norm_label:
                    # Trust a POSITIVE vest detection at 10% (0.10) because it's hard to see.
                    if c < 0.10:
                        continue
                else:
                    # Helmets use the slider threshold from the dashboard
                    if c < HELMET_CONF:
                        continue

                counts[label] = counts.get(label, 0) + 1

                if "no-hardhat" in norm_label or norm_label == "nohardhat":
                    potential_no_hardhat.append({"confidence": round(float(c), 4), "bbox": [round(float(v), 1) for v in bbox]})
                
                if "no-safety vest" in norm_label:
                    potential_no_vest.append({"confidence": round(float(c), 4), "bbox": [round(float(v), 1) for v in bbox]})

        result["counts"] = counts
        result["person_count"] = sum(v for k, v in counts.items() if _normalise(k) == "person")
        result["hardhat_count"] = sum(v for k, v in counts.items() if "hardhat" in _normalise(k) and "no" not in _normalise(k))
        result["no_hardhat_count"] = sum(v for k, v in counts.items() if "no-hardhat" in _normalise(k) or "nohardhat" in _normalise(k))
        result["vest_count"] = sum(v for k, v in counts.items() if "safety vest" in _normalise(k) and "no" not in _normalise(k))
        result["no_vest_count"] = sum(v for k, v in counts.items() if "no-safety vest" in _normalise(k))
        result["potential_no_hardhat"] = potential_no_hardhat
        result["potential_no_vest"] = potential_no_vest
        result["success"] = True

        # 3. Save Annotated Image
        annotated_name = f"det_{uuid.uuid4().hex[:10]}.jpg"
        annotated_abs = RENDERED_DIR / annotated_name
        import cv2
        cv2.imwrite(str(annotated_abs), pred.plot())
        
        result["annotated_abs"] = str(annotated_abs)
        result["annotated_rel"] = f"rendered/{annotated_name}"
        if hasattr(pred, "orig_shape") and pred.orig_shape:
            result["image_size"] = [int(pred.orig_shape[1]), int(pred.orig_shape[0])]

        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result