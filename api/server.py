"""FastAPI server for RareCam.
Wraps existing MegaDetector and ResNet-50 models for inference.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PytorchWildlife.models import detection as pw_detection
from src.explainability.daac import analyze_detection

# Constants matching demo/app.py exactly
CLASS_NAMES = [
    "opossum",
    "coyote",
    "deer",
    "raccoon",
    "bobcat",
    "mountain_lion",
]
MODEL_PATH = ROOT_DIR / "models" / "baseline_resnet50.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REVIEW_THRESHOLD = 0.60
RARITY_WEIGHTS = [0.049, 0.049, 0.067, 0.072, 0.101, 5.661]

app = FastAPI(
    title="RareCam API",
    description="Wildlife camera-trap detection, classification, and explainability API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
detector: Optional[pw_detection.MegaDetectorV6] = None
classifier: Optional[nn.Module] = None
models_loaded: bool = False
device_name: str = DEVICE


def load_all_models() -> None:
    global detector, classifier, models_loaded, device_name
    print(f"[RareCam API] Loading models on device: {DEVICE}...")

    # MegaDetector V6
    detector = pw_detection.MegaDetectorV6(
        version="MDV6-yolov10-c",
        device=DEVICE,
        pretrained=True,
    )

    # ResNet-50 Classifier
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    clf = models.resnet50(weights=None)
    clf.fc = nn.Linear(clf.fc.in_features, len(CLASS_NAMES))
    clf.load_state_dict(checkpoint["model_state_dict"])
    clf = clf.to(DEVICE)
    clf.eval()
    classifier = clf

    models_loaded = True
    print("[RareCam API] Models successfully loaded and ready.")


@app.on_event("startup")
def startup_event():
    try:
        load_all_models()
    except Exception as exc:
        print(f"[RareCam API] Warning: Error during model initialization: {exc}")


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok" if models_loaded else "initializing",
        "models_loaded": models_loaded,
        "detector_loaded": detector is not None,
        "classifier_loaded": classifier is not None,
        "device": DEVICE,
        "review_threshold": REVIEW_THRESHOLD,
        "classes": CLASS_NAMES,
    }


def preprocess_image(image: Image.Image) -> torch.Tensor:
    resized = image.resize((224, 224))
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0)
    return tensor.float()


def image_to_data_url(image: Image.Image, max_dim: Optional[int] = None, quality: int = 85) -> str:
    img = image.copy()
    if max_dim is not None and (img.width > max_dim or img.height > max_dim):
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not models_loaded or detector is None or classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded or initializing.")

    filename = file.filename or "unknown.jpg"
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(exc)}")

    # Downscaled full frame JPEG data URL (max 1280px)
    frame_image_data_url = image_to_data_url(pil_image, max_dim=1280, quality=80)
    img_width, img_height = pil_image.size

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_path = temp_file.name
        pil_image.save(temp_path, format="JPEG", quality=95)

    try:
        # MegaDetector inference
        det_result = detector.single_image_detection(temp_path)
        detections = det_result["detections"]
        boxes = detections.xyxy
        confidences = detections.confidence
        class_ids = detections.class_id

        # Category 0 = Animal, confidence >= 0.20
        animal_indices = [
            i
            for i, class_id in enumerate(class_ids)
            if int(class_id) == 0 and float(confidences[i]) >= 0.20
        ]

        if not animal_indices:
            return {
                "filename": filename,
                "no_animal": True,
                "detections": [],
            }

        image_array = np.array(pil_image)
        weight_tensor = torch.tensor(RARITY_WEIGHTS, dtype=torch.float32, device=DEVICE)

        results_list: List[Dict[str, Any]] = []

        for detection_number, index in enumerate(animal_indices, start=1):
            x1, y1, x2, y2 = map(int, boxes[index])
            det_conf = float(confidences[index])

            # Clamping box
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))

            crop = image_array[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_image = Image.fromarray(crop)
            crop_data_url = image_to_data_url(crop_image, max_dim=600, quality=90)

            # Normalized bounding box [x, y, w, h]
            bbox_norm = [
                round(x1 / img_width, 4),
                round(y1 / img_height, 4),
                round((x2 - x1) / img_width, 4),
                round((y2 - y1) / img_height, 4),
            ]

            # ResNet-50 Classifier inference
            tensor = preprocess_image(crop_image).to(DEVICE)
            with torch.no_grad():
                logits = classifier(tensor)
                probabilities = F.softmax(logits, dim=1)[0]

            predicted_index = int(probabilities.argmax())
            predicted_species = CLASS_NAMES[predicted_index]
            classifier_conf = float(probabilities[predicted_index])

            # Uncertainty (predictive entropy in nats)
            entropy = float(
                -(probabilities * torch.log(probabilities + 1e-12)).sum()
            )

            # Active Learning Scores
            rarity_score = float((probabilities * weight_tensor).sum())
            priority_score = float((entropy * rarity_score) * 10)

            # DAAC & Grad-CAM overlay
            daac_score: Optional[float] = None
            gradcam_data_url: Optional[str] = None
            try:
                daac_res, cam = analyze_detection(
                    model=classifier,
                    cam=None,
                    image_path=temp_path,
                    bbox=[x1, y1, x2, y2],
                )
                daac_score = round(float(daac_res["daac_score"]), 4)

                if cam is not None and cam.size > 0:
                    # cam is resized to the 10% padded crop dimensions
                    box_width = x2 - x1
                    box_height = y2 - y1
                    pad_x = int(box_width * 0.10)
                    pad_y = int(box_height * 0.10)
                    crop_x1 = max(0, x1 - pad_x)
                    crop_y1 = max(0, y1 - pad_y)
                    crop_x2 = min(img_width, x2 + pad_x)
                    crop_y2 = min(img_height, y2 + pad_y)

                    padded_crop = image_array[crop_y1:crop_y2, crop_x1:crop_x2]
                    # Ensure cam matches padded crop shape exactly
                    cam_resized = cv2.resize(cam, (padded_crop.shape[1], padded_crop.shape[0]))
                    cam_norm = np.uint8(255 * np.clip(cam_resized, 0, 1))
                    heatmap = cv2.applyColorMap(cam_norm, cv2.COLORMAP_JET)
                    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                    overlay = cv2.addWeighted(padded_crop, 0.6, heatmap_rgb, 0.4, 0)
                    overlay_pil = Image.fromarray(overlay)
                    gradcam_data_url = image_to_data_url(overlay_pil, max_dim=600, quality=85)
            except Exception as cam_exc:
                print(f"[RareCam API] Grad-CAM / DAAC warning for detection {detection_number}: {cam_exc}")
                daac_score = None
                gradcam_data_url = None

            # Review status logic (strictly matching demo/app.py)
            needs_review = classifier_conf < REVIEW_THRESHOLD
            review_status = "NEEDS REVIEW" if needs_review else "AUTO-PROCESSED"

            results_list.append({
                "filename": filename,
                "animal_index": detection_number,
                "species": predicted_species,
                "classifier_confidence": round(classifier_conf, 4),
                "detector_confidence": round(det_conf, 4),
                "uncertainty": round(entropy, 4),
                "uncertainty_unit": "nats",
                "rarity": round(rarity_score, 4),
                "priority": round(priority_score, 4),
                "daac": daac_score,
                "review_status": review_status,
                "bbox": bbox_norm,
                "frame_image": frame_image_data_url,
                "crop_image": crop_data_url,
                "gradcam_overlay": gradcam_data_url,
            })

        return {
            "filename": filename,
            "no_animal": False,
            "detections": results_list,
        }

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="127.0.0.1", port=8000, reload=False)
