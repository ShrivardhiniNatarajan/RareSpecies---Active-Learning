from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


MODEL_PATH = Path("models/baseline_resnet50.pth")
DETECTOR_RESULTS = Path("data/interim/megadetector_results.json")

CLASS_NAMES = [
    "opossum",
    "coyote",
    "deer",
    "raccoon",
    "bobcat",
    "mountain_lion",
]

IMAGE_SIZE = 224

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model


def load_detector_results():
    with open(
        DETECTOR_RESULTS,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def calculate_iou(mask_a, mask_b):
    intersection = np.logical_and(
        mask_a,
        mask_b
    ).sum()

    union = np.logical_or(
        mask_a,
        mask_b
    ).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def analyze_detection(
    model,
    cam,
    image_path,
    bbox
):
    """
    DAAC for one detector bounding box.

    Grad-CAM is generated on the same animal crop
    used by the classifier.
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    width, height = image.size

    x1, y1, x2, y2 = [
        int(v) for v in bbox
    ]

    # Clamp bounding box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    # --------------------------------------------------------
    # Animal crop
    # --------------------------------------------------------

    crop = image.crop(
        (x1, y1, x2, y2)
    )

    input_tensor = transform(
        crop
    ).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():
        logits = model(input_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        predicted_class = int(
            probabilities.argmax()
        )

        predicted_confidence = float(
            probabilities[predicted_class]
        )

    # --------------------------------------------------------
    # Grad-CAM on the crop
    # --------------------------------------------------------

    with GradCAM(
        model=model,
        target_layers=[model.layer4[-1]]
    ) as grad_cam:

        cam = grad_cam(
            input_tensor=input_tensor,
            targets=[
                ClassifierOutputTarget(
                    predicted_class
                )
            ]
        )[0]

    # Resize CAM to crop dimensions
    crop_width = x2 - x1
    crop_height = y2 - y1

    cam = cv2.resize(
        cam,
        (crop_width, crop_height),
        interpolation=cv2.INTER_LINEAR
    )

    # --------------------------------------------------------
    # Threshold CAM
    # --------------------------------------------------------

    cam_min = cam.min()
    cam_max = cam.max()
    cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    
    attention_mask_crop = (
        cam_norm >= 0.2
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Map CAM back to original image
    # --------------------------------------------------------

    attention_mask_full = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    attention_mask_full[
        y1:y2,
        x1:x2
    ] = attention_mask_crop

    # --------------------------------------------------------
    # MegaDetector bounding-box mask
    # --------------------------------------------------------

    detector_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    detector_mask[
        y1:y2,
        x1:x2
    ] = 1

    # --------------------------------------------------------
    # DAAC
    # --------------------------------------------------------

    daac_score = calculate_iou(
        attention_mask_full,
        detector_mask
    )

    return {
        "predicted_species":
            CLASS_NAMES[predicted_class],

        "classifier_confidence":
            predicted_confidence,

        "daac_score":
            daac_score,

        "bbox":
            [x1, y1, x2, y2],
    }, cam