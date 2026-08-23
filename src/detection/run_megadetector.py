from pathlib import Path
import json

import cv2
import pandas as pd
import torch
from tqdm import tqdm

from PytorchWildlife.models import detection as pw_detection


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = Path("data/raw")
METADATA_FILE = Path("data/metadata/subset_metadata.csv")

DETECTION_DIR = Path("data/interim")
CROP_DIR = Path("data/cropped")

DETECTION_OUTPUT = (
    DETECTION_DIR / "megadetector_results.json"
)

CROP_METADATA_OUTPUT = (
    DETECTION_DIR / "crop_metadata.csv"
)

CONFIDENCE_THRESHOLD = 0.20

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SETUP
# ============================================================

DETECTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CROP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD LABEL METADATA
# ============================================================

metadata_df = pd.read_csv(
    METADATA_FILE
)

label_map = dict(
    zip(
        metadata_df["file_name"],
        metadata_df["species"]
    )
)


# ============================================================
# LOAD MEGADETECTOR
# ============================================================

print("=" * 60)
print("LOADING MEGADETECTOR V6")
print("=" * 60)

print(f"Device: {DEVICE}")

model = pw_detection.MegaDetectorV6(
    version="MDV6-yolov10-c",
    device=DEVICE,
    pretrained=True
)

print("MegaDetector loaded.")


# ============================================================
# GET IMAGE LIST
# ============================================================

image_paths = sorted(
    IMAGE_DIR.glob("*.jpg")
)

print(
    f"\nImages found: {len(image_paths)}"
)


# ============================================================
# PROCESS IMAGES
# ============================================================

all_detection_results = []

crop_rows = []

images_with_animals = 0
images_without_animals = 0

total_crops = 0


for image_path in tqdm(
    image_paths,
    desc="Running MegaDetector",
    unit="img"
):

    try:

        result = model.single_image_detection(
            str(image_path)
        )

        detections = result["detections"]

        # ----------------------------------------------------
        # Store raw detector information
        # ----------------------------------------------------

        image_result = {
            "file_name": image_path.name,
            "detections": []
        }

        # ----------------------------------------------------
        # Load original image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        height, width = image.shape[:2]

        # ----------------------------------------------------
        # Extract detections
        # ----------------------------------------------------

        boxes = detections.xyxy
        confidences = detections.confidence
        class_ids = detections.class_id

        valid_detection_count = 0

        for det_idx, (
            box,
            confidence,
            class_id
        ) in enumerate(
            zip(
                boxes,
                confidences,
                class_ids
            )
        ):

            confidence = float(
                confidence
            )

            class_id = int(
                class_id
            )

            # ------------------------------------------------
            # Confidence filter
            # ------------------------------------------------

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # MegaDetector class 0 = animal
            if class_id != 0:
                continue

            x1, y1, x2, y2 = map(
                int,
                box
            )

            # ------------------------------------------------
            # Clamp coordinates
            # ------------------------------------------------

            x1 = max(
                0,
                min(x1, width - 1)
            )

            y1 = max(
                0,
                min(y1, height - 1)
            )

            x2 = max(
                0,
                min(x2, width)
            )

            y2 = max(
                0,
                min(y2, height)
            )

            # Ignore invalid bounding boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # ------------------------------------------------
            # Add small padding
            # ------------------------------------------------

            box_width = x2 - x1
            box_height = y2 - y1

            padding_x = int(
                box_width * 0.10
            )

            padding_y = int(
                box_height * 0.10
            )

            crop_x1 = max(
                0,
                x1 - padding_x
            )

            crop_y1 = max(
                0,
                y1 - padding_y
            )

            crop_x2 = min(
                width,
                x2 + padding_x
            )

            crop_y2 = min(
                height,
                y2 + padding_y
            )

            # ------------------------------------------------
            # Crop
            # ------------------------------------------------

            crop = image[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]

            if crop.size == 0:
                continue

            # ------------------------------------------------
            # Species comes from Caltech metadata
            # ------------------------------------------------

            species = label_map.get(
                image_path.name,
                "unknown"
            )

            if species == "unknown":
                continue

            # ------------------------------------------------
            # Create species directory
            # ------------------------------------------------

            species_dir = (
                CROP_DIR / species
            )

            species_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # ------------------------------------------------
            # Save crop
            # ------------------------------------------------

            crop_name = (
                f"{image_path.stem}"
                f"_crop{det_idx}.jpg"
            )

            crop_path = (
                species_dir / crop_name
            )

            cv2.imwrite(
                str(crop_path),
                crop
            )

            # ------------------------------------------------
            # Store detection metadata
            # ------------------------------------------------

            detection_info = {
                "confidence": confidence,
                "class_id": class_id,
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2
                ],
                "crop_path": str(
                    crop_path
                )
            }

            image_result[
                "detections"
            ].append(
                detection_info
            )

            crop_rows.append({
                "original_file": image_path.name,
                "crop_file": str(crop_path),
                "species": species,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })

            valid_detection_count += 1
            total_crops += 1

        # ----------------------------------------------------
        # Image-level statistics
        # ----------------------------------------------------

        if valid_detection_count > 0:

            images_with_animals += 1

        else:

            images_without_animals += 1

        all_detection_results.append(
            image_result
        )

    except Exception as error:

        print(
            f"\nError processing "
            f"{image_path.name}: {error}"
        )


# ============================================================
# SAVE DETECTION RESULTS
# ============================================================

with open(
    DETECTION_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_detection_results,
        f,
        indent=2
    )


# ============================================================
# SAVE CROP METADATA
# ============================================================

crop_df = pd.DataFrame(
    crop_rows
)

crop_df.to_csv(
    CROP_METADATA_OUTPUT,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MEGADETECTOR PIPELINE COMPLETE")
print("=" * 60)

print(
    f"Images processed       : {len(image_paths)}"
)

print(
    f"Images with animals    : {images_with_animals}"
)

print(
    f"Images without animals : {images_without_animals}"
)

print(
    f"Total animal crops     : {total_crops}"
)

print(
    f"\nDetection results:"
    f"\n{DETECTION_OUTPUT}"
)

print(
    f"\nCrop metadata:"
    f"\n{CROP_METADATA_OUTPUT}"
)

print(
    f"\nCrops saved in:"
    f"\n{CROP_DIR}"
)