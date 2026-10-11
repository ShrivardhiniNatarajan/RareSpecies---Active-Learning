"""Configuration for context margin study."""
import os
import shutil
import zipfile
from pathlib import Path

# Base Drive Root
DRIVE_ROOT = Path("/content/gdrive/MyDrive/RareCam")
LOCAL_ROOT = Path.cwd()

# 1. Detection JSON
# Drive path confirmed: /content/gdrive/MyDrive/RareCam/megadetector/megadetector_results.json
if (DRIVE_ROOT / "megadetector" / "megadetector_results.json").exists():
    DETECTIONS_JSON = DRIVE_ROOT / "megadetector" / "megadetector_results.json"
elif (LOCAL_ROOT / "data" / "interim" / "megadetector_results.json").exists():
    DETECTIONS_JSON = LOCAL_ROOT / "data" / "interim" / "megadetector_results.json"
else:
    DETECTIONS_JSON = DRIVE_ROOT / "megadetector" / "megadetector_results.json"

# 2. Splits Directory
SPLITS_DIR = LOCAL_ROOT / "data" / "splits"
if not (SPLITS_DIR / "train.csv").exists():
    drive_splits = DRIVE_ROOT / "classification"
    if (drive_splits / "train.csv").exists():
        print(f"Restoring splits from {drive_splits} to {SPLITS_DIR}...")
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        for split_file in ["train.csv", "val.csv", "test.csv"]:
            src = drive_splits / split_file
            if src.exists():
                shutil.copy2(src, SPLITS_DIR / split_file)
    else:
        SPLITS_DIR = drive_splits

# 3. Pre-cropped Images Directory (For Phase 0 fast path)
CROPPED_DIR = LOCAL_ROOT / "data" / "cropped"
if not CROPPED_DIR.exists() or len(list(CROPPED_DIR.rglob("*.jpg"))) == 0:
    crop_zip = DRIVE_ROOT / "megadetector" / "cropped_images.zip"
    if crop_zip.exists():
        print(f"Extracting {crop_zip} to {CROPPED_DIR}...")
        CROPPED_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(crop_zip, 'r') as z:
            z.extractall(CROPPED_DIR)

# 4. Raw Source Images Directory (For Phase 1 and Phase 2 context re-cropping)
IMAGES_DIR = LOCAL_ROOT / "data" / "raw"
if not IMAGES_DIR.exists() or len(list(IMAGES_DIR.rglob("*.jpg"))) == 0:
    alt_raw = Path("/content/data/raw")
    if alt_raw.exists() and len(list(alt_raw.rglob("*.jpg"))) > 0:
        IMAGES_DIR = alt_raw
    else:
        raw_zip = DRIVE_ROOT / "dataset" / "rarecam_raw.zip"
        if raw_zip.exists():
            print(f"Extracting {raw_zip} to {IMAGES_DIR}...")
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(raw_zip, 'r') as z:
                z.extractall(IMAGES_DIR)

RESULTS_DIR = LOCAL_ROOT / "results" / "context_study"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ALL_RUNS_CSV = RESULTS_DIR / "all_runs.csv"

# Restore previous runs from Drive if available
drive_runs = DRIVE_ROOT / "results" / "context_study" / "all_runs.csv"
if not ALL_RUNS_CSV.exists() and drive_runs.exists():
    print(f"Restoring existing runs from {drive_runs}...")
    shutil.copy2(drive_runs, ALL_RUNS_CSV)

# Classes
CLASS_NAMES = ["opossum", "coyote", "deer", "raccoon", "bobcat", "mountain_lion"]

# Training hyperparams
BATCH_SIZE = 32
MAX_EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-2
IMG_SIZE = 224

# Seeds
SEEDS = [0, 1, 2, 3, 4]
