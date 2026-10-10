"""Configuration for context margin study."""
from pathlib import Path

# Data paths (can be overridden via command line or env vars when running on Colab)
IMAGES_DIR = Path("/content/gdrive/MyDrive/RareCam/images")
DETECTIONS_JSON = Path("/content/gdrive/MyDrive/RareCam/megadetector/detections.json")
SPLITS_DIR = Path("data/splits")
RESULTS_DIR = Path("results/context_study")
ALL_RUNS_CSV = RESULTS_DIR / "all_runs.csv"

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
