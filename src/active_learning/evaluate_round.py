from pathlib import Path
import argparse

import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from torch.utils.data import DataLoader
from torchvision import models

from src.training.dataset import (
    CameraTrapDataset,
    eval_transform,
    CLASS_NAMES,
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--strategy",
    required=True,
    choices=[
        "random",
        "uncertainty",
        "rarity_weighted",
    ],
)

args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

TEST_CSV = "data/splits/test.csv"

MODEL_PATH = (
    Path("models/active_learning")
    / f"{args.strategy}_round_1.pth"
)

RESULTS_DIR = (
    Path("results/active_learning")
    / args.strategy
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 60)
print(
    f"ACTIVE LEARNING ROUND 1 EVALUATION"
)
print("=" * 60)

print("Strategy:", args.strategy)
print("Device:", DEVICE)
print("Model:", MODEL_PATH)


# ============================================================
# LOAD TEST DATASET
# ============================================================

test_dataset = CameraTrapDataset(
    TEST_CSV,
    transform=eval_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=(DEVICE == "cuda")
)

print(
    "Test samples:",
    len(test_dataset)
)


# ============================================================
# CREATE MODEL
# ============================================================

model = models.resnet50(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        predictions = (
            outputs.argmax(dim=1)
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()


print("\n============================================")
print("CLASSIFICATION REPORT")
print("============================================")

print(
    report_df[
        [
            "precision",
            "recall",
            "f1-score",
            "support"
        ]
    ]
)


# ============================================================
# SAVE REPORT
# ============================================================

report_df.to_csv(
    RESULTS_DIR /
    "classification_report.csv"
)


# ============================================================
# SAVE PER-SPECIES RECALL
# ============================================================

recall_df = report_df.loc[
    CLASS_NAMES,
    ["recall"]
].copy()

recall_df.to_csv(
    RESULTS_DIR /
    "per_species_recall.csv"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "strategy": args.strategy,
    "test_accuracy": report["accuracy"],
    "macro_f1": report["macro avg"]["f1-score"],
    "mountain_lion_recall": report[
        "mountain_lion"
    ]["recall"],
}

summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    RESULTS_DIR /
    "summary.csv",
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

cm_df.to_csv(
    RESULTS_DIR /
    "confusion_matrix.csv"
)


# ============================================================
# FINAL
# ============================================================

print("\n============================================")
print("EVALUATION COMPLETE")
print("============================================")

print(
    "Strategy:",
    args.strategy
)

print(
    "Accuracy:",
    report["accuracy"]
)

print(
    "Macro F1:",
    report["macro avg"]["f1-score"]
)

print(
    "Mountain-lion recall:",
    report["mountain_lion"]["recall"]
)

print(
    "Results:",
    RESULTS_DIR
)