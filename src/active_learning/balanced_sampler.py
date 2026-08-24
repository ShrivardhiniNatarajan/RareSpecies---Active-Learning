from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

from src.training.dataset import (
    CameraTrapDataset,
    eval_transform,
    CLASS_NAMES,
)


# ============================================================
# CONFIGURATION
# ============================================================

UNLABELED_FILE = Path(
    "data/active_learning/rarity_weighted/remaining_round_2.csv"
)

LABELED_FILE = Path(
    "data/active_learning/rarity_weighted/labeled_round_2.csv"
)

MODEL_FILE = Path(
    "models/active_learning/rarity_weighted_round_2.pth"
)

OUTPUT_DIR = Path(
    "data/active_learning/balanced"
)

BUDGET = 150
BATCH_SIZE = 32

ALPHA = 0.70
BETA = 0.30

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

unlabeled_df = pd.read_csv(
    UNLABELED_FILE
)

labeled_df = pd.read_csv(
    LABELED_FILE
)

print("=" * 60)
print("BALANCED UNCERTAINTY + RARITY SAMPLING")
print("=" * 60)

print(
    "Unlabeled crops:",
    len(unlabeled_df)
)

print(
    "Unique sequences:",
    unlabeled_df["seq_id"].nunique()
)


# ============================================================
# DATASET
# ============================================================

dataset = CameraTrapDataset(
    UNLABELED_FILE,
    transform=eval_transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=(DEVICE == "cuda")
)


# ============================================================
# MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

model = models.resnet50(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# RARITY WEIGHTS FROM CURRENT LABELED SET
# ============================================================

class_counts = (
    labeled_df["species"]
    .value_counts()
    .reindex(
        CLASS_NAMES,
        fill_value=0
    )
)

raw_weights = (
    1.0 /
    (
        class_counts.astype(float)
        + 1e-8
    )
)

rarity_weights = (
    raw_weights /
    raw_weights.mean()
)


# ============================================================
# MODEL SCORING
# ============================================================

uncertainties = []
expected_rarities = []

with torch.no_grad():

    for images, _ in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(images)

        probabilities = F.softmax(
            logits,
            dim=1
        )

        entropy = -(
            probabilities *
            torch.log(
                probabilities + 1e-12
            )
        ).sum(dim=1)

        weight_tensor = torch.tensor(
            [
                rarity_weights[s]
                for s in CLASS_NAMES
            ],
            dtype=probabilities.dtype,
            device=DEVICE
        )

        expected_rarity = (
            probabilities *
            weight_tensor
        ).sum(dim=1)

        uncertainties.extend(
            entropy.cpu().numpy()
        )

        expected_rarities.extend(
            expected_rarity.cpu().numpy()
        )


unlabeled_df["uncertainty"] = np.asarray(
    uncertainties
)

unlabeled_df["rarity_weight"] = np.asarray(
    expected_rarities
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            np.ones(len(series)),
            index=series.index
        )

    return (
        (series - minimum)
        /
        (maximum - minimum)
    )


unlabeled_df["uncertainty_norm"] = normalize(
    unlabeled_df["uncertainty"]
)

unlabeled_df["rarity_norm"] = normalize(
    unlabeled_df["rarity_weight"]
)


# ============================================================
# BALANCED SCORE
# ============================================================

unlabeled_df["priority_score"] = (
    ALPHA *
    unlabeled_df["uncertainty_norm"]
    +
    BETA *
    unlabeled_df["rarity_norm"]
)


# ============================================================
# ONE CROP PER SEQUENCE
# ============================================================

candidates = (
    unlabeled_df
    .sort_values(
        "priority_score",
        ascending=False
    )
    .drop_duplicates(
        subset=["seq_id"]
    )
)


# ============================================================
# SELECT
# ============================================================

selected = candidates.head(
    BUDGET
).copy()

selected_sequences = set(
    selected["seq_id"]
)

remaining = unlabeled_df[
    ~unlabeled_df["seq_id"].isin(
        selected_sequences
    )
].copy()


# ============================================================
# SAVE
# ============================================================

selected.to_csv(
    OUTPUT_DIR /
    "selected_round_3.csv",
    index=False
)

remaining.to_csv(
    OUTPUT_DIR /
    "remaining_round_3.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================")
print("BALANCED SAMPLING COMPLETE")
print("============================================")

print(
    "Selected:",
    len(selected)
)

print(
    "Unique sequences:",
    selected["seq_id"].nunique()
)

print(
    "\nSelected species:"
)

print(
    selected["species"].value_counts()
)

print(
    "\nScore statistics:"
)

print(
    selected["priority_score"].describe()
)