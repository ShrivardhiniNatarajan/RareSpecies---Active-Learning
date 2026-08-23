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

LABELED_SEED_FILE = Path(
    "data/active_learning/labeled_seed.csv"
)

UNLABELED_FILE = Path(
    "data/active_learning/unlabeled_pool.csv"
)

MODEL_FILE = Path(
    "models/active_learning_seed_model.pth"
)

OUTPUT_DIR = Path(
    "data/active_learning/rarity_weighted"
)

BUDGET = 150
BATCH_SIZE = 32

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
# LOAD LABELED SEED
# ============================================================

seed_df = pd.read_csv(
    LABELED_SEED_FILE
)

print("=" * 60)
print("RARITY-AWARE SAMPLING - ROUND 1")
print("=" * 60)

print(
    "Labeled seed:",
    len(seed_df)
)

print(
    "Unlabeled pool:",
    len(
        pd.read_csv(
            UNLABELED_FILE
        )
    )
)


# ============================================================
# ESTIMATE CLASS FREQUENCIES FROM LABELED DATA
# ============================================================
#
# IMPORTANT:
# We use ONLY the currently labeled seed.
# We do NOT use ground-truth labels from the unlabeled pool.
# ============================================================

class_counts = (
    seed_df["species"]
    .value_counts()
    .reindex(
        CLASS_NAMES,
        fill_value=0
    )
)

print("\nLabeled-seed class counts:")
print(class_counts)


# ============================================================
# CALCULATE RARITY WEIGHTS
# ============================================================
#
# Inverse frequency:
#
#     weight(c) = 1 / count(c)
#
# Then normalize so the average weight is approximately 1.
# ============================================================

raw_weights = 1.0 / (
    class_counts.astype(float) + 1e-8
)

mean_weight = raw_weights.mean()

rarity_weights = (
    raw_weights /
    mean_weight
)

print("\nRarity weights:")

for species in CLASS_NAMES:

    print(
        f"{species:15s}"
        f"{rarity_weights[species]:.4f}"
    )


# ============================================================
# LOAD UNLABELED DATA
# ============================================================

unlabeled_df = pd.read_csv(
    UNLABELED_FILE
)


# ============================================================
# CREATE DATASET
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
# LOAD SEED MODEL
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
# SCORE UNLABELED IMAGES
# ============================================================

all_uncertainties = []
all_rarity_scores = []


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

        # ----------------------------------------------------
        # 1. Predictive entropy
        # ----------------------------------------------------

        entropy = -(
            probabilities *
            torch.log(
                probabilities + 1e-12
            )
        ).sum(dim=1)

        # ----------------------------------------------------
        # 2. Expected rarity
        #
        #     sum P(class|x) * rarity_weight(class)
        #
        # This avoids relying on the single highest-probability
        # class when the model is uncertain.
        # ----------------------------------------------------

        weight_tensor = torch.tensor(
            [
                rarity_weights[
                    species
                ]
                for species in CLASS_NAMES
            ],
            dtype=probabilities.dtype,
            device=DEVICE
        )

        expected_rarity = (
            probabilities *
            weight_tensor
        ).sum(dim=1)

        all_uncertainties.extend(
            entropy.cpu().numpy()
        )

        all_rarity_scores.extend(
            expected_rarity.cpu().numpy()
        )


unlabeled_df["uncertainty"] = np.asarray(
    all_uncertainties
)

unlabeled_df["rarity_weight"] = np.asarray(
    all_rarity_scores
)


# ============================================================
# COMBINE UNCERTAINTY + RARITY
# ============================================================
#
# Normalize both components to [0, 1].
#
# priority =
# normalized uncertainty *
# normalized expected rarity
# ============================================================

def min_max_normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            np.ones(
                len(series)
            ),
            index=series.index
        )

    return (
        (series - minimum)
        /
        (maximum - minimum)
    )


unlabeled_df["uncertainty_norm"] = (
    min_max_normalize(
        unlabeled_df["uncertainty"]
    )
)

unlabeled_df["rarity_norm"] = (
    min_max_normalize(
        unlabeled_df["rarity_weight"]
    )
)

unlabeled_df["priority_score"] = (
    unlabeled_df["uncertainty_norm"]
    *
    unlabeled_df["rarity_norm"]
)


# ============================================================
# SEQUENCE CONSTRAINT
# ============================================================
#
# One selected crop per sequence.
#
# For each sequence, keep the crop with the highest
# priority score.
# ============================================================

sequence_candidates = (
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
# SELECT TOP BUDGET
# ============================================================

if len(sequence_candidates) < BUDGET:

    raise ValueError(
        "Not enough unique sequences "
        "for requested budget."
    )

selected = sequence_candidates.head(
    BUDGET
).copy()


# ============================================================
# REMOVE SELECTED SEQUENCES
# ============================================================

selected_sequence_ids = set(
    selected["seq_id"]
)

remaining = unlabeled_df[
    ~unlabeled_df["seq_id"].isin(
        selected_sequence_ids
    )
].copy()


# ============================================================
# SANITY CHECKS
# ============================================================

assert len(selected) == BUDGET

assert (
    selected["seq_id"].nunique()
    == BUDGET
)

assert (
    len(
        set(selected["seq_id"])
        &
        set(remaining["seq_id"])
    )
    == 0
)


# ============================================================
# SAVE
# ============================================================

selected.to_csv(
    OUTPUT_DIR / "selected_round_1.csv",
    index=False
)

remaining.to_csv(
    OUTPUT_DIR / "remaining_round_1.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================")
print("RARITY-AWARE SELECTION COMPLETE")
print("============================================")

print(
    "Selected crops:",
    len(selected)
)

print(
    "Selected sequences:",
    selected["seq_id"].nunique()
)

print(
    "Remaining crops:",
    len(remaining)
)

print(
    "Remaining sequences:",
    remaining["seq_id"].nunique()
)

print("\nSelected species distribution:")

print(
    selected["species"].value_counts()
)

print("\nPriority score statistics:")

print(
    selected["priority_score"].describe()
)

print(
    "\nSaved to:",
    OUTPUT_DIR
)