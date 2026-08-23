from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from src.training.dataset import (
    CameraTrapDataset,
    eval_transform,
    CLASS_NAMES,
)


# ============================================================
# CONFIGURATION
# ============================================================

UNLABELED_FILE = Path(
    "data/active_learning/unlabeled_pool.csv"
)

MODEL_FILE = Path(
    "models/active_learning_seed_model.pth"
)

OUTPUT_DIR = Path(
    "data/active_learning/uncertainty"
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
# LOAD UNLABELED DATA
# ============================================================

df = pd.read_csv(
    UNLABELED_FILE
)

print("=" * 60)
print("UNCERTAINTY SAMPLING - ROUND 1")
print("=" * 60)

print(
    "Unlabeled crops:",
    len(df)
)

print(
    "Unique sequences:",
    df["seq_id"].nunique()
)


# ============================================================
# LOAD DATASET
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
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

# Import torchvision model only here
from torchvision import models
import torch.nn as nn


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
# CALCULATE ENTROPY
# ============================================================

all_entropies = []

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

        # H(p) = -sum p log(p)
        entropy = -(
            probabilities *
            torch.log(
                probabilities + 1e-12
            )
        ).sum(dim=1)

        all_entropies.extend(
            entropy.cpu().numpy()
        )


df["uncertainty"] = np.asarray(
    all_entropies
)


# ============================================================
# HANDLE MULTIPLE CROPS FROM SAME SEQUENCE
# ============================================================
#
# We want ONE selected crop per sequence.
#
# For each sequence, keep its most uncertain crop.
# ============================================================

sequence_candidates = (
    df.sort_values(
        "uncertainty",
        ascending=False
    )
    .drop_duplicates(
        subset=["seq_id"]
    )
)


# ============================================================
# SELECT TOP 150 SEQUENCES
# ============================================================

if len(sequence_candidates) < BUDGET:

    raise ValueError(
        "Not enough unique sequences "
        "for the requested budget."
    )

selected = sequence_candidates.head(
    BUDGET
).copy()


# ============================================================
# REMAINING POOL
# ============================================================

selected_sequence_ids = set(
    selected["seq_id"]
)

remaining = df[
    ~df["seq_id"].isin(
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
    OUTPUT_DIR /
    "selected_round_1.csv",
    index=False
)

remaining.to_csv(
    OUTPUT_DIR /
    "remaining_round_1.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\nSelected crops:", len(selected))

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

print(
    "\nUncertainty statistics:"
)

print(
    selected["uncertainty"].describe()
)

print(
    "\nSelected species distribution:"
)

print(
    selected["species"].value_counts()
)

print(
    "\nSaved to:",
    OUTPUT_DIR
)