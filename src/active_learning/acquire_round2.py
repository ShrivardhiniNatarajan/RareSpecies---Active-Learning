from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# CONFIGURATION
# ============================================================

BUDGET = 150
BATCH_SIZE = 32
RANDOM_SEED = 43

STRATEGY = args.strategy

BASE_DIR = Path(
    "data/active_learning"
)

REMAINING_FILE = (
    BASE_DIR /
    STRATEGY /
    "remaining_round_1.csv"
)

LABELED_FILE = (
    BASE_DIR /
    STRATEGY /
    "labeled_round_1.csv"
)

OUTPUT_DIR = (
    BASE_DIR /
    STRATEGY
)

MODEL_FILE = (
    Path("models/active_learning") /
    f"{STRATEGY}_round_1.pth"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD REMAINING POOL
# ============================================================

df = pd.read_csv(
    REMAINING_FILE
)

print("=" * 60)
print(f"ACTIVE LEARNING ROUND 2 - {STRATEGY.upper()}")
print("=" * 60)

print(
    "Remaining crops:",
    len(df)
)

print(
    "Remaining sequences:",
    df["seq_id"].nunique()
)


# ============================================================
# RANDOM STRATEGY
# ============================================================

if STRATEGY == "random":

    candidates = (
        df.sample(
            frac=1,
            random_state=RANDOM_SEED
        )
        .drop_duplicates(
            subset=["seq_id"]
        )
    )

    selected = candidates.head(
        BUDGET
    ).copy()


# ============================================================
# MODEL-BASED STRATEGIES
# ============================================================

else:

    # --------------------------------------------------------
    # Load current labeled set for rarity computation
    # --------------------------------------------------------

    labeled_df = pd.read_csv(
        LABELED_FILE
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = CameraTrapDataset(
        REMAINING_FILE,
        transform=eval_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE == "cuda")
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    uncertainties = []

    probabilities_list = []

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

            uncertainties.extend(
                entropy.cpu().numpy()
            )

            probabilities_list.append(
                probabilities.cpu().numpy()
            )

    probabilities = np.concatenate(
        probabilities_list,
        axis=0
    )

    df["uncertainty"] = np.asarray(
        uncertainties
    )

    # --------------------------------------------------------
    # Rarity-aware scoring
    # --------------------------------------------------------

    if STRATEGY == "rarity_weighted":

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

        weight_tensor = torch.tensor(
            [
                rarity_weights[species]
                for species in CLASS_NAMES
            ],
            dtype=torch.float32
        )

        expected_rarity = (
            probabilities *
            weight_tensor.numpy()
        ).sum(axis=1)

        df["rarity_weight"] = (
            expected_rarity
        )

        # Normalize uncertainty
        u_min = df["uncertainty"].min()
        u_max = df["uncertainty"].max()

        if u_max > u_min:
            df["uncertainty_norm"] = (
                (df["uncertainty"] - u_min)
                /
                (u_max - u_min)
            )
        else:
            df["uncertainty_norm"] = 1.0

        # Normalize rarity
        r_min = df["rarity_weight"].min()
        r_max = df["rarity_weight"].max()

        if r_max > r_min:
            df["rarity_norm"] = (
                (df["rarity_weight"] - r_min)
                /
                (r_max - r_min)
            )
        else:
            df["rarity_norm"] = 1.0

        df["priority_score"] = (
            df["uncertainty_norm"]
            *
            df["rarity_norm"]
        )

        candidates = (
            df.sort_values(
                "priority_score",
                ascending=False
            )
            .drop_duplicates(
                subset=["seq_id"]
            )
        )

    else:

        candidates = (
            df.sort_values(
                "uncertainty",
                ascending=False
            )
            .drop_duplicates(
                subset=["seq_id"]
            )
        )

    selected = candidates.head(
        BUDGET
    ).copy()


# ============================================================
# REMOVE SELECTED SEQUENCES FROM NEXT POOL
# ============================================================

selected_sequences = set(
    selected["seq_id"]
)

remaining = df[
    ~df["seq_id"].isin(
        selected_sequences
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
# SAVE ROUND-2 SELECTION
# ============================================================

selected_path = (
    OUTPUT_DIR /
    "selected_round_2.csv"
)

remaining_path = (
    OUTPUT_DIR /
    "remaining_round_2.csv"
)

selected.to_csv(
    selected_path,
    index=False
)

remaining.to_csv(
    remaining_path,
    index=False
)


# ============================================================
# CREATE ROUND-2 LABELED DATASET
# ============================================================

labeled_df = pd.read_csv(
    LABELED_FILE
)

labeled_round_2 = pd.concat(
    [
        labeled_df,
        selected
    ],
    ignore_index=True
)

labeled_round_2 = (
    labeled_round_2
    .drop_duplicates(
        subset=["crop_file"]
    )
)

labeled_round_2.to_csv(
    OUTPUT_DIR /
    "labeled_round_2.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================")
print("ROUND 2 ACQUISITION COMPLETE")
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
    "Labeled after Round 2:",
    len(labeled_round_2)
)

print(
    "Remaining crops:",
    len(remaining)
)

print(
    "Remaining sequences:",
    remaining["seq_id"].nunique()
)

print("\nSelected species:")
print(
    selected["species"].value_counts()
)

print(
    "\nSaved selected:",
    selected_path
)

print(
    "Saved labeled set:",
    OUTPUT_DIR /
    "labeled_round_2.csv"
)