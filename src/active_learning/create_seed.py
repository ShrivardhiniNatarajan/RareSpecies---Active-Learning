from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = Path(
    "data/splits/train.csv"
)

OUTPUT_DIR = Path(
    "data/active_learning"
)

SEED_FRACTION = 0.20
RANDOM_SEED = 42


# ============================================================
# LOAD TRAINING DATA
# ============================================================

df = pd.read_csv(
    TRAIN_FILE
)

print("Total training crops:", len(df))
print(
    "Unique training sequences:",
    df["seq_id"].nunique()
)


# ============================================================
# CREATE ONE LABEL PER SEQUENCE
# ============================================================

sequence_labels = (
    df.groupby("seq_id")["species"]
      .agg(
          lambda x:
          x.value_counts().index[0]
      )
      .reset_index()
      .rename(
          columns={
              "species": "sequence_species"
          }
      )
)


# ============================================================
# SELECT 20% OF SEQUENCES
# ============================================================

seed_sequences, unlabeled_sequences = train_test_split(
    sequence_labels,
    test_size=1 - SEED_FRACTION,
    random_state=RANDOM_SEED,
    stratify=sequence_labels[
        "sequence_species"
    ]
)


# ============================================================
# GET SEQUENCE IDs
# ============================================================

seed_ids = set(
    seed_sequences["seq_id"]
)

unlabeled_ids = set(
    unlabeled_sequences["seq_id"]
)


# ============================================================
# CREATE SEED / UNLABELED DATASETS
# ============================================================

labeled_seed = df[
    df["seq_id"].isin(seed_ids)
].copy()

unlabeled_pool = df[
    df["seq_id"].isin(unlabeled_ids)
].copy()


# ============================================================
# VERIFY NO OVERLAP
# ============================================================

overlap = (
    set(labeled_seed["seq_id"])
    &
    set(unlabeled_pool["seq_id"])
)

print(
    "\nSequence overlap:",
    len(overlap)
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

labeled_seed.to_csv(
    OUTPUT_DIR / "labeled_seed.csv",
    index=False
)

unlabeled_pool.to_csv(
    OUTPUT_DIR / "unlabeled_pool.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================")
print("ACTIVE LEARNING SEED CREATED")
print("============================================")

print(
    "Labeled seed crops:",
    len(labeled_seed)
)

print(
    "Unlabeled pool crops:",
    len(unlabeled_pool)
)

print(
    "Labeled sequences:",
    labeled_seed["seq_id"].nunique()
)

print(
    "Unlabeled sequences:",
    unlabeled_pool["seq_id"].nunique()
)

print("\nSeed species distribution:")
print(
    labeled_seed["species"].value_counts()
)

print("\nUnlabeled species distribution:")
print(
    unlabeled_pool["species"].value_counts()
)