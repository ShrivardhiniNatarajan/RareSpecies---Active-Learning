import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/classifier_metadata.csv"
)

OUTPUT_DIR = Path(
    "data/splits"
)

RANDOM_SEED = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# CHECK SPLIT SIZES
# ============================================================

if abs(
    TRAIN_SIZE + VAL_SIZE + TEST_SIZE - 1.0
) > 1e-6:

    raise ValueError(
        "Train + validation + test "
        "must sum to 1.0"
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD CLASSIFIER METADATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Total crops:",
    len(df)
)

print(
    "Unique sequences:",
    df["seq_id"].nunique()
)


# ============================================================
# DETERMINE THE DOMINANT SPECIES OF EACH SEQUENCE
# ============================================================
#
# A sequence normally contains one species in our subset.
# If multiple species occur, the most frequent species
# becomes the sequence's representative label for splitting.
# ============================================================

sequence_species = (
    df.groupby("seq_id")["species"]
      .agg(lambda x: x.value_counts().index[0])
      .reset_index()
      .rename(
          columns={
              "species": "sequence_species"
          }
      )
)


print("\nSequence-level species counts:")
print(
    sequence_species[
        "sequence_species"
    ].value_counts()
)


# ============================================================
# FIRST SPLIT
#
# 70% train
# 30% temporary
# ============================================================

train_sequences, temp_sequences = train_test_split(
    sequence_species,
    test_size=(
        VAL_SIZE + TEST_SIZE
    ),
    random_state=RANDOM_SEED,
    stratify=sequence_species[
        "sequence_species"
    ]
)


# ============================================================
# SECOND SPLIT
#
# Divide the remaining 30%:
#
# 15% validation
# 15% test
#
# So among the temporary 30%:
# validation = 50%
# test       = 50%
# ============================================================

val_sequences, test_sequences = train_test_split(
    temp_sequences,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_sequences[
        "sequence_species"
    ]
)


# ============================================================
# GET SEQUENCE IDs
# ============================================================

train_ids = set(
    train_sequences["seq_id"]
)

val_ids = set(
    val_sequences["seq_id"]
)

test_ids = set(
    test_sequences["seq_id"]
)


# ============================================================
# ASSIGN EVERY CROP BASED ON ITS SEQUENCE
# ============================================================

def assign_split(seq_id):

    if seq_id in train_ids:
        return "train"

    if seq_id in val_ids:
        return "val"

    if seq_id in test_ids:
        return "test"

    raise ValueError(
        f"Unknown sequence: {seq_id}"
    )


df["split"] = df["seq_id"].apply(
    assign_split
)


# ============================================================
# SANITY CHECK — NO SEQUENCE LEAKAGE
# ============================================================

train_set = set(
    df.loc[
        df["split"] == "train",
        "seq_id"
    ]
)

val_set = set(
    df.loc[
        df["split"] == "val",
        "seq_id"
    ]
)

test_set = set(
    df.loc[
        df["split"] == "test",
        "seq_id"
    ]
)

print("\nSequence overlap checks:")

print(
    "Train ∩ Validation:",
    len(train_set & val_set)
)

print(
    "Train ∩ Test:",
    len(train_set & test_set)
)

print(
    "Validation ∩ Test:",
    len(val_set & test_set)
)


# ============================================================
# PRINT SPLIT SIZES
# ============================================================

print("\nCrop counts:")
print(
    df["split"].value_counts()
)


print("\nSpecies distribution by split:")

print(
    pd.crosstab(
        df["species"],
        df["split"]
    )
)


# ============================================================
# SAVE EACH SPLIT
# ============================================================

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "val"
].copy()

test_df = df[
    df["split"] == "test"
].copy()


train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "val.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)


# Save the complete metadata with split assignments
df.to_csv(
    Path(
        "data/processed/"
        "classifier_metadata_with_splits.csv"
    ),
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n============================================")
print("DATASET SPLIT COMPLETE")
print("============================================")

print(
    f"Train crops      : {len(train_df)}"
)

print(
    f"Validation crops : {len(val_df)}"
)

print(
    f"Test crops       : {len(test_df)}"
)

print(
    f"\nFiles saved in: {OUTPUT_DIR}"
)