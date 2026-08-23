from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

UNLABELED_FILE = Path(
    "data/active_learning/unlabeled_pool.csv"
)

OUTPUT_DIR = Path(
    "data/active_learning/random"
)

RANDOM_SEED = 42

BUDGET = 150


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD UNLABELED POOL
# ============================================================

df = pd.read_csv(
    UNLABELED_FILE
)

print(
    "Unlabeled pool:",
    len(df)
)


# ============================================================
# RANDOM SAMPLE
# ============================================================

if len(df) < BUDGET:

    raise ValueError(
        f"Only {len(df)} samples available, "
        f"but requested {BUDGET}."
    )


selected = df.sample(
    n=BUDGET,
    random_state=RANDOM_SEED
)

remaining = df.drop(
    selected.index
)


# ============================================================
# SAVE SELECTED SAMPLES
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
print("RANDOM SAMPLING - ROUND 1")
print("============================================")

print(
    "Selected:",
    len(selected)
)

print(
    "Remaining:",
    len(remaining)
)

print("\nSelected species distribution:")

print(
    selected["species"].value_counts()
)

print(
    "\nSaved to:",
    OUTPUT_DIR
)