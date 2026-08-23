from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

METADATA_FILE = Path(
    "data/metadata/subset_metadata.csv"
)

IMAGE_DIR = Path(
    "data/raw"
)

RESULTS_DIR = Path(
    "results/eda"
)


# ============================================================
# SETUP
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD SUBSET METADATA
# ============================================================

print("Loading subset metadata...")

df = pd.read_csv(
    METADATA_FILE
)

expected_files = (
    df["file_name"]
    .astype(str)
    .tolist()
)

print(
    f"Expected images: {len(expected_files)}"
)


# ============================================================
# CHECK FILE EXISTENCE
# ============================================================

print("\nChecking file existence...")

missing_files = []

for file_name in expected_files:

    image_path = IMAGE_DIR / file_name

    if not image_path.exists():
        missing_files.append(file_name)


print(
    f"Missing files: {len(missing_files)}"
)


# ============================================================
# CHECK IMAGE INTEGRITY
# ============================================================

print("\nChecking image integrity...")

corrupt_files = []

valid_files = []

for file_name in tqdm(
    expected_files,
    desc="Validating images",
    unit="img"
):

    image_path = IMAGE_DIR / file_name

    # Missing files were already recorded
    if not image_path.exists():
        continue

    # Check for empty files
    if image_path.stat().st_size == 0:

        corrupt_files.append({
            "file_name": file_name,
            "reason": "zero_byte_file"
        })

        continue

    try:

        # Open image
        with Image.open(image_path) as image:

            # Verify JPEG/image structure
            image.verify()

        valid_files.append(file_name)

    except Exception as error:

        corrupt_files.append({
            "file_name": file_name,
            "reason": str(error)
        })


# ============================================================
# SAVE MISSING FILES
# ============================================================

if missing_files:

    missing_df = pd.DataFrame({
        "file_name": missing_files
    })

    missing_path = (
        RESULTS_DIR /
        "missing_files.csv"
    )

    missing_df.to_csv(
        missing_path,
        index=False
    )

    print(
        f"Missing-file report saved to: "
        f"{missing_path}"
    )


# ============================================================
# SAVE CORRUPT FILES
# ============================================================

if corrupt_files:

    corrupt_df = pd.DataFrame(
        corrupt_files
    )

    corrupt_path = (
        RESULTS_DIR /
        "corrupt_files.csv"
    )

    corrupt_df.to_csv(
        corrupt_path,
        index=False
    )

    print(
        f"Corrupt-file report saved to: "
        f"{corrupt_path}"
    )


# ============================================================
# CHECK DOWNLOADED CLASS DISTRIBUTION
# ============================================================

print("\n============================================")
print("CLASS DISTRIBUTION")
print("============================================")

existing_df = df[
    df["file_name"].isin(valid_files)
].copy()

class_counts = (
    existing_df["species"]
    .value_counts()
)

print(class_counts)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n============================================")
print("DATASET VERIFICATION SUMMARY")
print("============================================")

print(
    f"Expected images       : {len(expected_files)}"
)

print(
    f"Valid images          : {len(valid_files)}"
)

print(
    f"Missing images        : {len(missing_files)}"
)

print(
    f"Corrupt images        : {len(corrupt_files)}"
)

print(
    f"Total files in folder : "
    f"{len(list(IMAGE_DIR.iterdir()))}"
)


# ============================================================
# FINAL PASS / FAIL
# ============================================================

if (
    len(missing_files) == 0
    and len(corrupt_files) == 0
):

    print(
        "\n✅ DATASET VERIFICATION PASSED"
    )

else:

    print(
        "\n⚠️ DATASET VERIFICATION FOUND ISSUES"
    )