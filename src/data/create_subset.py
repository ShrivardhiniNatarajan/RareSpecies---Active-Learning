import json
import random
from pathlib import Path

import pandas as pd
import yaml


# ============================================================
# 1. LOAD CONFIGURATION
# ============================================================

CONFIG_PATH = Path("configs/experiment.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


# ============================================================
# 2. READ CONFIGURATION VALUES
# ============================================================

RANDOM_SEED = config["project"]["random_seed"]

JSON_PATH = Path(
    config["dataset"]["metadata_file"]
)

OUTPUT_FILE = Path(
    config["paths"]["metadata_output"]
)

TARGET_COUNTS = config["dataset"]["subset_counts"]


# ============================================================
# 3. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. LOAD CALTECH JSON
# ============================================================

print("Loading Caltech metadata...")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Metadata loaded successfully.")


# ============================================================
# 5. CREATE CATEGORY MAPPING
#
# category_id -> species name
#
# Example:
# 6 -> bobcat
# 9 -> coyote
# ============================================================

category_map = {
    category["id"]: category["name"]
    for category in data["categories"]
}


# ============================================================
# 6. CREATE IMAGE MAPPING
#
# image_id -> complete image metadata
# ============================================================

image_map = {
    image["id"]: image
    for image in data["images"]
}


# ============================================================
# 7. FIND ALL SPECIES ASSOCIATED WITH EACH IMAGE
#
# An image may have multiple annotations.
#
# image_species looks like:
#
# {
#     "image_id_1": {"deer"},
#     "image_id_2": {"coyote"},
#     "image_id_3": {"deer", "bird"}
# }
#
# We will later keep only single-species images.
# ============================================================

image_species = {}

for annotation in data["annotations"]:

    image_id = annotation["image_id"]
    category_id = annotation["category_id"]

    species = category_map.get(category_id)

    if species is None:
        continue

    if image_id not in image_species:
        image_species[image_id] = set()

    image_species[image_id].add(species)


# ============================================================
# 8. BUILD CANDIDATE IMAGE POOLS
#
# We only use images with exactly ONE species.
# This avoids ambiguity during our initial classifier experiment.
# ============================================================

species_images = {
    species: []
    for species in TARGET_COUNTS
}

for image_id, species_set in image_species.items():

    # Ignore multi-species images
    if len(species_set) != 1:
        continue

    species = next(iter(species_set))

    # Ignore species that are not part of our experiment
    if species not in species_images:
        continue

    # Make sure image metadata exists
    if image_id not in image_map:
        continue

    species_images[species].append(image_id)


# ============================================================
# 9. PRINT AVAILABLE DATA
# ============================================================

print("\nAvailable images:")

for species, images in species_images.items():

    print(
        f"{species:15s} : {len(images)}"
    )


# ============================================================
# 10. SELECT RANDOM SAMPLE
# ============================================================

random.seed(RANDOM_SEED)

selected_rows = []

for species, requested_count in TARGET_COUNTS.items():

    available_images = species_images[species]

    print(
        f"\nSelecting {species}: "
        f"{requested_count} images"
    )

    # Prevent requesting more images than exist
    if len(available_images) < requested_count:

        raise ValueError(
            f"Not enough images for '{species}'. "
            f"Requested {requested_count}, "
            f"but only {len(available_images)} "
            f"are available."
        )

    selected_ids = random.sample(
        available_images,
        requested_count
    )

    # --------------------------------------------------------
    # Store metadata for every selected image
    # --------------------------------------------------------

    for image_id in selected_ids:

        image = image_map[image_id]

        selected_rows.append({

            "image_id": image_id,

            "file_name": image["file_name"],

            "species": species,

            "seq_id": image.get("seq_id"),

            "frame_num": image.get("frame_num"),

            "location": image.get("location"),

            "date_captured": image.get(
                "date_captured"
            ),

            "width": image.get("width"),

            "height": image.get("height"),
        })


# ============================================================
# 11. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(selected_rows)


# ============================================================
# 12. SHUFFLE FINAL DATASET
#
# random_state ensures we can reproduce the same subset.
# ============================================================

df = df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ============================================================
# 13. SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 14. PRINT FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("SUBSET CREATION COMPLETE")
print("==============================================")

print("\nSelected images per class:")

print(
    df["species"].value_counts()
)

print(
    f"\nTotal images: {len(df)}"
)

print(
    f"\nSaved metadata to:\n{OUTPUT_FILE}"
)