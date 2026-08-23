import json
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Paths
# ============================================================

JSON_PATH = Path("data/metadata/caltech_images_20210113.json")
RESULTS_DIR = Path("results/eda")

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Load metadata
# ============================================================

print("Loading Caltech metadata...")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Metadata loaded successfully.")


# ============================================================
# 3. Create category_id -> species mapping
# ============================================================

category_map = {
    category["id"]: category["name"]
    for category in data["categories"]
}


# ============================================================
# 4. Create image_id -> image information mapping
# ============================================================

image_map = {
    image["id"]: image
    for image in data["images"]
}


# ============================================================
# 5. Find which species occur in each image
#
# One image can have multiple annotations.
# Therefore we store a SET of species for each image.
# ============================================================

image_species = defaultdict(set)

for annotation in data["annotations"]:

    image_id = annotation["image_id"]

    category_id = annotation["category_id"]

    species = category_map.get(
        category_id,
        "unknown"
    )

    image_species[image_id].add(species)


# ============================================================
# 6. Count UNIQUE IMAGES for each species
# ============================================================

species_images = defaultdict(set)

for image_id, species_set in image_species.items():

    for species in species_set:

        species_images[species].add(image_id)


# ============================================================
# 7. Convert to counts
# ============================================================

rows = []

for species, image_ids in species_images.items():

    rows.append({
        "species": species,
        "unique_images": len(image_ids)
    })


df = pd.DataFrame(rows)

df = df.sort_values(
    by="unique_images",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 8. Calculate percentage
# ============================================================

total_species_images = df["unique_images"].sum()

df["percentage"] = (
    df["unique_images"]
    / total_species_images
    * 100
)


# ============================================================
# 9. Print full table
# ============================================================

print("\n==============================================")
print("UNIQUE IMAGE COUNT PER CATEGORY")
print("==============================================")

print(
    df.to_string(
        index=False,
        formatters={
            "percentage": "{:.2f}%".format
        }
    )
)


# ============================================================
# 10. Separate non-animal categories
# ============================================================

non_animal = {
    "empty",
    "car"
}

animal_df = df[
    ~df["species"].isin(non_animal)
].copy()


print("\n==============================================")
print("ANIMAL CLASSES ONLY")
print("==============================================")

print(
    animal_df.to_string(
        index=False,
        formatters={
            "percentage": "{:.2f}%".format
        }
    )
)


# ============================================================
# 11. Save animal-only distribution
# ============================================================

animal_csv = RESULTS_DIR / "animal_image_distribution.csv"

animal_df.to_csv(
    animal_csv,
    index=False
)

print(
    f"\nSaved animal distribution to: {animal_csv}"
)


# ============================================================
# 12. Plot animal class distribution
# ============================================================

plt.figure(figsize=(12, 7))

plt.bar(
    animal_df["species"],
    animal_df["unique_images"]
)

plt.xlabel("Species")
plt.ylabel("Unique Images")
plt.title(
    "Caltech Camera Traps - "
    "Animal Class Distribution"
)

plt.xticks(
    rotation=60,
    ha="right"
)

plt.tight_layout()

plot_path = (
    RESULTS_DIR /
    "animal_class_distribution.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()

print(
    f"Saved plot to: {plot_path}"
)


# ============================================================
# 13. Show rarest animal classes
# ============================================================

print("\n==============================================")
print("RAREST ANIMAL CLASSES")
print("==============================================")

print(
    animal_df.tail(8).to_string(
        index=False,
        formatters={
            "percentage": "{:.2f}%".format
        }
    )
)


# ============================================================
# 14. Check how many images have multiple species
# ============================================================

multi_species_images = sum(
    1
    for species_set in image_species.values()
    if len(species_set) > 1
)

single_species_images = sum(
    1
    for species_set in image_species.values()
    if len(species_set) == 1
)

print("\n==============================================")
print("IMAGE LABEL STRUCTURE")
print("==============================================")

print(
    "Images with one category:",
    single_species_images
)

print(
    "Images with multiple categories:",
    multi_species_images
)

print("\nAnalysis complete.")