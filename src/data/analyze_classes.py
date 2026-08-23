import json
from pathlib import Path
from collections import defaultdict

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

print("Metadata loaded successfully.\n")


# ============================================================
# 3. Build category_id -> category_name mapping
#
# Example:
# 6  -> bobcat
# 1  -> opossum
# 9  -> coyote
# ============================================================

category_map = {
    category["id"]: category["name"]
    for category in data["categories"]
}

print("Category mapping:")
for category_id, category_name in category_map.items():
    print(f"{category_id:>4} -> {category_name}")


# ============================================================
# 4. Count annotations by category
#
# Each annotation tells us:
#   image_id
#   category_id
#
# We convert category_id into the actual species name.
# ============================================================

class_counts = defaultdict(int)

for annotation in data["annotations"]:

    category_id = annotation["category_id"]

    category_name = category_map.get(
        category_id,
        "unknown"
    )

    class_counts[category_name] += 1


# ============================================================
# 5. Convert to DataFrame
# ============================================================

df = pd.DataFrame(
    list(class_counts.items()),
    columns=["species", "count"]
)

df = df.sort_values(
    by="count",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 6. Calculate percentage of annotations
# ============================================================

total = df["count"].sum()

df["percentage"] = (
    df["count"] / total * 100
)


# ============================================================
# 7. Display results
# ============================================================

print("\n==============================================")
print("CLASS DISTRIBUTION")
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
# 8. Save CSV
# ============================================================

csv_path = RESULTS_DIR / "class_distribution.csv"

df.to_csv(
    csv_path,
    index=False
)

print(
    f"\nSaved class distribution to: {csv_path}"
)


# ============================================================
# 9. Plot class distribution
# ============================================================

plt.figure(figsize=(12, 7))

plt.bar(
    df["species"],
    df["count"]
)

plt.xlabel("Species / Category")
plt.ylabel("Number of Annotations")
plt.title("Caltech Camera Traps - Class Distribution")

plt.xticks(
    rotation=60,
    ha="right"
)

plt.tight_layout()


plot_path = RESULTS_DIR / "class_distribution.png"

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()

print(
    f"Saved class distribution plot to: {plot_path}"
)


# ============================================================
# 10. Identify rare classes
#
# Here we simply show the least frequent classes.
# We are NOT defining the final rarity threshold yet.
# ============================================================

print("\n==============================================")
print("LEAST FREQUENT CLASSES")
print("==============================================")

print(
    df.tail(8).to_string(
        index=False,
        formatters={
            "percentage": "{:.2f}%".format
        }
    )
)


print("\nAnalysis complete.")