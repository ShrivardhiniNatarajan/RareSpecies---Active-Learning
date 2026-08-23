import json
from pathlib import Path

# ---------------------------------------------------------
# 1. Path to the Caltech metadata JSON
# ---------------------------------------------------------

JSON_PATH = Path("data/metadata/caltech_images_20210113.json")

# ---------------------------------------------------------
# 2. Load JSON
# ---------------------------------------------------------

print("Loading dataset metadata...")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\nJSON loaded successfully!")
print("Top-level type:", type(data))

# ---------------------------------------------------------
# 3. Show top-level keys
# ---------------------------------------------------------

if isinstance(data, dict):
    print("\nTop-level keys:")
    for key in data.keys():
        value = data[key]

        if isinstance(value, list):
            print(f"  {key}: list with {len(value)} items")
        elif isinstance(value, dict):
            print(f"  {key}: dictionary with {len(value)} keys")
        else:
            print(f"  {key}: {type(value).__name__}")

# ---------------------------------------------------------
# 4. Inspect first image record
# ---------------------------------------------------------

if "images" in data and len(data["images"]) > 0:

    first_image = data["images"][0]

    print("\n--------------------------------")
    print("FIRST IMAGE RECORD")
    print("--------------------------------")

    for key, value in first_image.items():
        print(f"{key}: {value}")

else:
    print("\nNo 'images' field found.")

# ---------------------------------------------------------
# 5. Inspect categories if present
# ---------------------------------------------------------

if "categories" in data:

    print("\n--------------------------------")
    print("CATEGORIES")
    print("--------------------------------")

    categories = data["categories"]

    print("Number of categories:", len(categories))

    for category in categories[:20]:
        print(category)

# ---------------------------------------------------------
# 6. Inspect annotations if present
# ---------------------------------------------------------

if "annotations" in data:

    print("\n--------------------------------")
    print("ANNOTATIONS")
    print("--------------------------------")

    annotations = data["annotations"]

    print("Number of annotations:", len(annotations))

    if len(annotations) > 0:
        print("\nFirst annotation:")
        print(annotations[0])

print("\n================================")
print("DATASET INSPECTION COMPLETE")
print("================================")