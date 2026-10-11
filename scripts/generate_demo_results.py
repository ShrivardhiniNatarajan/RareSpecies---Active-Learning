"""Generate verified public/demo_results.json using real backend inference on sample images."""
import asyncio
import json
import os
import sys
from pathlib import Path
from fastapi import UploadFile

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.server import analyze_image, load_all_models

# Sample images from data/raw/
SAMPLE_FILES = [
    "5858bf4a-23d2-11e8-a6a3-ec086b02610b.jpg", # raccoon
    "5858bfd2-23d2-11e8-a6a3-ec086b02610b.jpg", # mountain lion
    "5858c07c-23d2-11e8-a6a3-ec086b02610b.jpg", # coyote
    "5858c1a6-23d2-11e8-a6a3-ec086b02610b.jpg", # bobcat
    "5858c2ee-23d2-11e8-a6a3-ec086b02610b.jpg", # deer
    "585a62c7-23d2-11e8-a6a3-ec086b02610b.jpg", # opossum
]

async def main():
    print("Initializing models...")
    load_all_models()

    raw_dir = ROOT_DIR / "data" / "raw"
    out_file = ROOT_DIR / "frontend" / "public" / "demo_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_detections = []

    for fname in SAMPLE_FILES:
        fpath = raw_dir / fname
        if not fpath.exists():
            print(f"Skipping {fname} (not found in {raw_dir})")
            continue

        print(f"Analyzing {fname}...")
        with open(fpath, "rb") as f:
            content = f.read()
            upload = UploadFile(filename=fname, file=None)
            # Patch read method
            async def custom_read():
                return content
            upload.read = custom_read

            res = await analyze_image(upload)
            print(f"  Found {len(res['detections'])} detections in {fname}")
            all_detections.extend(res["detections"])

    print(f"Total detections collected: {len(all_detections)}")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2)

    print(f"Saved verified demo results to: {out_file} ({out_file.stat().st_size} bytes)")

if __name__ == "__main__":
    asyncio.run(main())
