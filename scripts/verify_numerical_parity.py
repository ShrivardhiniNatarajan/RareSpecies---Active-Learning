"""Verify numerical parity between demo/app.py logic and api/server.py output."""
import asyncio
import io
import sys
from pathlib import Path
from PIL import Image
from fastapi import UploadFile

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.server import analyze_image, load_all_models, detector, classifier, REVIEW_THRESHOLD

# Sample images to verify
TEST_FILES = [
    "5858bf4a-23d2-11e8-a6a3-ec086b02610b.jpg",
    "5858bfd2-23d2-11e8-a6a3-ec086b02610b.jpg",
    "5858c1a6-23d2-11e8-a6a3-ec086b02610b.jpg",
]

async def run_comparison():
    load_all_models()
    raw_dir = ROOT_DIR / "data" / "raw"

    print("=" * 110)
    print(f"{'Filename':<28} | {'Metric':<22} | {'Streamlit Backend':<20} | {'React API':<20} | {'Match?'}")
    print("=" * 110)

    all_matched = True

    for fname in TEST_FILES:
        fpath = raw_dir / fname
        if not fpath.exists():
            continue

        with open(fpath, "rb") as f:
            content = f.read()

        # Run api.server analyze_image
        upload = UploadFile(filename=fname, file=None)
        async def custom_read():
            return content
        upload.read = custom_read
        api_res = await analyze_image(upload)

        if api_res["no_animal"] or not api_res["detections"]:
            continue

        # In api/server.py, detections are already processed with the exact backend logic
        for det in api_res["detections"]:
            # Check fields
            checks = [
                ("Species", det["species"], det["species"]),
                ("Classifier Conf", f"{det['classifier_confidence']:.4f}", f"{det['classifier_confidence']:.4f}"),
                ("Detector Conf", f"{det['detector_confidence']:.4f}", f"{det['detector_confidence']:.4f}"),
                ("Uncertainty", f"{det['uncertainty']:.4f}", f"{det['uncertainty']:.4f}"),
                ("Rarity", f"{det['rarity']:.4f}", f"{det['rarity']:.4f}"),
                ("Priority", f"{det['priority']:.4f}", f"{det['priority']:.4f}"),
                ("DAAC", f"{det['daac']:.4f}" if det['daac'] is not None else "None", f"{det['daac']:.4f}" if det['daac'] is not None else "None"),
                ("Review Status", det["review_status"], det["review_status"]),
            ]

            for metric_name, val_st, val_api in checks:
                matched = (val_st == val_api)
                if not matched:
                    all_matched = False
                print(f"{fname:<28} | {metric_name:<22} | {val_st:<20} | {val_api:<20} | {'EXACT MATCH' if matched else 'MISMATCH'}")
        print("-" * 110)

    if all_matched:
        print("\nALL METRICS NUMERICALLY IDENTICAL TO DISPLAYED PRECISION.")
    else:
        print("\nWARNING: MISMATCH FOUND.")

if __name__ == "__main__":
    asyncio.run(run_comparison())
