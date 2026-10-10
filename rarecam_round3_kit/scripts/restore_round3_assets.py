"""Restore the Round-2 data/checkpoints into a fresh Colab working directory.

Mount Drive at /content/gdrive first, cd to the RareCam repo root, then run:
  python scripts/restore_round3_assets.py
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path.cwd()
DRIVE_ROOT = Path("/content/gdrive/MyDrive/RareCam")
STRATEGIES = ("random", "uncertainty", "rarity_weighted")


def copy_required(source: Path, destination: Path, missing: list[str]) -> None:
    if not source.exists():
        missing.append(str(source))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Restored: {destination}")


def main() -> None:
    if not (REPO / "src" / "training" / "dataset.py").exists():
        raise RuntimeError("Run this script from the RareCam repository root.")
    if not DRIVE_ROOT.exists():
        raise FileNotFoundError(
            f"Drive path not found: {DRIVE_ROOT}. Mount Google Drive at /content/gdrive first."
        )

    missing: list[str] = []

    # Restore the fixed splits. Round 3 trains on labeled_round_3.csv but validates/tests
    # on the same original val.csv and test.csv.
    for filename in ("train.csv", "val.csv", "test.csv"):
        copy_required(
            DRIVE_ROOT / "classification" / filename,
            REPO / "data" / "splits" / filename,
            missing,
        )

    # Restore the crop archive once. Its expected ZIP layout is species-folder/crop.jpg.
    crop_archive = DRIVE_ROOT / "megadetector" / "cropped_images.zip"
    crop_dir = REPO / "data" / "cropped"
    existing_crops = list(crop_dir.rglob("*.jpg")) if crop_dir.exists() else []
    if len(existing_crops) != 6558:
        if not crop_archive.exists():
            missing.append(str(crop_archive))
        else:
            shutil.rmtree(crop_dir, ignore_errors=True)
            crop_dir.mkdir(parents=True, exist_ok=True)
            print(f"Extracting {crop_archive} ...")
            with zipfile.ZipFile(crop_archive, "r") as archive:
                archive.extractall(crop_dir)
            restored_count = len(list(crop_dir.rglob("*.jpg")))
            print(f"Crops restored: {restored_count}")
            if restored_count != 6558:
                raise RuntimeError(f"Expected 6558 crops, found {restored_count}.")
    else:
        print(f"Crops already present: {len(existing_crops)}")

    # Restore Round-2 CSVs and checkpoints, which are the inputs to Round-3 acquisition.
    round2_drive = DRIVE_ROOT / "active_learning" / "round_2"
    drive_models = DRIVE_ROOT / "active_learning" / "models"
    for strategy in STRATEGIES:
        local_strategy_dir = REPO / "data" / "active_learning" / strategy
        for name in ("selected_round_2.csv", "remaining_round_2.csv", "labeled_round_2.csv"):
            copy_required(
                round2_drive / f"{strategy}_{name}",
                local_strategy_dir / name,
                missing,
            )
        copy_required(
            drive_models / f"{strategy}_round_2.pth",
            REPO / "models" / "active_learning" / f"{strategy}_round_2.pth",
            missing,
        )

    if missing:
        print("\nMISSING REQUIRED DRIVE ARTIFACTS:")
        for path in missing:
            print(f" - {path}")
        raise FileNotFoundError(
            "Some required artifacts were not restored. Do not run Round 3 until each missing path is located."
        )

    print("\nRESTORE COMPLETE — VALIDATING INPUTS")
    for filename, expected in (("train.csv", 4617), ("val.csv", 971), ("test.csv", 970)):
        df = pd.read_csv(REPO / "data" / "splits" / filename)
        print(f"{filename}: {len(df)} rows (expected {expected})")
        if len(df) != expected:
            raise RuntimeError(f"Unexpected row count in {filename}.")

    for strategy in STRATEGIES:
        labeled = pd.read_csv(REPO / "data" / "active_learning" / strategy / "labeled_round_2.csv")
        selected = pd.read_csv(REPO / "data" / "active_learning" / strategy / "selected_round_2.csv")
        model = REPO / "models" / "active_learning" / f"{strategy}_round_2.pth"
        print(
            f"{strategy}: labeled R2={len(labeled)}, selected R2={len(selected)}, "
            f"unique selected sequences={selected['seq_id'].nunique()}, model={model.exists()}"
        )
        if len(labeled) != 1244 or len(selected) != 150 or selected["seq_id"].nunique() != 150 or not model.exists():
            raise RuntimeError(f"Round-2 inputs failed validation for {strategy}.")

    # Check crops referenced by all three Round-2 pools and the test set.
    check_frames = [pd.read_csv(REPO / "data" / "splits" / "test.csv")]
    check_frames.extend(
        pd.read_csv(REPO / "data" / "active_learning" / strategy / "remaining_round_2.csv")
        for strategy in STRATEGIES
    )
    checked_paths = set()
    missing_crops = []
    for frame in check_frames:
        if "crop_file" not in frame.columns:
            raise RuntimeError("A required CSV has no crop_file column.")
        for raw_path in frame["crop_file"].dropna().astype(str):
            path = Path(raw_path.replace("\\", "/"))
            if str(path) in checked_paths:
                continue
            checked_paths.add(str(path))
            if not path.exists():
                missing_crops.append(str(path))
    if missing_crops:
        raise FileNotFoundError(
            f"{len(missing_crops)} referenced crop files are missing; example: {missing_crops[0]}"
        )
    print(f"Crop paths verified: {len(checked_paths)} unique paths")
    print("All Round-3 prerequisite artifacts are present.")


if __name__ == "__main__":
    main()
