"""Back up Round-3 pools, checkpoints, and result files to mounted Google Drive."""

from __future__ import annotations

import shutil
from pathlib import Path

DRIVE_ROOT = Path("/content/gdrive/MyDrive/RareCam")
STRATEGIES = ("random", "uncertainty", "rarity_weighted")


def copy_file(source: Path, destination: Path) -> bool:
    if not source.exists():
        print(f"SKIP (not found): {source}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Backed up: {destination}")
    return True


def main() -> None:
    copied = 0
    for strategy in STRATEGIES:
        data_dir = Path("data/active_learning") / strategy
        for filename in ("selected_round_3.csv", "remaining_round_3.csv", "labeled_round_3.csv"):
            copied += copy_file(
                data_dir / filename,
                DRIVE_ROOT / "active_learning" / "round_3" / f"{strategy}_{filename}",
            )
        copied += copy_file(
            Path("models/active_learning") / f"{strategy}_round_3.pth",
            DRIVE_ROOT / "active_learning" / "models" / f"{strategy}_round_3.pth",
        )
        result_dir = Path("results/active_learning") / strategy / "round_3"
        if result_dir.exists():
            for source in result_dir.iterdir():
                if source.is_file():
                    copied += copy_file(
                        source,
                        DRIVE_ROOT / "active_learning" / "results" / "round_3" / strategy / source.name,
                    )
    print(f"\nBackup routine finished. Files copied: {copied}")


if __name__ == "__main__":
    main()
