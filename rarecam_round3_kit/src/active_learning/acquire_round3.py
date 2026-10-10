"""Acquire Round 3 samples for RareCam's three existing AL strategies.

Run from the repository root:
  python -m src.active_learning.acquire_round3 --strategy random
  python -m src.active_learning.acquire_round3 --strategy uncertainty
  python -m src.active_learning.acquire_round3 --strategy rarity_weighted

Selection uses only images, the Round-2 model predictions, and (for rarity-aware)
class counts from labeled_round_2.csv. Ground-truth species in the unlabeled CSV is
not read by the inference Dataset; it is only retained in the selected CSV to
simulate human annotation after selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from src.training.dataset import CLASS_NAMES, eval_transform


BUDGET = 150
BATCH_SIZE = 32
RANDOM_SEED = 44
STRATEGIES = ("random", "uncertainty", "rarity_weighted")
AL_DIR = Path("data/active_learning")
MODEL_DIR = Path("models/active_learning")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CropOnlyDataset(Dataset):
    """Inference-only dataset: reads crop_file, never returns/reads labels."""

    def __init__(self, frame: pd.DataFrame):
        self.paths = frame["crop_file"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = Path(self.paths[index].replace("\\", "/"))
        if not image_path.exists():
            raise FileNotFoundError(f"Crop not found: {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return eval_transform(image)


def load_checkpoint(path: Path):
    # These are project-generated checkpoints. Explicit weights_only=False keeps
    # compatibility with checkpoints containing simple metadata as well as tensors.
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:  # older PyTorch versions
        return torch.load(path, map_location=DEVICE)


def load_round2_model(path: Path) -> torch.nn.Module:
    if not path.exists():
        raise FileNotFoundError(
            f"Round-2 checkpoint missing: {path}. Restore it from Google Drive first."
        )
    checkpoint = load_checkpoint(path)
    saved_classes = checkpoint.get("class_names")
    if saved_classes is not None and list(saved_classes) != list(CLASS_NAMES):
        raise ValueError(
            f"Checkpoint class order {saved_classes} does not match {CLASS_NAMES}."
        )
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def normalize_minmax(values: pd.Series) -> pd.Series:
    lo = float(values.min())
    hi = float(values.max())
    if np.isclose(hi, lo):
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    return (values - lo) / (hi - lo)


def predict_scores(df: pd.DataFrame, model: torch.nn.Module) -> tuple[np.ndarray, np.ndarray]:
    missing = [p for p in df["crop_file"].astype(str) if not Path(p.replace("\\", "/")).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} crop files are missing. Example: {missing[0]}. "
            "Restore data/cropped from the Drive archive before acquisition."
        )

    dataset = CropOnlyDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2 if DEVICE == "cuda" else 0,
        pin_memory=(DEVICE == "cuda"),
    )
    entropies: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []

    with torch.inference_mode():
        for batch_index, images in enumerate(loader, start=1):
            images = images.to(DEVICE, non_blocking=(DEVICE == "cuda"))
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            probs_np = probs.cpu().numpy()
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
            probability_batches.append(probs_np)
            entropies.append(entropy.cpu().numpy())
            if batch_index % 25 == 0:
                print(f"Scored {min(batch_index * BATCH_SIZE, len(df))}/{len(df)} crops...")

    probabilities = np.concatenate(probability_batches, axis=0)
    uncertainty = np.concatenate(entropies, axis=0)
    return uncertainty, probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    args = parser.parse_args()
    strategy = args.strategy

    remaining_file = AL_DIR / strategy / "remaining_round_2.csv"
    labeled_file = AL_DIR / strategy / "labeled_round_2.csv"
    model_file = MODEL_DIR / f"{strategy}_round_2.pth"
    output_dir = AL_DIR / strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    if not remaining_file.exists():
        raise FileNotFoundError(f"Missing Round-2 pool: {remaining_file}")
    if not labeled_file.exists():
        raise FileNotFoundError(f"Missing Round-2 labeled set: {labeled_file}")

    pool = pd.read_csv(remaining_file)
    labeled = pd.read_csv(labeled_file)
    required = {"crop_file", "species", "seq_id"}
    for label, frame in (("remaining pool", pool), ("labeled set", labeled)):
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{label} is missing columns: {sorted(missing_columns)}")
        if frame["seq_id"].isna().any():
            raise ValueError(f"{label} contains missing seq_id values.")

    print("=" * 64)
    print(f"RARECAM ROUND 3 ACQUISITION — {strategy.upper()}")
    print("=" * 64)
    print(f"Device: {DEVICE}")
    print(f"Round-2 labeled crops: {len(labeled)}")
    print(f"Remaining crops: {len(pool)}")
    print(f"Remaining sequences: {pool['seq_id'].nunique()}")
    print(f"Acquisition budget: {BUDGET} (one crop per sequence)")

    # Use only image/model signals to rank candidates.
    if strategy == "random":
        candidates = (
            pool.sample(frac=1, random_state=RANDOM_SEED)
            .drop_duplicates(subset=["seq_id"])
        )
        selected = candidates.head(BUDGET).copy()
    else:
        model = load_round2_model(model_file)
        uncertainty, probabilities = predict_scores(pool, model)
        pool = pool.copy()
        pool["uncertainty"] = uncertainty

        if strategy == "uncertainty":
            candidates = pool.sort_values("uncertainty", ascending=False).drop_duplicates("seq_id")
        else:
            counts = (
                labeled["species"].value_counts()
                .reindex(CLASS_NAMES, fill_value=0)
                .astype(float)
            )
            if (counts <= 0).any():
                absent = counts[counts <= 0].index.tolist()
                raise ValueError(f"Cannot estimate rarity; no labeled examples for: {absent}")
            raw_weights = 1.0 / (counts + 1e-8)
            rarity_weights = raw_weights / raw_weights.mean()
            weight_vector = np.asarray([rarity_weights[c] for c in CLASS_NAMES], dtype=np.float32)
            pool["expected_rarity"] = probabilities @ weight_vector
            pool["uncertainty_norm"] = normalize_minmax(pool["uncertainty"])
            pool["rarity_norm"] = normalize_minmax(pool["expected_rarity"])
            # Match the rarity-aware formula used in Rounds 1 and 2.
            pool["priority_score"] = pool["uncertainty_norm"] * pool["rarity_norm"]
            candidates = pool.sort_values("priority_score", ascending=False).drop_duplicates("seq_id")

        selected = candidates.head(BUDGET).copy()

    if len(selected) != BUDGET:
        raise ValueError(f"Selected {len(selected)} unique sequences, but budget is {BUDGET}.")
    if selected["seq_id"].nunique() != BUDGET:
        raise AssertionError("Selection contains duplicate sequences.")

    selected_seq_ids = set(selected["seq_id"])
    labeled_seq_ids = set(labeled["seq_id"])
    if selected_seq_ids & labeled_seq_ids:
        raise AssertionError("Selected sequences overlap the existing labeled set.")

    # Remove every crop belonging to a selected sequence from the remaining pool.
    remaining = pool.loc[~pool["seq_id"].isin(selected_seq_ids)].copy()
    if selected_seq_ids & set(remaining["seq_id"]):
        raise AssertionError("Selected sequences remain in the unlabeled pool.")

    # Simulated human oracle: reveal the existing ground-truth labels only AFTER selection.
    labeled_round3 = pd.concat([labeled, selected], ignore_index=True)
    labeled_round3 = labeled_round3.drop_duplicates(subset=["crop_file"]).reset_index(drop=True)
    if len(labeled_round3) != len(labeled) + BUDGET:
        raise AssertionError(
            f"Expected {len(labeled) + BUDGET} labeled rows, got {len(labeled_round3)}."
        )

    selected.to_csv(output_dir / "selected_round_3.csv", index=False)
    remaining.to_csv(output_dir / "remaining_round_3.csv", index=False)
    labeled_round3.to_csv(output_dir / "labeled_round_3.csv", index=False)

    print("\nROUND 3 ACQUISITION COMPLETE")
    print(f"Selected crops: {len(selected)}")
    print(f"Selected unique sequences: {selected['seq_id'].nunique()}")
    print(f"Labeled after Round 3: {len(labeled_round3)}")
    print(f"Remaining crops: {len(remaining)}")
    print(f"Remaining sequences: {remaining['seq_id'].nunique()}")
    # This is an offline simulation diagnostic only; species labels were not used to rank the pool.
    print("\nSelected class distribution (ground-truth diagnostic after selection only):")
    print(selected["species"].value_counts())
    print(f"\nSaved under: {output_dir}")


if __name__ == "__main__":
    main()
