"""Evaluate an active-learning model on the fixed held-out test split.

Example:
  python -m src.active_learning.evaluate_round --strategy random --round 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import models

from src.training.dataset import CLASS_NAMES, CameraTrapDataset, eval_transform


STRATEGIES = ("random", "uncertainty", "rarity_weighted")
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--round", required=True, type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    round_number = args.round

    test_csv = Path("data/splits/test.csv")
    model_path = Path("models/active_learning") / f"{args.strategy}_round_{round_number}.pth"
    results_dir = Path("results/active_learning") / args.strategy / f"round_{round_number}"
    results_dir.mkdir(parents=True, exist_ok=True)

    for path in (test_csv, model_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    test_df = pd.read_csv(test_csv)
    missing = [p for p in test_df["crop_file"].astype(str) if not Path(p.replace("\\", "/")).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} test crops missing. Example: {missing[0]}. Restore data/cropped first."
        )

    dataset = CameraTrapDataset(str(test_csv), transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2 if DEVICE == "cuda" else 0,
        pin_memory=(DEVICE == "cuda"),
    )

    checkpoint = load_checkpoint(model_path)
    saved_classes = checkpoint.get("class_names")
    if saved_classes is not None and list(saved_classes) != list(CLASS_NAMES):
        raise ValueError(f"Checkpoint classes {saved_classes} do not match {CLASS_NAMES}.")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("=" * 64)
    print(f"ACTIVE LEARNING EVALUATION — ROUND {round_number} — {args.strategy.upper()}")
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_path}")
    print(f"Test samples: {len(dataset)}")

    truth: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=(DEVICE == "cuda"))
            logits = model(images)
            predictions.extend(logits.argmax(1).cpu().tolist())
            truth.extend(labels.tolist())

    report = classification_report(
        truth,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(results_dir / "classification_report.csv")

    per_species = report_df.loc[CLASS_NAMES, ["precision", "recall", "f1-score", "support"]].copy()
    per_species.to_csv(results_dir / "per_species_metrics.csv")
    per_species[["recall"]].to_csv(results_dir / "per_species_recall.csv")

    cm = confusion_matrix(truth, predictions, labels=list(range(len(CLASS_NAMES))))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(results_dir / "confusion_matrix.csv")

    summary = {
        "strategy": args.strategy,
        "round": round_number,
        "label_budget": int(checkpoint.get("labeled_samples", -1)),
        "test_samples": len(dataset),
        "test_accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "mountain_lion_recall": float(report["mountain_lion"]["recall"]),
        "best_val_accuracy": float(checkpoint.get("best_val_accuracy", float("nan"))),
    }
    pd.DataFrame([summary]).to_csv(results_dir / "summary.csv", index=False)

    print("\nCLASSIFICATION REPORT")
    print(report_df[["precision", "recall", "f1-score", "support"]])
    print("\nEVALUATION SUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
