"""Evaluate experimental models for RareCam.
Run from the repository root, for example:
  python -m src.training.evaluate_experiments --strategy random --round 3 --experiment B
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from src.training.dataset import (
    CLASS_NAMES,
    CameraTrapDataset,
    eval_transform,
)

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
    parser.add_argument("--experiment", required=True, choices=["A", "B", "C"])
    args = parser.parse_args()
    round_number = args.round

    test_csv = Path("data/splits/test.csv")
    model_path = Path("models/experiments") / f"exp{args.experiment}_{args.strategy}_round_{round_number}.pth"
    out_dir = Path(f"results/experiments/exp{args.experiment}/{args.strategy}/round_{round_number}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    test_dataset = CameraTrapDataset(str(test_csv), transform=eval_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2 if DEVICE == "cuda" else 0, pin_memory=(DEVICE == "cuda")
    )

    checkpoint = load_checkpoint(model_path)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    print(f"Evaluating Experiment {args.experiment} - {args.strategy} - Round {round_number} on {DEVICE}...")
    start_time = time.time()
    
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=(DEVICE == "cuda"))
            labels = labels.to(DEVICE, non_blocking=(DEVICE == "cuda"))
            logits = model(images)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    inference_time = time.time() - start_time
    print(f"Inference time for {len(test_dataset)} samples: {inference_time:.2f}s")

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(out_dir / "classification_report.csv", index=True)

    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(out_dir / "confusion_matrix.csv", index=True)

    mt_lion_idx = CLASS_NAMES.index("mountain_lion")
    mt_lion_recall = report["mountain_lion"]["recall"]
    
    summary = pd.DataFrame([{
        "strategy": args.strategy,
        "round": round_number,
        "experiment": args.experiment,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mountain_lion_recall": mt_lion_recall,
        "inference_time_seconds": inference_time,
        "model_parameters": sum(p.numel() for p in model.parameters())
    }])
    summary.to_csv(out_dir / "summary.csv", index=False)
    
    print("\nResults:")
    print(summary.to_string(index=False))
    print(f"Saved results to {out_dir}")

if __name__ == "__main__":
    main()
