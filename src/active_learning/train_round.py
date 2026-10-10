"""Train a strategy-specific active-learning model for Round 1, 2, or 3.

Run from the repository root, for example:
  python -m src.active_learning.train_round --strategy random --round 3
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from src.training.dataset import (
    CLASS_NAMES,
    CameraTrapDataset,
    eval_transform,
    train_transform,
)


STRATEGIES = ("random", "uncertainty", "rarity_weighted")
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--round", required=True, type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    round_number = args.round

    train_csv = Path("data/active_learning") / args.strategy / f"labeled_round_{round_number}.csv"
    val_csv = Path("data/splits/val.csv")
    model_dir = Path("models/active_learning")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{args.strategy}_round_{round_number}.pth"

    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

    print("=" * 64)
    print(f"ACTIVE LEARNING TRAINING — ROUND {round_number} — {args.strategy.upper()}")
    print("=" * 64)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CPU training is likely to be much slower. Use a GPU runtime for all three models.")
    print(f"Training CSV: {train_csv}")

    train_dataset = CameraTrapDataset(str(train_csv), transform=train_transform)
    val_dataset = CameraTrapDataset(str(val_csv), transform=eval_transform)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    if round_number == 3 and len(train_dataset) != 1394:
        raise ValueError(f"Round 3 should have 1394 labeled crops; found {len(train_dataset)}.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2 if DEVICE == "cuda" else 0,
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2 if DEVICE == "cuda" else 0,
        pin_memory=(DEVICE == "cuda"),
    )

    # To preserve the Rounds 1–2 protocol, each budget is retrained from the same
    # ImageNet-pretrained initialization. Round-2 model weights are used for
    # selection only, not as Round-3 training initialization.
    print("\nLoading ImageNet-pretrained ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_val_accuracy = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=(DEVICE == "cuda"))
            labels = labels.to(DEVICE, non_blocking=(DEVICE == "cuda"))
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * images.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        train_accuracy = train_correct / max(train_total, 1)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=(DEVICE == "cuda"))
                labels = labels.to(DEVICE, non_blocking=(DEVICE == "cuda"))
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * images.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_accuracy = val_correct / max(val_total, 1)
        print(f"Train loss: {train_loss:.4f} | train accuracy: {train_accuracy:.4f}")
        print(f"Val loss:   {val_loss:.4f} | val accuracy:   {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "class_names": CLASS_NAMES,
                    "strategy": args.strategy,
                    "round": round_number,
                    "labeled_samples": len(train_dataset),
                    "best_val_accuracy": best_val_accuracy,
                    "seed": SEED,
                },
                model_path,
            )
            print(f"Saved best checkpoint: {model_path}")

    print("\nTRAINING COMPLETE")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Checkpoint: {model_path}")


if __name__ == "__main__":
    main()
