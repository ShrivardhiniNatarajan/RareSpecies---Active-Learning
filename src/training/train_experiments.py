"""Train experimental models for RareCam (Phase 3).
Run from the repository root, for example:
  python -m src.training.train_experiments --strategy random --round 3 --experiment C
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import f1_score

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

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_class_weights(train_csv: Path) -> torch.Tensor:
    """Computes inverse-square-root class weights from training frequencies."""
    df = pd.read_csv(train_csv)
    # We must match the order in CLASS_NAMES
    counts = df['species'].value_counts()
    weights = []
    for cls in CLASS_NAMES:
        count = counts.get(cls, 0)
        # Add a small epsilon to avoid div by zero if a class is entirely missing
        weights.append(1.0 / np.sqrt(count + 1e-5))
    
    weights = torch.tensor(weights, dtype=torch.float32)
    # Normalize weights so the mean is 1.0 (to preserve learning rate scale roughly)
    weights = weights / weights.mean()
    return weights

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--round", required=True, type=int, choices=[1, 2, 3])
    parser.add_argument("--experiment", required=True, choices=["A", "B", "C"], 
                        help="A: Control, B: Head warm-up + finetune, C: Class-balanced")
    args = parser.parse_args()
    round_number = args.round

    set_seed(SEED)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_csv = Path("data/active_learning") / args.strategy / f"labeled_round_{round_number}.csv"
    val_csv = Path("data/splits/val.csv")
    model_dir = Path("models/experiments")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"exp{args.experiment}_{args.strategy}_round_{round_number}.pth"

    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

    print("=" * 64)
    print(f"EXPERIMENT {args.experiment} ?" {args.strategy.upper()} ROUND {round_number}")
    print("=" * 64)
    
    train_dataset = CameraTrapDataset(str(train_csv), transform=train_transform)
    val_dataset = CameraTrapDataset(str(val_csv), transform=eval_transform)
    print(f"Training samples: {len(train_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2 if DEVICE == "cuda" else 0, pin_memory=(DEVICE == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2 if DEVICE == "cuda" else 0, pin_memory=(DEVICE == "cuda")
    )

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.to(DEVICE)

    # ----------------------------------------------------
    # Experiment Configs
    # ----------------------------------------------------
    if args.experiment == "C":
        print("Using Class-Balanced Inverse-Square-Root Weights...")
        class_weights = get_class_weights(train_csv).to(DEVICE)
        print(f"Weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    if args.experiment == "B":
        print("Experiment B: Freezing backbone for 3 warm-up epochs...")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(model.fc.parameters(), lr=1e-3)
        # Total epochs: 3 warmup + 10 finetune = 13
        total_epochs = 13 
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        total_epochs = EPOCHS

    best_val_accuracy = -1.0
    best_val_macro = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch + 1}/{total_epochs}")
        
        # Unfreeze for Exp B after epoch 3
        if args.experiment == "B" and epoch == 3:
            print("Unfreezing backbone for fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW([
                {'params': model.conv1.parameters(), 'lr': 1e-5},
                {'params': model.bn1.parameters(), 'lr': 1e-5},
                {'params': model.layer1.parameters(), 'lr': 1e-5},
                {'params': model.layer2.parameters(), 'lr': 1e-5},
                {'params': model.layer3.parameters(), 'lr': 5e-5},
                {'params': model.layer4.parameters(), 'lr': 1e-4},
                {'params': model.fc.parameters(), 'lr': 1e-4}
            ])

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
        all_preds = []
        all_labels = []
        
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=(DEVICE == "cuda"))
                labels = labels.to(DEVICE, non_blocking=(DEVICE == "cuda"))
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * images.size(0)
                preds = logits.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss_sum / max(val_total, 1)
        val_accuracy = val_correct / max(val_total, 1)
        val_macro = f1_score(all_labels, all_preds, average='macro')
        
        print(f"Train loss: {train_loss:.4f} | train accuracy: {train_accuracy:.4f}")
        print(f"Val loss:   {val_loss:.4f} | val accuracy:   {val_accuracy:.4f} | val macro-F1: {val_macro:.4f}")

        # Save checkpoint if accuracy improves (or if it's the first epoch)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_macro = val_macro
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "class_names": CLASS_NAMES,
                    "strategy": args.strategy,
                    "round": round_number,
                    "experiment": args.experiment,
                    "labeled_samples": len(train_dataset),
                    "best_val_accuracy": best_val_accuracy,
                    "best_val_macro": best_val_macro,
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
