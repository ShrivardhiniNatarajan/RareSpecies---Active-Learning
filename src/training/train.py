import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import models

from dataset import (
    CameraTrapDataset,
    train_transform,
    eval_transform,
    CLASS_NAMES,
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CSV = "data/splits/train.csv"
VAL_CSV = "data/splits/val.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR /
    "baseline_resnet50.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

NUM_CLASSES = len(CLASS_NAMES)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# INFORMATION
# ============================================================

print("=" * 60)
print("RESNET-50 BASELINE TRAINING")
print("=" * 60)

print("Device:", DEVICE)

if DEVICE == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print(
    "Number of classes:",
    NUM_CLASSES
)

print(
    "Classes:",
    CLASS_NAMES
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = CameraTrapDataset(
    TRAIN_CSV,
    transform=train_transform
)

val_dataset = CameraTrapDataset(
    VAL_CSV,
    transform=eval_transform
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=(DEVICE == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=(DEVICE == "cuda")
)


# ============================================================
# LOAD PRETRAINED RESNET-50
# ============================================================

print("\nLoading pretrained ResNet-50...")

weights = (
    models.ResNet50_Weights.DEFAULT
)

model = models.resnet50(
    weights=weights
)


# ============================================================
# REPLACE FINAL CLASSIFIER
# ============================================================

num_features = (
    model.fc.in_features
)

model.fc = nn.Linear(
    num_features,
    NUM_CLASSES
)


# Move model to GPU/CPU
model = model.to(DEVICE)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING TRACKING
# ============================================================

best_val_accuracy = 0.0

best_model_state = copy.deepcopy(
    model.state_dict()
)


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    print(
        f"\nEpoch "
        f"{epoch + 1}/{EPOCHS}"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        train_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(dim=1)
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += (
            labels.size(0)
        )

    epoch_train_loss = (
        train_loss /
        train_total
    )

    epoch_train_accuracy = (
        train_correct /
        train_total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs.argmax(dim=1)
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += (
                labels.size(0)
            )

    epoch_val_loss = (
        val_loss /
        val_total
    )

    epoch_val_accuracy = (
        val_correct /
        val_total
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Train Loss: "
        f"{epoch_train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{epoch_train_accuracy:.4f}"
    )

    print(
        f"Val Loss: "
        f"{epoch_val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{epoch_val_accuracy:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if (
        epoch_val_accuracy
        > best_val_accuracy
    ):

        best_val_accuracy = (
            epoch_val_accuracy
        )

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict":
                    best_model_state,

                "class_names":
                    CLASS_NAMES,

                "image_size":
                    IMAGE_SIZE,

                "best_val_accuracy":
                    best_val_accuracy,
            },
            MODEL_PATH
        )

        print(
            "✓ Best model saved."
        )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Model saved to: "
    f"{MODEL_PATH}"
)