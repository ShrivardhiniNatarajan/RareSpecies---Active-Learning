from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from torch.utils.data import DataLoader
from torchvision import models

from dataset import (
    CameraTrapDataset,
    eval_transform,
    CLASS_NAMES,
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_CSV = "data/splits/test.csv"

MODEL_PATH = (
    "models/baseline_resnet50.pth"
)

RESULTS_DIR = Path(
    "results/baseline"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BATCH_SIZE = 32

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("=" * 60)
print("RESNET-50 BASELINE EVALUATION")
print("=" * 60)

print("Device:", DEVICE)

test_dataset = CameraTrapDataset(
    TEST_CSV,
    transform=eval_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=(DEVICE == "cuda")
)

print(
    "Test samples:",
    len(test_dataset)
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\nLoading ResNet-50...")

model = models.resnet50(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print(
    "Best validation accuracy:",
    checkpoint.get(
        "best_val_accuracy",
        "not stored"
    )
)


# ============================================================
# RUN TEST PREDICTIONS
# ============================================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        predictions = (
            outputs.argmax(dim=1)
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

print("\n============================================")
print("CLASSIFICATION REPORT")
print("============================================")

print(
    report_df[
        [
            "precision",
            "recall",
            "f1-score",
            "support"
        ]
    ]
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = (
    RESULTS_DIR /
    "classification_report.csv"
)

report_df.to_csv(
    report_path
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(
    figsize=(9, 8)
)

display.plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Blues",
    colorbar=False
)

ax.set_title(
    "ResNet-50 Test Confusion Matrix"
)

fig.tight_layout()

cm_path = (
    RESULTS_DIR /
    "confusion_matrix.png"
)

fig.savefig(
    cm_path,
    dpi=300
)

plt.show()

plt.close(fig)


# ============================================================
# IMPORTANT PROJECT METRIC:
# PER-SPECIES RECALL
# ============================================================

recall_df = report_df.loc[
    CLASS_NAMES,
    ["recall"]
].copy()

recall_df = (
    recall_df
    .sort_values(
        "recall",
        ascending=False
    )
)

print("\n============================================")
print("PER-SPECIES RECALL")
print("============================================")

print(recall_df)


recall_path = (
    RESULTS_DIR /
    "per_species_recall.csv"
)

recall_df.to_csv(
    recall_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n============================================")
print("EVALUATION COMPLETE")
print("============================================")

print(
    "Accuracy:",
    report["accuracy"]
)

print(
    "Macro F1:",
    report["macro avg"]["f1-score"]
)

print(
    "\nSaved:"
)

print(
    report_path
)

print(
    cm_path
)

print(
    recall_path
)