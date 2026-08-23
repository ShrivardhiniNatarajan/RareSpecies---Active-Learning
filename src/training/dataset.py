from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ============================================================
# CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = [
    "opossum",
    "coyote",
    "deer",
    "raccoon",
    "bobcat",
    "mountain_lion",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

INDEX_TO_CLASS = {
    index: name
    for name, index in CLASS_TO_INDEX.items()
}


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

IMAGE_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# DATASET CLASS
# ============================================================

class CameraTrapDataset(Dataset):

    def __init__(
        self,
        csv_file,
        transform=None
    ):

        self.csv_file = Path(csv_file)

        self.data = pd.read_csv(
            self.csv_file
        )

        self.transform = transform

        # ----------------------------------------------------
        # Check required columns
        # ----------------------------------------------------

        required_columns = [
            "crop_file",
            "species"
        ]

        for column in required_columns:

            if column not in self.data.columns:

                raise ValueError(
                    f"Missing required column: "
                    f"{column}"
                )

        # ----------------------------------------------------
        # Convert species names to integer labels
        # ----------------------------------------------------

        self.data["label"] = (
            self.data["species"]
            .map(CLASS_TO_INDEX)
        )

        # ----------------------------------------------------
        # Make sure every species is known
        # ----------------------------------------------------

        if self.data["label"].isna().any():

            unknown_species = (
                self.data.loc[
                    self.data["label"].isna(),
                    "species"
                ].unique()
            )

            raise ValueError(
                f"Unknown species found: "
                f"{unknown_species}"
            )

        self.data["label"] = (
            self.data["label"]
            .astype(int)
        )

    # ========================================================
    # Number of samples
    # ========================================================

    def __len__(self):

        return len(self.data)

    # ========================================================
    # Get one sample
    # ========================================================

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = Path(
            row["crop_file"]
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: "
                f"{image_path}"
            )

        # Open image
        image = Image.open(
            image_path
        ).convert("RGB")

        # Apply transformation
        if self.transform is not None:
            image = self.transform(image)

        # Get integer label
        label = torch.tensor(
            row["label"],
            dtype=torch.long
        )

        return image, label


# ============================================================
# TEST THE DATASET
# ============================================================

if __name__ == "__main__":

    train_dataset = CameraTrapDataset(
        "data/splits/train.csv",
        transform=train_transform
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    image, label = train_dataset[0]

    print(
        "Image shape:",
        image.shape
    )

    print(
        "Label index:",
        label.item()
    )

    print(
        "Label name:",
        INDEX_TO_CLASS[label.item()]
    )