from pathlib import Path
import tempfile

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image
from torchvision import models

from PytorchWildlife.models import detection as pw_detection


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "opossum",
    "coyote",
    "deer",
    "raccoon",
    "bobcat",
    "mountain_lion",
]

MODEL_PATH = Path(
    "models/baseline_resnet50.pth"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

REVIEW_THRESHOLD = 0.60


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="RareCam",
    page_icon="📷",
    layout="wide"
)

st.title("RareCam")
st.subheader(
    "Active Learning for Camera-Trap Wildlife Monitoring"
)

st.write(
    "Upload a camera-trap image to detect the animal, "
    "classify its species, and identify predictions "
    "that may require human review."
)


# ============================================================
# LOAD MEGADETECTOR
# ============================================================

@st.cache_resource
def load_detector():

    model = pw_detection.MegaDetectorV6(
        version="MDV6-yolov10-c",
        device=DEVICE,
        pretrained=True
    )

    return model


# ============================================================
# LOAD CLASSIFIER
# ============================================================

@st.cache_resource
def load_classifier():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model = models.resnet50(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.resize(
        (224, 224)
    )

    image = np.array(
        image
    ).astype(
        np.float32
    ) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406]
    )

    std = np.array(
        [0.229, 0.224, 0.225]
    )

    image = (
        image - mean
    ) / std

    image = torch.tensor(
        image
    ).permute(
        2, 0, 1
    ).unsqueeze(0)

    return image


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a camera-trap image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PROCESS
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    if st.button(
        "Analyze Image",
        type="primary"
    ):

        detector = load_detector()
        classifier = load_classifier()

        # ----------------------------------------------------
        # Save temporary image
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as temp_file:

            temp_path = temp_file.name

            image.save(
                temp_path
            )

        # ----------------------------------------------------
        # MegaDetector
        # ----------------------------------------------------

        with st.spinner(
            "Detecting animals..."
        ):

            result = (
                detector
                .single_image_detection(
                    temp_path
                )
            )

        detections = result["detections"]

        boxes = detections.xyxy
        confidences = detections.confidence
        class_ids = detections.class_id

        animal_indices = [
            i
            for i, class_id in enumerate(
                class_ids
            )
            if int(class_id) == 0
            and float(confidences[i])
            >= 0.20
        ]

        if not animal_indices:

            st.warning(
                "No animal detected."
            )

        else:

            st.success(
                f"{len(animal_indices)} "
                "animal detection(s) found."
            )

            image_array = np.array(
                image
            )

            # ------------------------------------------------
            # Process detections
            # ------------------------------------------------

            for detection_number, index in enumerate(
                animal_indices,
                start=1
            ):

                x1, y1, x2, y2 = map(
                    int,
                    boxes[index]
                )

                confidence = float(
                    confidences[index]
                )

                # --------------------------------------------
                # Clamp box
                # --------------------------------------------

                x1 = max(
                    0,
                    min(
                        x1,
                        image.width - 1
                    )
                )

                y1 = max(
                    0,
                    min(
                        y1,
                        image.height - 1
                    )
                )

                x2 = max(
                    0,
                    min(
                        x2,
                        image.width
                    )
                )

                y2 = max(
                    0,
                    min(
                        y2,
                        image.height
                    )
                )

                crop = image_array[
                    y1:y2,
                    x1:x2
                ]

                if crop.size == 0:
                    continue

                crop_image = Image.fromarray(
                    crop
                )

                # --------------------------------------------
                # Classify crop
                # --------------------------------------------

                tensor = preprocess_image(
                    crop_image
                ).to(DEVICE)

                with torch.no_grad():

                    logits = classifier(
                        tensor
                    )

                    probabilities = F.softmax(
                        logits,
                        dim=1
                    )[0]

                predicted_index = int(
                    probabilities.argmax()
                )

                predicted_species = (
                    CLASS_NAMES[
                        predicted_index
                    ]
                )

                predicted_confidence = float(
                    probabilities[
                        predicted_index
                    ]
                )

                # --------------------------------------------
                # Entropy
                # --------------------------------------------

                entropy = float(
                    -(
                        probabilities
                        * torch.log(
                            probabilities + 1e-12
                        )
                    ).sum()
                )

                # --------------------------------------------
                # Display
                # --------------------------------------------

                st.divider()

                st.write(
                    f"### Animal {detection_number}"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        crop_image,
                        caption="Detected animal",
                        use_container_width=True
                    )

                with col2:

                    st.metric(
                        "Species",
                        predicted_species
                    )

                    st.metric(
                        "Classifier confidence",
                        f"{predicted_confidence:.2%}"
                    )

                    st.metric(
                        "Detector confidence",
                        f"{confidence:.2%}"
                    )

                    st.metric(
                        "Uncertainty",
                        f"{entropy:.3f}"
                    )

                    # ----------------------------------------
                    # Human review decision
                    # ----------------------------------------

                    if (
                        predicted_confidence
                        < REVIEW_THRESHOLD
                    ):

                        st.error(
                            "⚠ Human review recommended"
                        )

                        st.write(
                            "The classifier is not "
                            "confident enough in this prediction."
                        )

                    else:

                        st.success(
                            "Prediction confident enough "
                            "for automatic processing."
                        )