from pathlib import Path
import tempfile
import base64
from io import BytesIO

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd

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
# PAGE & CSS
# ============================================================

st.set_page_config(
    page_title="RareCam",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Animated gradient background */
@keyframes gradientBG {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}
.stApp {
    background: linear-gradient(-45deg, #0d1214, #12181B, #172722, #12181B);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}
.stApp > header {
    background-color: transparent;
}
/* Hide Streamlit default marks */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stHeader"] {visibility: hidden;}

/* Custom Fonts & Hierarchy */
h1, h2, h3, h4, h5, h6, p, span {
    font-family: "Inter", "Segoe UI", sans-serif;
    color: #e2e8f0;
}

h1 {
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    color: #10b981 !important;
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
h2 {
    font-weight: 700 !important;
    color: #f1f5f9 !important;
}
h3 {
    font-weight: 500 !important;
    color: #94a3b8 !important;
    margin-top: 0 !important;
    padding-top: 0.5rem !important;
}

/* Custom stat cards */
.stat-card {
    background-color: #1A2125;
    border: 1px solid #283339;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.stat-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
    font-weight: 600;
}
.stat-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f8fafc;
    text-transform: capitalize;
}
.confidence-bar {
    height: 8px;
    border-radius: 4px;
    background-color: #283339;
    margin-top: 0.75rem;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease-in-out;
}
.confidence-high { background-color: #10b981; } /* Emerald */
.confidence-medium { background-color: #f59e0b; }
.confidence-low { background-color: #ef4444; }

/* Badges */
.badge-container {
    margin-top: 1.5rem;
    background-color: #1A2125;
    border: 1px solid #283339;
    border-radius: 12px;
    padding: 1.2rem;
}
.badge {
    padding: 0.5rem 1.25rem;
    border-radius: 9999px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.95rem;
}
.badge-success {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.4);
}
.badge-warning {
    background-color: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.4);
}
.help-text {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 0.5rem;
}

/* Image styling */
div[data-testid="stImage"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    border: 1px solid #283339;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0d1214 !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# STATE MANAGEMENT
# ============================================================
if "processed_results" not in st.session_state:
    st.session_state.processed_results = []


# ============================================================
# BACKEND: LOAD MEGADETECTOR
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
# BACKEND: LOAD CLASSIFIER
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
# BACKEND: IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.resize(
        (224, 224)
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    image = (
        image - mean
    ) / std

    tensor = torch.from_numpy(
        image
    ).permute(
        2, 0, 1
    ).unsqueeze(0)

    return tensor.float()


# ============================================================
# BACKEND: PROCESSOR FUNCTION
# ============================================================
# This function wraps the original processing logic EXACTLY as it was written.

def process_single_image(uploaded_file, detector, classifier):
    image = Image.open(
        uploaded_file
    ).convert("RGB")
    
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

    image_array = np.array(
        image
    )

    processed_animals = []

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
        
        needs_review = predicted_confidence < REVIEW_THRESHOLD
        
        # Create base64 thumbnail for dataframe display
        buffered = BytesIO()
        crop_image_resized = crop_image.copy()
        crop_image_resized.thumbnail((150, 150))
        crop_image_resized.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        thumbnail_url = f"data:image/jpeg;base64,{img_str}"

        processed_animals.append({
            "detection_number": detection_number,
            "crop_image": crop_image,  # Keep full PIL image for detail view
            "thumbnail": thumbnail_url,
            "filename": uploaded_file.name,
            "predicted_species": predicted_species,
            "predicted_confidence": predicted_confidence,
            "confidence": confidence,
            "entropy": entropy,
            "needs_review": needs_review
        })
        
    return {
        "filename": uploaded_file.name,
        "full_image": image,
        "animals": processed_animals
    }


# ============================================================
# UI: SHARED COMPONENTS
# ============================================================

def render_animal_detail(animal):
    """Renders the detailed stat cards for a single animal detection."""
    st.markdown(f"## Animal {animal['detection_number']} ({animal['filename']})")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.image(
            animal["crop_image"],
            use_container_width=True
        )

    with col2:
        scol1, scol2 = st.columns(2)
        
        with scol1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Species</div>
                <div class="stat-value">{animal["predicted_species"].replace('_', ' ')}</div>
            </div>
            """, unsafe_allow_html=True)

        with scol2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Uncertainty Score</div>
                <div class="stat-value">{animal["entropy"]:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        ccol1, ccol2 = st.columns(2)
        
        with ccol1:
            pc = animal["predicted_confidence"]
            conf_class = "confidence-high" if pc >= 0.8 else ("confidence-medium" if pc >= REVIEW_THRESHOLD else "confidence-low")
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Classifier Confidence</div>
                <div class="stat-value">{pc:.1%}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill {conf_class}" style="width: {pc * 100}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with ccol2:
            dc = animal["confidence"]
            det_class = "confidence-high" if dc >= 0.8 else ("confidence-medium" if dc >= 0.5 else "confidence-low")
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Detector Confidence</div>
                <div class="stat-value">{dc:.1%}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill {det_class}" style="width: {dc * 100}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Human review decision
        if animal["needs_review"]:
            st.markdown("""
            <div class="badge-container">
                <div class="badge badge-warning">
                    <span>⚠</span> Needs Human Review
                </div>
                <div class="help-text">The classifier is not confident enough in this prediction.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="badge-container">
                <div class="badge badge-success">
                    <span>✓</span> Auto-Processed
                </div>
                <div class="help-text">Prediction confident enough for automatic processing.</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)


# ============================================================
# UI: SIDEBAR NAVIGATION
# ============================================================

logo_path = Path(__file__).parent / "logo.jpg"
if logo_path.exists():
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    img_src = f"data:image/jpeg;base64,{logo_base64}"
else:
    img_src = "https://images.unsplash.com/photo-1588693892782-eb061482855f?auto=format&fit=crop&w=300&q=80"

st.sidebar.markdown(f"""
<div style="text-align: center; margin-bottom: 1.5rem; margin-top: 1rem;">
    <img src="{img_src}" style="width: 140px; height: 140px; border-radius: 50%; object-fit: cover; border: 4px solid #10b981; margin-bottom: 1rem; box-shadow: 0 4px 10px rgba(0,0,0,0.4);">
    <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #10b981 !important; letter-spacing: -0.02em;">RareCam</h1>
    <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 0.5rem; font-weight: 500; line-height: 1.4;">
        Active Learning for Camera-Trap<br>Wildlife Monitoring
    </p>
</div>
<hr style="border-top: 1px solid #283339; margin-bottom: 1.5rem;">
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Upload & Analyze", "Batch Results", "About the Model"],
    label_visibility="collapsed"
)


# ============================================================
# PAGE: UPLOAD & ANALYZE
# ============================================================
if page == "Upload & Analyze":
    st.markdown("## Upload & Analyze")
    st.write(
        "Upload one or more camera-trap images to detect animals, "
        "classify species, and identify predictions requiring human review."
    )
    
    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Analyze Images", type="primary"):
            detector = load_detector()
            classifier = load_classifier()
            
            # Clear previous batch results if a new batch is submitted
            st.session_state.processed_results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name} ({i+1}/{len(uploaded_files)})...")
                
                result_data = process_single_image(file, detector, classifier)
                
                if not result_data["animals"]:
                    st.warning(f"No animal detected in {file.name}.")
                
                st.session_state.processed_results.append(result_data)
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Processing complete!")
            st.success(f"Successfully processed {len(uploaded_files)} images.")
            
            # Display results immediately
            for res in st.session_state.processed_results:
                for animal in res["animals"]:
                    render_animal_detail(animal)
                    st.divider()


# ============================================================
# PAGE: BATCH RESULTS
# ============================================================
elif page == "Batch Results":
    st.markdown("## Batch Results Summary")
    
    if not st.session_state.processed_results:
        st.info("No results yet. Please go to 'Upload & Analyze' to process some images.")
    else:
        # Collect all animals across all processed images
        all_animals = []
        for res in st.session_state.processed_results:
            all_animals.extend(res["animals"])
            
        if not all_animals:
            st.warning("No animals were detected in the processed batch.")
        else:
            # Sort with Needs Review at the top
            all_animals.sort(key=lambda x: not x["needs_review"])
            
            # Build DataFrame for display
            df_data = []
            for a in all_animals:
                df_data.append({
                    "Thumbnail": a["thumbnail"],
                    "Filename": a["filename"],
                    "Species": a["predicted_species"].replace('_', ' ').title(),
                    "Classifier Confidence": a["predicted_confidence"],
                    "Detector Confidence": a["confidence"],
                    "Uncertainty": round(a["entropy"], 3),
                    "Review Status": "🔴 Needs Review" if a["needs_review"] else "🟢 Auto-Processed"
                })
            
            df = pd.DataFrame(df_data)
            
            st.dataframe(
                df,
                column_config={
                    "Thumbnail": st.column_config.ImageColumn(
                        "Crop", help="Animal crop"
                    ),
                    "Classifier Confidence": st.column_config.ProgressColumn(
                        "Classifier Conf",
                        help="Confidence score of the classifier",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                    "Detector Confidence": st.column_config.ProgressColumn(
                        "Detector Conf",
                        help="Confidence score of the detector",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    )
                },
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            st.markdown("### Detailed View")
            # Expanders for detailed views
            for a in all_animals:
                status_icon = "🔴" if a["needs_review"] else "🟢"
                with st.expander(f"{status_icon} {a['filename']} - Animal {a['detection_number']} - {a['predicted_species'].replace('_', ' ').title()}"):
                    render_animal_detail(a)


# ============================================================
# PAGE: ABOUT THE MODEL
# ============================================================
elif page == "About the Model":
    st.markdown("## About the Model")
    st.write(
        "RareCam uses a dual-model pipeline to detect and classify wildlife in camera-trap images."
    )
    
    st.markdown("### MegaDetector")
    st.write("MegaDetector is an open-source object detection model specifically trained to find animals, humans, and vehicles in camera trap images. We use YOLOv10 to quickly filter out empty images and isolate crops of the detected subjects.")
    
    st.markdown("### Species Classifier")
    st.write("The classifier is a customized ResNet-50 architecture trained on a specialized subset of wildlife data to identify the following classes:")
    st.markdown("- Opossum\n- Coyote\n- Deer\n- Raccoon\n- Bobcat\n- Mountain Lion")
    
    st.markdown("### Active Learning")
    st.write(f"When the model's confidence falls below the Review Threshold ({REVIEW_THRESHOLD*100}%), the system flags the prediction. These uncertain images are meant to be routed to human experts for verification, which can then be fed back to retrain and improve the model over time.")