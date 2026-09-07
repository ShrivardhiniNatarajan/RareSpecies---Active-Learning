# RareCam

An active-learning system for wildlife camera-trap image classification focusing on underrepresented species.

## Problem & Motivation
Large camera-trap datasets require significant human effort to label. While object detection (like MegaDetector) can automate finding animals, species classification often struggles with class imbalance (e.g., thousands of deer, but very few mountain lions). 

## Objectives
1. Automatically detect animals using MegaDetector V6.
2. Classify species using a ResNet-50 transfer-learning baseline.
3. Use active learning to prioritize informative cases for human review.
4. Enhance explainability using Detection-Aligned Attention Consistency (DAAC).

## Dataset & Architecture
- **Dataset**: Caltech Camera Traps (LILA BC) - subset of 4,895 images (6,558 crops).
- **Target Classes**: opossum, coyote, deer, raccoon, bobcat, mountain_lion (primary rare target).
- **Detector**: MegaDetector V6 (YOLOv10-c).
- **Classifier**: ResNet-50 (ImageNet transfer learning).
- **Data Split**: Sequence-aware splitting ensures no sequence leakage between train, validation, and test.

## DAAC Contribution
Detection-Aligned Attention Consistency (DAAC) measures how well the classifier's visual attention (Grad-CAM) aligns with the bounding box automatically provided by MegaDetector. A high DAAC score implies the model is looking at the animal; a low score implies it is relying on background features. This provides an automated explainability metric without requiring costly human-drawn attention masks.

## Experiments & Results
Active learning simulates a human labeling budget. We compared Random Sampling, Uncertainty Sampling (Predictive Entropy), and Rarity-Aware Sampling.

**Full-Data Baseline**:
- Test accuracy: 82.06%
- Mountain-lion recall: 74.29%

**Active Learning (At 1,244 labels)**:
- Rarity-aware sampling achieves the best overall accuracy (77.94%) and macro-F1 (74.52%).
- Uncertainty sampling achieves the best mountain-lion recall (51.43%).
- *Honest Limitation*: Rarity-aware sampling does **not** currently beat uncertainty sampling on the specific mountain-lion target metric. 

## Git Workflow
We use feature branches and conventional commits (`feat:`, `fix:`, `docs:`) to maintain a clean history. Large artifacts are ignored via `.gitignore` and stored externally.

## How to Run the Demo

Activate the environment and run Streamlit:
```powershell
camtrap-env\Scripts\activate
streamlit run demo\app.py
```
The demo includes full batch processing and priority queueing using Uncertainty, Rarity, and DAAC scoring.

## Future Work
- Complete Round 3 of active learning (requires GPU/Kaggle).
- Evaluate a combined metric: Uncertainty + Rarity + DAAC to select review images.