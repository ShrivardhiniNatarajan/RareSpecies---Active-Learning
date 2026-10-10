# RareCam: Rare-Species-Focused Active Learning with Detection-Aligned Attention Consistency

RareCam is an end-to-end camera-trap pipeline built to investigate how active learning strategies impact the recall of rare species. Automated wildlife monitoring pipelines often prioritize overall accuracy, which is naturally dominated by abundant species. RareCam evaluates random, uncertainty-based, and rarity-aware acquisition strategies on a highly imbalanced dataset to measure true performance on conservation-priority species.

## Research Problem and Motivation
Camera traps generate millions of images per survey. While deep learning greatly reduces manual review effort, active learning is required to select the most informative images to label. However, standard uncertainty sampling optimizes for overall accuracy and can leave rare species with poor recall. RareCam empirically investigates whether rarity-weighted active learning can outperform standard uncertainty sampling for rare species recognition.

## Dataset and Sequence-Aware Splitting
We use a six-class subset of the **Caltech Camera Traps** dataset (4,895 images):
- Opossum (1,300), Coyote (1,200), Deer (900), Raccoon (700), Bobcat (650), Mountain Lion (145).

To prevent data leakage, the dataset is split using a **sequence-aware** algorithm. Images from the same burst/event (`seq_id`) are never split across train, validation, and test sets, ensuring the model generalizes to new events rather than memorizing backgrounds.

## Pipeline Architecture
1. **Detection (MegaDetector V6):** Detects and crops animals, removing empty frames.
2. **Classification (ResNet-50):** A transfer-learning classifier trained on the padded crops.
3. **Active Learning Loop:** Simulates a human-in-the-loop by selecting 150 unlabeled crops per round to be added to the training set.
4. **Explainability (DAAC):** A novel attention metric to support human review.

### Detection-Aligned Attention Consistency (DAAC)
DAAC measures the intersection between the ResNet-50's Grad-CAM heatmap ($R$) and the MegaDetector bounding box ($B$):
$DAAC = \frac{|R \cap B|}{|R|}$
It uses the free detector box as the ground-truth attention mask, flagging predictions where the model is looking strictly at the background context rather than the animal.

## Active-Learning Strategies
- **Random:** Uniform random sampling from the unlabeled pool.
- **Uncertainty:** Highest predictive entropy.
- **Rarity-Aware:** Entropy weighted by the inverse of the predicted class's frequency in the labeled set.

## Experimental Results
A full-data baseline achieved **82.06% accuracy** but only **74.29% mountain-lion recall**, confirming the motivating problem. 
After multiple rounds of active learning, we evaluated three configurations:
1. Standard fine-tuning
2. Transfer Learning with Head Warmup (Experiment B)
3. Class-balanced Inverse-Square-Root Weights (Experiment C)

**Experiment B (Backbone freezing for 3 epochs followed by fine-tuning)** achieved the highest Validation Macro-F1. However, uncertainty sampling generally maintained higher mountain-lion recall than rarity-aware sampling, demonstrating that simple rarity-weighting is not a panacea. 
DAAC analysis revealed no correlation with misclassification (AUROC 0.426), proving that camera-trap classifiers legitimately rely on habitat context immediately outside the bounding box.

## Setup and Installation
1. Clone the repository and install dependencies.
```bash
git clone https://github.com/ShrivardhiniNatarajan/RareSpecies---Active-Learning.git
cd RareSpecies---Active-Learning
python -m venv camtrap-env
# Activate virtual environment (Windows)
.\camtrap-env\Scripts\Activate.ps1
# Activate (Linux/Mac)
source camtrap-env/bin/activate
pip install -r requirements.txt
```

2. Download the data from Google Drive (Requires Colab):
```bash
python scripts/restore_round3_assets.py
```

## How to Reproduce Training and Evaluation
Run the unified experiment script (e.g., Experiment B):
```bash
python -m src.training.train_experiments --strategy uncertainty --round 3 --experiment B
python -m src.training.evaluate_experiments --strategy uncertainty --round 3 --experiment B
```

To generate the final figures and visualizations:
```bash
python scripts/generate_final_plots.py
```
Generated artifacts (CSV results, plots, and confusion matrices) are saved to `results/experiments/` and `results/final/`.

## Streamlit Demo
A local Streamlit app is provided for interactive inference, DAAC visualization, and review-queue sorting.
```bash
streamlit run demo/app.py
```
The app strictly sorts "Needs Review" items (low confidence or low DAAC) to the top of the queue.

## Known Limitations
- Simulated active learning relies on a pre-labeled pool rather than timing real human annotation.
- Rarity is defined relative to the dataset frequency, not biological IUCN rarity.
- Only two rounds were fully evaluated in the initial paper draft due to compute constraints.