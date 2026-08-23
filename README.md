# RareSpecies-ActiveLearning

An explainable, label-efficient active learning framework for
camera-trap wildlife monitoring.

## Project Objective

This project investigates whether an active learning strategy
can allocate a limited human labeling budget more effectively
toward underrepresented or conservation-priority wildlife
occurrences, rather than optimizing only overall classification
accuracy.

## Dataset

Caltech Camera Traps dataset obtained through LILA BC.

The original dataset contains:

- 243,100 images
- 245,118 annotations
- 22 categories

## Current Pipeline

```text
Caltech Camera-Trap Metadata
          ↓
Dataset Analysis
          ↓
Experimental Subset
          ↓
MegaDetector
          ↓
Animal Cropping
          ↓
ResNet-50 Classification
          ↓
Per-Species Recall
          ↓
Active Learning
          ↓
Random vs Uncertainty vs Proposed Strategy
          ↓
Grad-CAM Explainability