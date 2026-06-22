# Digital Image Processing Final Project Submission Package
1. Project Overview

This repository contains the full scientific submission package for the automated waste classification project on the benchmark TrashNet dataset. It comparatively evaluates three major epochs of Computer Vision:

Simple Baseline (DIP): Otsu binarization coupled with disk morphological closing filters and geometric descriptor classification.

Classical Machine Learning (ML): Histogram of Oriented Gradients (HOG) combined with a kernelized Support Vector Machine (SVM) with an RBF kernel.

Modern Deep Learning (DL): MobileNetV2 and ResNet50 pre-trained ImageNet architectures trained through progressive multi-stage transfer learning.

Our final report demonstrating a state-of-the-art accuracy of 95.4% can be found in 20291273_Final.pdf (prepared strictly using the IEEE template).

## Requirements and Dependencies
To install the required Python libraries, ensure you have Python 3.10+ installed and run:
```bash
pip install -r requirements.txt
```


## Required Packages (requirements.txt):
```bash
numpy>=1.22.0
opencv-python>=4.5.5
scikit-image>=0.19.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
matplotlib>=3.5.0
```

### Training and Evaluation Scripts
The script train_eval.py is configured with an automated mock data generator to ensure that the code compiles and runs successfully on any standard workstation immediately upon execution.

```bash
python train_eval.py
```
Pipeline Steps Executed in the Code:

Method 1 (Otsu Baseline): Converts images to grayscale, applies Gaussian blur, computes Otsu's optimal threshold, runs ellipse-based morphological closing, and estimates waste category using compactness metrics ($C = \frac{4\pi A}{P^2}$).

Method 2 (HOG + SVM): Extracts 9-bin local Histograms of Oriented Gradients over $16 \times 16$ pixel grids, normalizes across blocks, and classifies features using an RBF-kernel SVM.

Method 3 (Deep Transfer Learning): Loads a MobileNetV2 architecture with ImageNet weights, freezes the base to train a custom dense classifier head, and then unfreezes for progressive fine-tuning with a very low learning rate ($\eta = 10^{-5}$).

### Hardware and Experiment Logs
Device CPU: Intel Core i7-11800H @ 2.30GHz, 16GB RAM.

Accelerator GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM).

OS: Windows 11 / Linux Ubuntu 22.04 LTS.

Hyperparameter Specifications:

Otsu Structuring Element: Disk-shape, $r=3$.

SVM Hyperparameters: $C=10.0$, $\gamma = \text{"scale"}$, Radial Basis Function (RBF) Kernel.

Deep Learning Stage 1 (Feature Extraction): Adam, $\eta=10^{-4}$, 10 Epochs.

Deep Learning Stage 2 (Fine-Tuning): Adam, $\eta=10^{-5}$, 20 Epochs.
- add specific results (and parameters used) that were achieved after the CS 229 project deadline
- add saving of confusion matrix data and creation of graphic to `plot.lua`
- rewrite the data preprocessing to only reprocess new images if the dimensions have not changed
