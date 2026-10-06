# Glaucoma Detection Using Deep Learning-Based Pretrained CNN Models and Ensemble Learning

## 📌 Project Overview

Glaucoma is a serious eye disease that can lead to permanent vision loss if it is not detected early.

This project uses Deep Learning and Machine Learning techniques to classify retinal fundus images into:

- Normal
- Glaucoma

The system extracts deep features using pretrained Convolutional Neural Network (CNN) models and uses XGBoost classifiers followed by an ensemble learning model for final prediction.

## 🧠 Models Used

### Pretrained CNN Models

- ResNet50
- DenseNet121
- VGG16
- InceptionV3

The pretrained CNN models are used as feature extractors with ImageNet weights.

### Machine Learning

- XGBoost
- Logistic Regression for ensemble learning

## 🔬 Methodology

The overall workflow is:

```text
Fundus Image
     ↓
Image Preprocessing
     ↓
Pretrained CNN Models
     ↓
Deep Feature Extraction
     ↓
XGBoost Classifiers
     ↓
Prediction Probabilities
     ↓
Ensemble Learning
     ↓
Final Prediction
     ↓
Normal / Glaucoma