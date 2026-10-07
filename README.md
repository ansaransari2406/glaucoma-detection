# 👁️ Glaucoma Detection Using Deep Learning and Ensemble Learning

A deep learning-based research project for classifying retinal fundus images into **Glaucoma** and **Normal** categories.

The system uses pretrained CNN models for feature extraction, XGBoost for classification, and a Logistic Regression ensemble model for the final prediction. A Streamlit web application provides an easy-to-use interface for uploading and analyzing fundus images.

## 📌 Project Overview

Glaucoma is an eye disease that can lead to permanent vision loss if it is not detected and managed in time.

This project explores the use of deep learning and ensemble machine learning techniques for automated glaucoma image classification.

The system follows this workflow:

```text
Fundus Image
     ↓
Image Preprocessing
     ↓
CNN Feature Extraction
     ↓
ResNet50 ────────┐
DenseNet121 ─────┤
VGG16 ───────────┤
InceptionV3 ─────┘
     ↓
XGBoost Models
     ↓
Logistic Regression Ensemble
     ↓
Glaucoma / Normal