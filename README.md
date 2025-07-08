# Fer2013EmotionCnn

# Emotion Detection Using Neural Networks

**Author:** Mohamed Emad

---

## 📄 Project Summary

A real-time facial emotion recognition system using Convolutional Neural Networks (CNNs). The system detects emotions from webcam video feed and classifies them into five categories:

> **Angry**, **Happy**, **Neutral**, **Sad**, **Surprise**

---

## 📌 Table of Contents

1. [Project Overview](#1-project-overview)  
2. [Tools and Technologies](#2-tools-and-technologies)  
3. [System Architecture](#3-system-architecture)  
4. [Real-Time Emotion Detection](#4-real-time-emotion-detection)  
5. [Model Training](#5-model-training)  
6. [Performance Evaluation](#6-performance-evaluation)  
7. [Version History](#7-version-history)

---

## 1. Project Overview

### 🎯 Purpose
To build a robust and accurate emotion detection system using a CNN-based model that classifies emotions from live webcam input.

### 🔍 Scope
- Real-time webcam-based emotion detection  
- Offline training with the FER2013 dataset  
- Modular system for both training and inference  

---

## 2. Tools and Technologies

- **Python** - Core language  
- **TensorFlow / Keras** - Deep learning framework  
- **OpenCV** - Image capture and preprocessing  
- **MediaPipe** - Face detection  
- **NumPy / Pandas** - Data handling  
- **Matplotlib / Seaborn** - Visualization  
- **Google Colab** - Model training environment  
- **Kaggle API** - Dataset download

---

## 3. System Architecture

### 🧠 Model Training
1. Load and filter data from `fer2013.csv`  
2. Apply preprocessing and data augmentation  
3. Build CNN model (4 Conv blocks + BatchNorm + Dropout)  
4. Train using callbacks: `EarlyStopping`, `ReduceLROnPlateau`  
5. Save best model to disk

### 📷 Real-Time Detection
1. Load trained model  
2. Use MediaPipe to detect faces  
3. Resize and normalize face to 48×48  
4. Predict emotion  
5. Overlay label and confidence on the frame  

---

## 4. Real-Time Emotion Detection

Implemented in `EmotionDetector` class:

- Face detection: MediaPipe  
- Preprocessing: grayscale → resize → normalize → reshape  
- Live prediction from webcam  
- Emotions supported:
  - Angry
  - Happy
  - Neutral
  - Sad
  - Surprise

---

## 5. Model Training

### ⚙️ Training Details
- Dataset: [FER2013 via Kaggle](https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge)
- Selected labels: `[0: Angry, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral]`
- Model structure:
  - Conv2D + ELU + BatchNorm + Dropout
  - Dense + Softmax (5 classes)
- Loss: `categorical_crossentropy`
- Optimizer: `Nadam`
- Augmentation: Rotation, Shift, Zoom, Flip

### 📦 Training Config
- Epochs: 100  
- Batch Size: 64  
- Validation Split: 10%  
- Best Model Saved To:
```bash
saved_models/best_emotion_model.keras
