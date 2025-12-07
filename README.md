# 🐦 BirdCLEF 2024 – Bird Sound Classification using CNNs & Spectrograms  
### Deep Learning Project | InceptionV1 / ResNet18 / MobileNetV3 / VGG-like

---

## 📌 Project Overview

This project focuses on **automatic bird species classification** from audio recordings using  
**Mel-Spectrograms + Convolutional Neural Networks (CNNs)**.

We trained and evaluated **four deep learning architectures**:

- **SimpleVGG** (custom VGG-like CNN from scratch)  
- **ResNet18** (Transfer Learning)  
- **Inception V1 – GoogLeNet** (Transfer Learning)  
- **MobileNetV3-Small** (Transfer Learning)  

Each model was evaluated using:

- Accuracy  
- Precision, Recall, F1-score (Macro)  
- Confusion Matrix  
- ROC Curves  
- Macro ROC–AUC  
- Training Curves (Loss & Accuracy)

This repository includes code, training pipeline, evaluation tools, and saved model weights.

---

## 📂 Dataset Description

We use a subset of **BirdCLEF 2024** dataset.

- Audio sampled at **32 kHz**
- Filtered clips with `rating ≥ 3`
- Top **10 most common species**
- Data split:
  - **80% Training**
  - **20% Validation**
- Each class is oversampled to **1000 samples**  
  → total **10,000 training examples**

---

## 🎧 Audio Preprocessing

Each audio file is processed as follows:

1. Load audio at 32 kHz  
2. Crop / pad to **5 seconds**  
3. Apply audio augmentations:
   - Time shifting  
   - Pitch shifting  
   - Additive noise  
   - Time stretching  
4. Convert to **Log-Mel Spectrogram**  
5. Normalize spectrogram  
6. Apply **SpecAugment** (time/frequency masking)  
7. Resize to:
   - **224×224** for VGG, ResNet, MobileNet, InceptionV1  
   - **299×299** only for InceptionV3 (if used)

---

## 🧠 Model Architectures

### **1) SimpleVGG (from scratch)**
Small VGG-style CNN stacked with Conv + ReLU + MaxPooling.

**Pros:** simple, fast, interpretable  
**Cons:** shallow, weaker accuracy, no skip connections  

---

### **2) ResNet18 (Transfer Learning)**
Uses skip connections → enables deep feature learning.

**Pros:** stable training, strong performance  
**Cons:** heavier than MobileNet  

---

### **3) Inception V1 (GoogLeNet) – Transfer Learning**
Uses multi-branch convolutions → captures multi-scale patterns.

**Pros:** excellent for spectrogram textures  
**Cons:** includes auxiliary outputs (handled in training)  

---

### **4) MobileNetV3-Small (Transfer Learning)**
Lightweight, optimized for mobile deployment.

**Pros:** fastest model, smallest size  
**Cons:** slightly lower accuracy  

---

## 🏋️ Training Details

- Optimizer: **Adam**  
- Learning Rate: `1e-4`  
- Loss: **CrossEntropy**  
- Batch size: **32**  
- Epochs: **20**  
- Early Stopping (patience = 3)  
- Supports:
  - 1-output models
  - 2-output models (InceptionV3)
  - 3-output models (InceptionV1 / GoogLeNet)

---

## 📊 Evaluation Metrics

We compute:

- Accuracy  
- Precision (macro)  
- Recall (macro)  
- F1-score (macro)  
- ROC curves per class  
- Macro ROC–AUC  
- Confusion Matrix  
- Training Curves (Loss & Accuracy)

---

## 📈 Results Summary

| Model            | Accuracy | Macro AUC | Notes |
|------------------|----------|-----------|-------|
| **Inception V1** | ⭐ High   | ⭐ High    | Best multi-scale performance |
| **ResNet18**     | High     | High      | Stable & reliable |
| **MobileNetV3**  | Very Good| Very Good | Fastest + lightweight |
| **SimpleVGG**    | Good     | Good      | Strong baseline |
