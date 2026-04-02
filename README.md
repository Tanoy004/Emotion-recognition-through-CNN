# Four-Layer ConvNet for Facial Emotion Recognition

This repository contains the implementation and experimental analysis of the research work:

**“Four-layer ConvNet to facial emotion recognition with minimal epochs and the significance of data diversity”**  
Published in *Scientific Reports (Nature), 2022*

---

## 📌 Overview

This project presents a deep learning-based framework for facial emotion recognition using a custom-designed Convolutional Neural Network (ConvNet). The model focuses on achieving high accuracy with minimal training epochs while emphasizing the importance of data diversity in improving generalization performance.

The system is capable of recognizing seven human emotions:
- Anger
- Disgust
- Fear
- Happiness
- Neutral
- Sadness
- Surprise

---

## 🧠 Methodology

- Designed a **4-layer Convolutional Neural Network (ConvNet)** architecture  
- Integrated feature extraction techniques including:
  - Local Binary Patterns (LBP)
  - ORB (Oriented FAST and Rotated BRIEF)
- Applied **transfer learning (VGG16)** for improved performance  
- Used **data diversity strategies** to enhance generalization  
- Evaluated performance on multiple benchmark datasets  

---

## 📊 Datasets

The model was evaluated on publicly available datasets:
- FER2013 dataset  
- JAFFE dataset  
- CK+ dataset  

---

## 📈 Results

- Achieved up to:
  - **98.13% accuracy on CK+ dataset**
  - **~92% accuracy on JAFFE dataset**
- Demonstrated strong generalization capability across datasets  
- Outperformed several traditional machine learning approaches  

---

## ⚙️ Implementation

- Language: Python  
- Frameworks: TensorFlow / Keras  
- Model type: CNN + Transfer Learning  
- Includes:
  - Data preprocessing scripts  
  - Model training pipelines  
  - Evaluation and testing modules  

---

## 👨‍💻 Role

First author and lead researcher. Designed the ConvNet architecture, conducted experiments, analyzed results, and led the overall research and implementation of the project.

---

## 🔗 Publication

Debnath, T., et al.  
*Four-layer ConvNet to facial emotion recognition with minimal epochs and the significance of data diversity*  
Scientific Reports, 2022  

---

## 📎 Repository Link (Cited in Paper)

This repository is publicly referenced in the published article as part of the research reproducibility.

---

## 🚀 Applications

- Human-computer interaction  
- Affective computing  
- Healthcare monitoring  
- Behavioral analysis systems  

---

## 📌 Note

This repository reflects the research implementation corresponding to the published work. It is intended for academic and research purposes.

---
