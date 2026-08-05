# ❤️ ECGDarshan
## AI-Powered Cardiac Risk Analysis System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-90.27%25-green)
![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey)

---

## 📌 About

ECGDarshan is an AI-powered cardiac risk analysis
system that automatically classifies ECG heartbeat
signals into 5 categories using a 1D Convolutional
Neural Network and generates personalized Cardiac
Risk Scores by combining ECG results with 17 patient
health parameters.

---

## ✅ Features

- 1D CNN model — 90.27% accuracy
- MIT-BIH Arrhythmia Database (87,554 samples)
- 5 class ECG classification
- PQRST wave detection with measurements
- Cardiac Risk Score calculator (0-100)
- 4 risk levels with color coding
- Flask web dashboard — medical green theme
- ECG image/PDF upload with OpenCV extraction
- Disease detection and clinical recommendations
- Annotated ECG graph with P/Q/R/S/T peaks

---

## 🛠️ Technology Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.11.0 | Core language |
| TensorFlow | 2.21.0 | CNN model |
| Flask | 3.1.3 | Web dashboard |
| OpenCV | 4.13.0 | Image processing |
| NeuroKit2 | 0.2.13 | PQRST detection |
| Scikit-learn | 1.9.0 | Preprocessing |
| Imbalanced-learn | 0.14.2 | SMOTE balancing |
| Matplotlib | 3.11.0 | ECG graphs |
| PyMuPDF | Latest | PDF conversion |

---

## 📊 Dataset

MIT-BIH Arrhythmia Database from PhysioNet

| Split | Samples |
|-------|---------|
| Training | 87,554 |
| Testing | 21,892 |

| Class | Label | Training |
|-------|-------|----------|
| Normal | 0 | 72,471 |
| Supraventricular | 1 | 2,223 |
| Ventricular | 2 | 5,788 |
| Fusion | 3 | 641 |
| Unknown | 4 | 6,431 |

---

## 📈 Results

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Normal | 0.94 | 1.00 | 0.97 |
| Supraventricular | 0.92 | 0.59 | 0.72 |
| Ventricular | 0.57 | 0.94 | 0.71 |
| Overall | | | 0.90 |

---

## 🚀 How to Run

### 1. Install requirements:
