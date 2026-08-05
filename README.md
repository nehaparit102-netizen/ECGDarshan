# ❤️ ECGDarshan
## AI-Powered Cardiac Risk Analysis System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-90.27%25-green)
![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey)

---

## 📌 About Project

ECGDarshan is an AI-powered cardiac risk analysis
system developed as B.Tech Final Year Capstone
Project at RIT Islampur.

It automatically classifies ECG heartbeat signals
into 5 categories using 1D CNN and generates
personalized Cardiac Risk Score by combining ECG
results with 17 patient health parameters.

---

## ✅ Features

- 1D CNN model — 90.27% accuracy
- MIT-BIH Arrhythmia Database (87,554 samples)
- 5 class ECG classification
- PQRST wave detection with measurements
- Cardiac Risk Score (0-100)
- 4 risk levels with color coding
- Flask web dashboard
- ECG image/PDF upload feature
- Disease detection and recommendations

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.11 | Core language |
| TensorFlow | 2.21 | CNN model |
| Keras | 3.15 | Deep learning |
| Flask | 3.1 | Web dashboard |
| OpenCV | 4.13 | Image processing |
| NeuroKit2 | 0.2.13 | PQRST detection |
| Scikit-learn | 1.9 | Preprocessing |
| Imbalanced-learn | 0.14 | SMOTE balancing |
| Matplotlib | 3.11 | ECG graphs |
| PyMuPDF | Latest | PDF conversion |

---

## 📊 Dataset

MIT-BIH Arrhythmia Database — PhysioNet

| Split | Samples |
|-------|---------|
| Training | 87,554 |
| Testing | 21,892 |

| Class | Label | Count |
|-------|-------|-------|
| Normal | 0 | 72,471 |
| Supraventricular | 1 | 2,223 |
| Ventricular | 2 | 5,788 |
| Fusion | 3 | 641 |
| Unknown | 4 | 6,431 |

---

## 📈 Model Results

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Normal | 0.94 | 1.00 | 0.97 |
| Supraventricular | 0.92 | 0.59 | 0.72 |
| Ventricular | 0.57 | 0.94 | 0.71 |
| Overall Accuracy | | | 0.90 |

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| app.py | Flask web application |
| model.py | CNN model V1 training |
| model2.py | CNN model V2 with SMOTE |
| preprocessing.py | Data preprocessing |
| risk_score.py | Risk score calculator |
| pqrst_analysis.py | PQRST detection |
| analyze_image.py | ECG image analysis |
| generate_test_ecg.py | Generate test images |
| templates/index.html | Web dashboard |

---

## 🚀 How to Run

### Step 1 — Install libraries:
pip install tensorflow numpy scipy pandas scikit-learn flask imbalanced-learn opencv-python PyMuPDF matplotlib neurokit2 wfdb

### Step 2 — Download MIT-BIH dataset:
Go to physionet.org and download MIT-BIH
Arrhythmia Database. Place CSV files in data/ folder.

### Step 3 — Preprocess data:
python preprocessing.py

### Step 4 — Train model:
python model.py

### Step 5 — Start website:
python app.py

### Step 6 — Open browser:
http://127.0.0.1:5000

---

## 🎯 How to Use

1. Enter patient age and sex
2. Enter vital signs (HR, BP, SpO2, QRS)
3. Check applicable health conditions
4. Upload ECG report image (JPG/PNG)
5. Click ANALYZE CARDIAC RISK
6. View results with ECG graph and recommendations

---

## 📊 Risk Levels

| Score | Risk Level | Action |
|-------|-----------|--------|
| 0-19 | 🟢 Normal | Regular checkup |
| 20-34 | 🟡 Low Risk | Monitor health |
| 35-64 | 🟠 Moderate Risk | Doctor consultation |
| 65-100 | 🔴 High Risk | Immediate attention |

---

## 👥 Team Members

| Name       | Role |
|------ ----esting and Analysis  |------|
| Neha Parit | Team Lead ,Model Development,Web Dashboard,Data Processing,Testing and Analysis | 


---

## 🏫 Institution

Rajarambapu Institute of Technology, Islampur
Department of Electronics and Telecommunication
B.Tech Final Year Capstone Project 2025-26

---

## ⚠️ Limitations

- Trained on MIT-BIH database only
- Image extraction is approximate
- Local deployment only currently

---

## 🔮 Future Scope

- Live ECG machine integration
- Cloud deployment
- Mobile application
- Indian patient dataset training
- Multi-lead ECG analysis
