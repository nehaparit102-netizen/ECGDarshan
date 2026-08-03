import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

print("Loading model and data...")
model = tf.keras.models.load_model('data/ecgdarshan_model.h5')

# Load normal and abnormal data
normal   = pd.read_csv('data/Normal.csv',   header=None)
abnormal = pd.read_csv('data/Abnormal .csv', header=None)

print(f"Normal samples available:   {len(normal)}")
print(f"Abnormal samples available: {len(abnormal)}")

class_names = ['Normal', 'Supraventricular',
               'Ventricular', 'Fusion', 'Unknown']

def compute_risk_score(age, sex, hypertension,
                       diabetes, smoking, obesity):
    age_score = (age - 18) / 72 * 8
    sex_score = 2 if sex == 'M' else 0
    comorbid  = 0
    if hypertension: comorbid += 15
    if diabetes:     comorbid += 20
    if smoking:      comorbid += 18
    if obesity:      comorbid += 12
    return min(age_score + sex_score + comorbid, 100)

def get_risk_level(ecg_class, risk_score):
    if risk_score >= 50 and ecg_class != 0:
        return "🔴 HIGH RISK", "Immediate medical attention!"
    elif risk_score >= 65:
        return "🔴 HIGH RISK", "Immediate medical attention!"
    elif risk_score >= 35 or ecg_class in [1,2,3,4]:
        return "🟠 MODERATE RISK", "Doctor consultation needed"
    elif risk_score >= 20:
        return "🟡 LOW RISK", "Regular checkup recommended"
    else:
        return "🟢 NORMAL", "No immediate action needed"

def analyze_from_normal(sample_id, age, sex,
                        hypertension, diabetes,
                        smoking, obesity):
    # Get ECG from Normal.csv
    ecg = normal.iloc[sample_id, :187].values

    # Normalize
    scaler = StandardScaler()
    ecg    = scaler.fit_transform(ecg.reshape(-1,1)).flatten()
    ecg    = ecg.reshape(1, 187, 1)

    # Predict
    prediction = model.predict(ecg, verbose=0)
    ecg_class  = np.argmax(prediction[0])
    confidence = round(float(prediction[0][ecg_class])*100, 1)

    # Risk score
    risk_score = round(compute_risk_score(
        age, sex, hypertension,
        diabetes, smoking, obesity
    ), 1)

    risk_level, action = get_risk_level(ecg_class, risk_score)

    print("\n" + "="*45)
    print("     ECGDarshan - Friend ECG Analysis")
    print("="*45)
    print(f"ECG Source         : Normal.csv")
    print(f"Sample ID          : {sample_id}")
    print(f"ECG Classification : {class_names[ecg_class]}")
    print(f"Confidence         : {confidence}%")
    print(f"Patient Age        : {age} years")
    print(f"Risk Score         : {risk_score} / 100")
    print(f"Risk Level         : {risk_level}")
    print(f"Recommendation     : {action}")
    print("="*45)

def analyze_from_abnormal(sample_id, age, sex,
                          hypertension, diabetes,
                          smoking, obesity):
    # Get ECG from Abnormal.csv
    ecg = abnormal.iloc[sample_id, :187].values

    # Normalize
    scaler = StandardScaler()
    ecg    = scaler.fit_transform(ecg.reshape(-1,1)).flatten()
    ecg    = ecg.reshape(1, 187, 1)

    # Predict
    prediction = model.predict(ecg, verbose=0)
    ecg_class  = np.argmax(prediction[0])
    confidence = round(float(prediction[0][ecg_class])*100, 1)

    # Risk score
    risk_score = round(compute_risk_score(
        age, sex, hypertension,
        diabetes, smoking, obesity
    ), 1)

    risk_level, action = get_risk_level(ecg_class, risk_score)

    print("\n" + "="*45)
    print("     ECGDarshan - Friend ECG Analysis")
    print("="*45)
    print(f"ECG Source         : Abnormal.csv")
    print(f"Sample ID          : {sample_id}")
    print(f"ECG Classification : {class_names[ecg_class]}")
    print(f"Confidence         : {confidence}%")
    print(f"Patient Age        : {age} years")
    print(f"Risk Score         : {risk_score} / 100")
    print(f"Risk Level         : {risk_level}")
    print(f"Recommendation     : {action}")
    print("="*45)

# ---- TEST CASES ----

# Friend 1 - Normal ECG, Young healthy girl
print("\n👧 Friend 1 - Young Healthy Girl:")
analyze_from_normal(
    sample_id    = 0,
    age          = 21,
    sex          = 'F',
    hypertension = False,
    diabetes     = False,
    smoking      = False,
    obesity      = False
)

# Friend 2 - Abnormal ECG, Middle age man
print("\n👨 Friend 2 - Middle Age Man:")
analyze_from_abnormal(
    sample_id    = 0,
    age          = 50,
    sex          = 'M',
    hypertension = True,
    diabetes     = True,
    smoking      = False,
    obesity      = False
)