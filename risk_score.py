import numpy as np
import tensorflow as tf

# Load trained model
print("Loading ECGDarshan model...")
model = tf.keras.models.load_model('data/ecgdarshan_model.h5')
print("Model loaded! ✅")

# Risk labels
class_names = ['Normal', 'Supraventricular', 
                'Ventricular', 'Fusion', 'Unknown']

def compute_risk_score(age, sex, hypertension, 
                       diabetes, smoking, obesity):
    # Demographics score
    age_score = (age - 18) / 72 * 8

    # Sex score
    sex_score = 2 if sex == 'M' else 0

    # Comorbidity scores
    comorbid_score = 0
    if hypertension: comorbid_score += 15
    if diabetes:     comorbid_score += 20
    if smoking:      comorbid_score += 18
    if obesity:      comorbid_score += 12

    total = age_score + sex_score + comorbid_score
    return min(total, 100)

def get_risk_level(ecg_class, risk_score):
    # Combine ECG result with risk score
    if ecg_class == 0 and risk_score < 25:
        return "🟢 NORMAL", "No immediate action needed"
    elif ecg_class == 0 and risk_score < 50:
        return "🟡 LOW RISK", "Regular checkup recommended"
    elif ecg_class in [1,2] or risk_score < 75:
        return "🟠 MODERATE RISK", "Doctor consultation needed"
    else:
        return "🔴 HIGH RISK", "Immediate medical attention!"

def analyze_patient(ecg_signal, age, sex, 
                    hypertension, diabetes, 
                    smoking, obesity):

    # Reshape signal for model
    signal = np.array(ecg_signal).reshape(1, 187, 1)

    # Predict ECG class
    prediction   = model.predict(signal, verbose=0)
    ecg_class    = np.argmax(prediction[0])
    confidence   = prediction[0][ecg_class] * 100

    # Compute risk score
    risk_score = compute_risk_score(
        age, sex, hypertension, 
        diabetes, smoking, obesity
    )

    # Get risk level
    risk_level, action = get_risk_level(ecg_class, risk_score)

    # Print results
    print("\n" + "="*45)
    print("       ECGDarshan Analysis Report")
    print("="*45)
    print(f"ECG Classification : {class_names[ecg_class]}")
    print(f"Confidence         : {confidence:.1f}%")
    print(f"Patient Age        : {age} years")
    print(f"Risk Score         : {risk_score:.1f} / 100")
    print(f"Risk Level         : {risk_level}")
    print(f"Recommendation     : {action}")
    print("="*45)

# ---- TEST with sample patient ----
print("\nTesting with sample patient...")

# Load one real ECG from test data
X_test = np.load('data/X_test.npy')
sample_ecg = X_test[0].flatten()  # first ECG signal

# Sample patient details
analyze_patient(
    ecg_signal   = sample_ecg,
    age          = 55,
    sex          = 'M',
    hypertension = True,
    diabetes     = True,
    smoking      = False,
    obesity      = False
)