from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pqrst_analysis import (
    analyze_pqrst,
    detect_disease_from_pqrst,
    generate_pqrst_graph
)

app = Flask(__name__)

print("Loading ECGDarshan model...")
try:
    model = tf.keras.models.load_model(
        'data/ecgdarshan_model_v2.keras'
    )
    print("V2 Model loaded! ✅")
except:
    model = tf.keras.models.load_model(
        'data/ecgdarshan_model.h5'
    )
    print("V1 Model loaded! ✅")

print("All data loaded! ✅")

class_names = ['Normal', 'Supraventricular',
               'Ventricular', 'Fusion', 'Unknown']

diseases = {
    0: {
        'name'   : 'No Cardiac Disease Detected',
        'details': 'Sinus Rhythm — Normal heartbeat',
        'icon'   : '✅',
        'color'  : '#2ecc71'
    },
    1: {
        'name'   : 'Supraventricular Arrhythmia',
        'details': 'Possible: Atrial Fibrillation, '
                   'Flutter, PAC or PSVT',
        'icon'   : '⚠️',
        'color'  : '#3498db'
    },
    2: {
        'name'   : 'Ventricular Arrhythmia',
        'details': 'Possible: PVC, Ventricular '
                   'Tachycardia or Bundle Branch Block',
        'icon'   : '🚨',
        'color'  : '#e63946'
    },
    3: {
        'name'   : 'Fusion Beat Detected',
        'details': 'Mixed Normal + Ventricular beat, '
                   'Competing electrical signals',
        'icon'   : '⚠️',
        'color'  : '#f39c12'
    },
    4: {
        'name'   : 'Unknown Cardiac Pattern',
        'details': 'Possible: Pacemaker beat, '
                   'Rare arrhythmia or Signal artifact',
        'icon'   : '❓',
        'color'  : '#95a5a6'
    }
}

condition_advice = {
    0: [
        '✅ Heart rhythm is normal',
        '✅ No arrhythmia detected',
        '✅ QRS complex is normal',
        '✅ P wave is normal',
        '💊 Continue regular health checkups'
    ],
    1: [
        '⚠️ Abnormal P wave detected',
        '⚠️ Premature beats from atria',
        '🏥 Cardiologist consultation needed',
        '💊 May need antiarrhythmic medication',
        '📋 Holter monitor test recommended'
    ],
    2: [
        '🚨 Wide QRS complex detected',
        '🚨 Ventricular origin beat found',
        '🏥 Immediate cardiology referral',
        '💊 May need urgent medication',
        '📋 Echo and stress test recommended'
    ],
    3: [
        '⚠️ Mixed heartbeat pattern detected',
        '⚠️ Two competing electrical signals',
        '🏥 Doctor consultation needed',
        '💊 Monitor heart rhythm closely',
        '📋 24-hour ECG monitoring recommended'
    ],
    4: [
        '❓ Unrecognized ECG pattern',
        '❓ Could be pacemaker or rare condition',
        '🏥 Immediate doctor review needed',
        '💊 Further cardiac tests required',
        '📋 Specialist referral recommended'
    ]
}

def compute_risk_score(age, sex,
                       hypertension, diabetes,
                       smoking, obesity,
                       heart_failure, previous_mi,
                       kidney_disease, thyroid,
                       high_cholesterol, family_history,
                       heart_rate, bp_systolic,
                       spo2, blood_sugar,
                       qrs_duration):
    age_score = (age - 18) / 72 * 8
    sex_score = 2 if sex == 'M' else 0
    comorbid  = 0

    if hypertension:     comorbid += 15
    if diabetes:         comorbid += 20
    if smoking:          comorbid += 18
    if obesity:          comorbid += 12
    if heart_failure:    comorbid += 25
    if previous_mi:      comorbid += 22
    if kidney_disease:   comorbid += 10
    if thyroid:          comorbid += 8
    if high_cholesterol: comorbid += 12
    if family_history:   comorbid += 10

    if heart_rate:
        if heart_rate > 100: comorbid += 10
        if heart_rate < 50:  comorbid += 15

    if bp_systolic:
        if bp_systolic > 140: comorbid += 10
        if bp_systolic > 180: comorbid += 20

    if spo2:
        if spo2 < 95: comorbid += 15
        if spo2 < 90: comorbid += 25

    if blood_sugar:
        if blood_sugar > 126: comorbid += 10
        if blood_sugar > 200: comorbid += 20

    if qrs_duration:
        if qrs_duration > 120: comorbid += 15
        if qrs_duration > 150: comorbid += 25

    return min(age_score + sex_score + comorbid, 100)

def get_risk_level(ecg_class, risk_score):
    if risk_score >= 50 and ecg_class != 0:
        return "🔴 HIGH RISK", "Immediate medical attention!", "high"
    elif risk_score >= 65:
        return "🔴 HIGH RISK", "Immediate medical attention!", "high"
    elif risk_score >= 35 or ecg_class in [1,2,3,4]:
        return "🟠 MODERATE RISK", "Doctor consultation needed", "moderate"
    elif risk_score >= 20:
        return "🟡 LOW RISK", "Regular checkup recommended", "low"
    else:
        return "🟢 NORMAL", "No immediate action needed", "normal"

def extract_signal_from_image(image_path,
                               num_points=187):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read image!")

    # Remove red/pink grid lines
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0,   50,  50])
    upper_red1 = np.array([10,  255, 255])
    lower_red2 = np.array([170, 50,  50])
    upper_red2 = np.array([180, 255, 255])
    mask1    = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2    = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[red_mask > 0] = 255
    gray = cv2.equalizeHist(gray)

    _, thresh = cv2.threshold(
        gray, 127, 255,
        cv2.THRESH_BINARY_INV
    )

    kernel = np.ones((2,2), np.uint8)
    thresh  = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel
    )

    h, w = thresh.shape
    signal = []

    for x in range(w):
        column      = thresh[:, x]
        dark_pixels = np.where(column > 0)[0]
        if len(dark_pixels) > 0:
            y = int(np.mean(dark_pixels))
            signal.append(h - y)
        else:
            signal.append(np.nan)

    signal = np.array(signal, dtype=float)
    nans   = np.isnan(signal)

    if nans.any():
        if np.sum(~nans) > 1:
            signal[nans] = np.interp(
                np.flatnonzero(nans),
                np.flatnonzero(~nans),
                signal[~nans]
            )
        else:
            signal = np.zeros(len(signal))

    x_old  = np.linspace(0, 1, len(signal))
    x_new  = np.linspace(0, 1, num_points)
    signal = np.interp(x_new, x_old, signal)

    if signal.max() != signal.min():
        signal = (signal - signal.min()) / \
                 (signal.max() - signal.min())
        signal = signal * 6 - 3

    return signal

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/analyze_upload', methods=['POST'])
def analyze_upload():
    age              = int(request.form.get('age2', 50))
    sex              = request.form.get('sex2', 'M')
    hypertension     = bool(request.form.get('hypertension2'))
    diabetes         = bool(request.form.get('diabetes2'))
    smoking          = bool(request.form.get('smoking2'))
    obesity          = bool(request.form.get('obesity2'))
    heart_failure    = bool(request.form.get('heart_failure2'))
    previous_mi      = bool(request.form.get('previous_mi2'))
    kidney_disease   = bool(request.form.get('kidney_disease2'))
    thyroid          = bool(request.form.get('thyroid2'))
    high_cholesterol = bool(request.form.get('high_cholesterol2'))
    family_history   = bool(request.form.get('family_history2'))

    heart_rate   = request.form.get('heart_rate')
    bp_systolic  = request.form.get('bp_systolic')
    spo2         = request.form.get('spo2')
    blood_sugar  = request.form.get('blood_sugar')
    qrs_duration = request.form.get('qrs_duration')

    heart_rate   = int(heart_rate)   if heart_rate   else None
    bp_systolic  = int(bp_systolic)  if bp_systolic  else None
    spo2         = int(spo2)         if spo2         else None
    blood_sugar  = int(blood_sugar)  if blood_sugar  else None
    qrs_duration = int(qrs_duration) if qrs_duration else None

    file     = request.files['ecg_image']
    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    raw_signal  = extract_signal_from_image(filepath)
    scaler      = StandardScaler()
    norm_signal = scaler.fit_transform(
        raw_signal.reshape(-1, 1)
    ).flatten()
    model_input = norm_signal.reshape(1, 187, 1)

    prediction = model.predict(model_input, verbose=0)
    ecg_class  = int(np.argmax(prediction[0]))
    confidence = round(
        float(prediction[0][ecg_class]) * 100, 1
    )

    risk_score = round(compute_risk_score(
        age, sex,
        hypertension, diabetes,
        smoking, obesity,
        heart_failure, previous_mi,
        kidney_disease, thyroid,
        high_cholesterol, family_history,
        heart_rate, bp_systolic,
        spo2, blood_sugar,
        qrs_duration
    ), 1)

    risk_level, action, css = get_risk_level(
        ecg_class, risk_score
    )

    # PQRST Analysis
    pqrst_data   = analyze_pqrst(raw_signal)
    disease_info = detect_disease_from_pqrst(
        pqrst_data, ecg_class
    )

    # Generate PQRST annotated graph
    generate_pqrst_graph(
        raw_signal, pqrst_data,
        ecg_class, class_names
    )

    # Get PQRST measurements
    hr_detected  = pqrst_data['heart_rate']   if pqrst_data else 0
    qrs_detected = pqrst_data['qrs_duration'] if pqrst_data else 0
    pr_detected  = pqrst_data['pr_interval']  if pqrst_data else 0
    qt_detected  = pqrst_data['qt_interval']  if pqrst_data else 0

    result = {
        'ecg_class'      : class_names[ecg_class],
        'ecg_class_id'   : ecg_class,
        'confidence'     : confidence,
        'risk_score'     : risk_score,
        'risk_level'     : risk_level,
        'action'         : action,
        'css'            : css,
        'age'            : age,
        'source_name'    : f"Uploaded: {file.filename}",
        'graph'          : True,
        'disease_name'   : diseases[ecg_class]['name'],
        'disease_detail' : diseases[ecg_class]['details'],
        'disease_icon'   : diseases[ecg_class]['icon'],
        'disease_color'  : diseases[ecg_class]['color'],
        'advice_list'    : condition_advice[ecg_class],
        'pqrst_diseases' : disease_info['diseases'],
        'pqrst_severity' : disease_info['severity'],
        'pqrst_recs'     : disease_info['recommendations'],
        'hr_detected'    : hr_detected,
        'qrs_detected'   : qrs_detected,
        'pr_detected'    : pr_detected,
        'qt_detected'    : qt_detected,
        'pqrst_available': pqrst_data is not None
    }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    print("\n🚀 ECGDarshan is running!")
    print("Open browser and go to:")
    print("http://127.0.0.1:5000")
    app.run(debug=True)