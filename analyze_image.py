import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

print("Loading model...")
model = tf.keras.models.load_model('data/ecgdarshan_model.h5')
class_names = ['Normal', 'Supraventricular',
               'Ventricular', 'Fusion', 'Unknown']

def extract_signal_from_image(image_path, num_points=187):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image! Check file path.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w = thresh.shape
    signal = []

    for x in range(w):
        column = thresh[:, x]
        dark_pixels = np.where(column > 0)[0]
        if len(dark_pixels) > 0:
            y = dark_pixels[0]
            signal.append(h - y)
        else:
            signal.append(np.nan)

    signal = np.array(signal, dtype=float)

    nans = np.isnan(signal)
    if nans.any():
        signal[nans] = np.interp(
            np.flatnonzero(nans),
            np.flatnonzero(~nans),
            signal[~nans]
        )

    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, num_points)
    signal_resampled = np.interp(x_new, x_old, signal)

    return signal_resampled

def analyze_ecg_image(image_path, age, sex,
                      hypertension, diabetes,
                      smoking, obesity):
    print(f"\nExtracting signal from: {image_path}")
    raw_signal = extract_signal_from_image(image_path)

    scaler = StandardScaler()
    norm_signal = scaler.fit_transform(
        raw_signal.reshape(-1, 1)
    ).flatten()

    model_input = norm_signal.reshape(1, 187, 1)

    prediction  = model.predict(model_input, verbose=0)
    ecg_class   = np.argmax(prediction[0])
    confidence  = round(float(prediction[0][ecg_class]) * 100, 1)

    age_score = (age - 18) / 72 * 8
    sex_score = 2 if sex == 'M' else 0
    comorbid  = 0
    if hypertension: comorbid += 15
    if diabetes:     comorbid += 20
    if smoking:      comorbid += 18
    if obesity:      comorbid += 12
    risk_score = round(min(age_score + sex_score + comorbid, 100), 1)

    if risk_score >= 50 and ecg_class != 0:
        risk_level = "🔴 HIGH RISK"
    elif risk_score >= 65:
        risk_level = "🔴 HIGH RISK"
    elif risk_score >= 35 or ecg_class in [1,2,3,4]:
        risk_level = "🟠 MODERATE RISK"
    elif risk_score >= 20:
        risk_level = "🟡 LOW RISK"
    else:
        risk_level = "🟢 NORMAL"

    print("\n" + "="*45)
    print("   ECGDarshan - Uploaded ECG Report Analysis")
    print("="*45)
    print(f"Image File          : {image_path}")
    print(f"ECG Classification  : {class_names[ecg_class]}")
    print(f"Confidence          : {confidence}%")
    print(f"Patient Age         : {age} years")
    print(f"Risk Score          : {risk_score} / 100")
    print(f"Risk Level          : {risk_level}")
    print("="*45)
    print("\n⚠️ Note: Signal extracted from image is an")
    print("approximation. Real clinical use needs direct")
    print("digital ECG data for full accuracy.")

# ---- TEST ----
analyze_ecg_image(
    image_path   = 'data/friend_ecg.jpg',
    age          = 22,
    sex          = 'M',
    hypertension = False,
    diabetes     = False,
    smoking      = False,
    obesity      = False
)