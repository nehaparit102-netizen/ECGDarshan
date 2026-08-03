import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def analyze_pqrst(signal, sampling_rate=360):
    try:
        import neurokit2 as nk
        ecg_signals, info = nk.ecg_process(
            signal, sampling_rate=sampling_rate
        )
        r_peaks = info['ECG_R_Peaks']
        p_peaks = info.get('ECG_P_Peaks', np.array([]))
        q_peaks = info.get('ECG_Q_Peaks', np.array([]))
        s_peaks = info.get('ECG_S_Peaks', np.array([]))
        t_peaks = info.get('ECG_T_Peaks', np.array([]))

        if len(r_peaks) > 1:
            rr_interval = np.mean(
                np.diff(r_peaks)
            ) / sampling_rate * 1000
            heart_rate = round(60000 / rr_interval, 1)
        else:
            rr_interval = 0
            heart_rate  = 0

        if len(q_peaks) > 0 and len(s_peaks) > 0:
            min_len = min(len(q_peaks), len(s_peaks))
            qrs_duration = round(np.mean(
                s_peaks[:min_len] - q_peaks[:min_len]
            ) / sampling_rate * 1000, 1)
        else:
            qrs_duration = 0

        if len(p_peaks) > 0 and len(r_peaks) > 0:
            min_len = min(len(p_peaks), len(r_peaks))
            pr_interval = round(np.mean(
                r_peaks[:min_len] - p_peaks[:min_len]
            ) / sampling_rate * 1000, 1)
        else:
            pr_interval = 0

        if len(q_peaks) > 0 and len(t_peaks) > 0:
            min_len = min(len(q_peaks), len(t_peaks))
            qt_interval = round(np.mean(
                t_peaks[:min_len] - q_peaks[:min_len]
            ) / sampling_rate * 1000, 1)
        else:
            qt_interval = 0

        return {
            'r_peaks'     : r_peaks.tolist(),
            'p_peaks'     : p_peaks.tolist() if len(p_peaks) > 0 else [],
            'q_peaks'     : q_peaks.tolist() if len(q_peaks) > 0 else [],
            's_peaks'     : s_peaks.tolist() if len(s_peaks) > 0 else [],
            't_peaks'     : t_peaks.tolist() if len(t_peaks) > 0 else [],
            'heart_rate'  : heart_rate,
            'rr_interval' : round(rr_interval, 1),
            'qrs_duration': qrs_duration,
            'pr_interval' : pr_interval,
            'qt_interval' : qt_interval
        }

    except Exception as e:
        print(f"PQRST analysis error: {e}")
        return None

def detect_disease_from_pqrst(pqrst_data, ecg_class):
    if pqrst_data is None:
        return get_default_disease(ecg_class)

    hr  = pqrst_data['heart_rate']
    qrs = pqrst_data['qrs_duration']
    pr  = pqrst_data['pr_interval']
    qt  = pqrst_data['qt_interval']

    diseases_found  = []
    severity        = 'Normal'
    recommendations = []

    # Heart Rate checks
    if hr > 100:
        diseases_found.append(
            '⚡ Tachycardia (HR > 100 bpm)'
        )
        severity = 'Moderate'
        recommendations.append(
            'Monitor heart rate closely'
        )
    elif hr < 60 and hr > 0:
        diseases_found.append(
            '🐢 Bradycardia (HR < 60 bpm)'
        )
        severity = 'Moderate'
        recommendations.append(
            'Check for heart block'
        )

    # QRS Duration checks
    if qrs > 120:
        diseases_found.append(
            '📊 Wide QRS — Bundle Branch Block'
        )
        severity = 'High'
        recommendations.append(
            'Cardiology referral needed'
        )
    elif qrs > 100 and qrs <= 120:
        diseases_found.append(
            '⚠️ Borderline QRS width detected'
        )
        if severity == 'Normal':
            severity = 'Low'

    # PR Interval checks
    if pr > 200:
        diseases_found.append(
            '⏱️ Long PR — First Degree AV Block'
        )
        if severity == 'Normal':
            severity = 'Moderate'
        recommendations.append(
            'ECG monitoring recommended'
        )
    elif pr < 120 and pr > 0:
        diseases_found.append(
            '⚡ Short PR — Pre-excitation Syndrome'
        )
        severity = 'Moderate'
        recommendations.append(
            'Check for WPW syndrome'
        )

    # QT Interval checks
    if qt > 450:
        diseases_found.append(
            '⚠️ Long QT Syndrome — Arrhythmia Risk!'
        )
        severity = 'High'
        recommendations.append(
            'Urgent cardiology evaluation!'
        )
    elif qt > 440 and qt <= 450:
        diseases_found.append(
            '⚠️ Borderline QT prolongation'
        )
        if severity in ['Normal', 'Low']:
            severity = 'Moderate'

    # ECG class specific diseases
    class_diseases = {
        1: {
            'diseases': [
                '🔵 Supraventricular Arrhythmia',
                'Possible: AFib, PAC or PSVT'
            ],
            'severity': 'Moderate',
            'recs': [
                'Holter monitor test',
                'Antiarrhythmic medication review',
                'Cardiologist consultation'
            ]
        },
        2: {
            'diseases': [
                '🔴 Ventricular Arrhythmia',
                'Possible: PVC or Ventricular Tachycardia'
            ],
            'severity': 'High',
            'recs': [
                'Immediate cardiology referral',
                'Echocardiogram and stress test',
                'Consider antiarrhythmic therapy'
            ]
        },
        3: {
            'diseases': [
                '🟡 Fusion Beat Detected',
                'Competing electrical signals in heart'
            ],
            'severity': 'Moderate',
            'recs': [
                'Doctor consultation needed',
                '24-hour ECG monitoring recommended'
            ]
        },
        4: {
            'diseases': [
                '❓ Unknown ECG Pattern',
                'Possible: Pacemaker or rare arrhythmia'
            ],
            'severity': 'Moderate',
            'recs': [
                'Specialist referral needed',
                'Further cardiac tests required'
            ]
        }
    }

    if ecg_class in class_diseases:
        info = class_diseases[ecg_class]
        diseases_found.extend(info['diseases'])
        if severity == 'Normal':
            severity = info['severity']
        recommendations.extend(info['recs'])

    # If nothing found
    if len(diseases_found) == 0:
        diseases_found = [
            '✅ No significant abnormality detected',
            '✅ Normal sinus rhythm',
            '✅ Normal PQRST intervals'
        ]
        recommendations = [
            'Continue regular health checkups',
            'Maintain healthy lifestyle',
            'Annual ECG screening recommended'
        ]

    return {
        'diseases'       : diseases_found,
        'severity'       : severity,
        'recommendations': recommendations,
        'hr'             : hr,
        'qrs'            : qrs,
        'pr'             : pr,
        'qt'             : qt
    }

def get_default_disease(ecg_class):
    defaults = {
        0: {
            'diseases'       : ['✅ Normal Sinus Rhythm',
                                '✅ No arrhythmia detected'],
            'severity'       : 'Normal',
            'recommendations': ['Regular health checkups',
                                'Healthy lifestyle'],
            'hr': 0, 'qrs': 0, 'pr': 0, 'qt': 0
        },
        1: {
            'diseases'       : ['⚠️ Supraventricular Arrhythmia',
                                'Possible AFib or PAC'],
            'severity'       : 'Moderate',
            'recommendations': ['Cardiologist consultation',
                                'Holter monitor test'],
            'hr': 0, 'qrs': 0, 'pr': 0, 'qt': 0
        },
        2: {
            'diseases'       : ['🚨 Ventricular Arrhythmia',
                                'Possible PVC or V-Tach'],
            'severity'       : 'High',
            'recommendations': ['Immediate medical attention',
                                'Emergency cardiology referral'],
            'hr': 0, 'qrs': 0, 'pr': 0, 'qt': 0
        },
        3: {
            'diseases'       : ['⚠️ Fusion Beat Detected',
                                'Mixed electrical signals'],
            'severity'       : 'Moderate',
            'recommendations': ['Doctor consultation',
                                '24-hour ECG monitoring'],
            'hr': 0, 'qrs': 0, 'pr': 0, 'qt': 0
        },
        4: {
            'diseases'       : ['❓ Unknown Pattern',
                                'Needs specialist review'],
            'severity'       : 'Moderate',
            'recommendations': ['Specialist referral',
                                'Further cardiac tests'],
            'hr': 0, 'qrs': 0, 'pr': 0, 'qt': 0
        }
    }
    return defaults.get(ecg_class, defaults[0])

def generate_pqrst_graph(signal, pqrst_data,
                          ecg_class, class_names):
    label = class_names[ecg_class]
    colors = {
        0: '#2ecc71', 1: '#3498db',
        2: '#e63946', 3: '#f39c12',
        4: '#95a5a6'
    }
    color = colors.get(ecg_class, '#2ecc71')

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(signal, color=color,
            linewidth=2, label='ECG Signal',
            zorder=2)

    if pqrst_data:
        # Mark P peaks
        for p in pqrst_data['p_peaks']:
            if 0 <= p < len(signal):
                ax.plot(p, signal[p], 'bo',
                        markersize=8, zorder=3)
                ax.annotate('P',
                    xy=(p, signal[p]),
                    xytext=(p-3, signal[p]+0.5),
                    fontsize=9, color='blue',
                    fontweight='bold')

        # Mark Q peaks
        for q in pqrst_data['q_peaks']:
            if 0 <= q < len(signal):
                ax.plot(q, signal[q], 'gs',
                        markersize=8, zorder=3)
                ax.annotate('Q',
                    xy=(q, signal[q]),
                    xytext=(q-3, signal[q]-0.7),
                    fontsize=9, color='green',
                    fontweight='bold')

        # Mark R peaks
        for r in pqrst_data['r_peaks']:
            if 0 <= r < len(signal):
                ax.plot(r, signal[r], 'r^',
                        markersize=10, zorder=3)
                ax.annotate('R',
                    xy=(r, signal[r]),
                    xytext=(r+2, signal[r]+0.3),
                    fontsize=9, color='red',
                    fontweight='bold')

        # Mark S peaks
        for s in pqrst_data['s_peaks']:
            if 0 <= s < len(signal):
                ax.plot(s, signal[s], 'mv',
                        markersize=8, zorder=3)
                ax.annotate('S',
                    xy=(s, signal[s]),
                    xytext=(s+2, signal[s]-0.7),
                    fontsize=9, color='purple',
                    fontweight='bold')

        # Mark T peaks
        for t in pqrst_data['t_peaks']:
            if 0 <= t < len(signal):
                ax.plot(t, signal[t], 'rd',
                        markersize=8, zorder=3)
                ax.annotate('T',
                    xy=(t, signal[t]),
                    xytext=(t+2, signal[t]+0.3),
                    fontsize=9, color='darkred',
                    fontweight='bold')

        # Measurements box
        hr  = pqrst_data['heart_rate']
        qrs = pqrst_data['qrs_duration']
        pr  = pqrst_data['pr_interval']
        qt  = pqrst_data['qt_interval']

        info_text = (
            f'HR: {hr} bpm  |  '
            f'QRS: {qrs} ms  |  '
            f'PR: {pr} ms  |  '
            f'QT: {qt} ms'
        )
        ax.text(0.02, 0.05, info_text,
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle='round',
                          facecolor='lightyellow',
                          alpha=0.9))

    # Disease message
    messages = {
        0: ('✅ Normal rhythm detected',   '#d8f3dc'),
        1: ('⚠️ Supraventricular pattern', '#cce5ff'),
        2: ('🚨 Ventricular pattern!',     '#ffccd5'),
        3: ('⚠️ Fusion beat pattern',      '#fff3cd'),
        4: ('❓ Unknown pattern',          '#e2e3e5')
    }
    msg, bg = messages.get(ecg_class, messages[0])
    ax.text(0.02, 0.92, msg,
            transform=ax.transAxes,
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round',
                      facecolor=bg, alpha=0.9))

    # Legend
    legend_elements = [
        mpatches.Patch(color='blue',
                       label='P — Atrial activity'),
        mpatches.Patch(color='green',
                       label='Q — Initial ventricular'),
        mpatches.Patch(color='red',
                       label='R — Peak ventricular'),
        mpatches.Patch(color='purple',
                       label='S — Terminal ventricular'),
        mpatches.Patch(color='darkred',
                       label='T — Ventricular recovery'),
    ]
    ax.legend(handles=legend_elements,
              loc='upper right',
              fontsize=8, ncol=2)

    ax.set_title(
        f'ECGDarshan — PQRST Analysis ({label})',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.set_xlabel(
        'Time Points (187 samples = 1 heartbeat)',
        fontsize=11
    )
    ax.set_ylabel('Amplitude (normalized)',
                  fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    os.makedirs('static', exist_ok=True)
    plt.savefig('static/ecg_result_graph.png',
                dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Testing PQRST analysis...")
    X_test = np.load('data/X_test.npy')
    signal = X_test[0].flatten()

    pqrst = analyze_pqrst(signal)
    if pqrst:
        print(f"Heart Rate:    {pqrst['heart_rate']} bpm")
        print(f"QRS Duration:  {pqrst['qrs_duration']} ms")
        print(f"PR Interval:   {pqrst['pr_interval']} ms")
        print(f"QT Interval:   {pqrst['qt_interval']} ms")

    disease = detect_disease_from_pqrst(pqrst, 0)
    print("\nDiseases found:")
    for d in disease['diseases']:
        print(f"  {d}")
    print(f"Severity: {disease['severity']}")
    print("Recommendations:")
    for r in disease['recommendations']:
        print(f"  → {r}")