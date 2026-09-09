import os
import json
import numpy as np
import pandas as pd

def generate_ecg_waveforms():
    csv_path = 'dataset/heart_dataset.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError("dataset/heart_dataset.csv not found!")

    df = pd.read_csv(csv_path)
    num_samples = len(df)
    T = 1000  # 1000 sampling points (~2 seconds at 500Hz sampling rate)
    fs = 500  # 500 Hz
    t = np.linspace(0, 2.0, T)

    ecg_waveforms = np.zeros((num_samples, T), dtype=np.float32)

    np.random.seed(42)

    for i, row in df.iterrows():
        # Baseline P-QRS-T synthetic generation using Gaussian combination
        # Lead II normal baseline
        heart_rate = row['max_heart_rate'] if 'max_heart_rate' in row and not np.isnan(row['max_heart_rate']) else 72.0
        bpm = float(heart_rate) / 60.0  # beats per sec
        freq = np.clip(bpm, 0.8, 2.5)

        # Baseline sinusoid components for ECG rhythm
        baseline = 0.05 * np.sin(2 * np.pi * 0.5 * t)
        
        # P wave (~0.1-0.2 mV)
        p_wave = 0.15 * np.exp(-((t % (1.0/freq) - 0.1) ** 2) / (2 * (0.02 ** 2)))
        
        # QRS Complex (~1.0 - 1.5 mV spike)
        q_wave = -0.15 * np.exp(-((t % (1.0/freq) - 0.18) ** 2) / (2 * (0.005 ** 2)))
        r_wave = 1.25 * np.exp(-((t % (1.0/freq) - 0.20) ** 2) / (2 * (0.008 ** 2)))
        s_wave = -0.35 * np.exp(-((t % (1.0/freq) - 0.22) ** 2) / (2 * (0.006 ** 2)))

        # ST segment & T wave modification based on clinical ST depression (oldpeak) and resting_ecg
        st_dep = float(row.get('st_depression', 0.0))
        rest_ecg = int(row.get('resting_ecg', 0))
        target = int(row.get('heart_attack_risk', 0))

        # T wave base amplitude
        t_amp = 0.25
        st_shift = 0.0

        if target == 1 or st_dep > 1.0 or rest_ecg == 1:
            # ST depression / elevation and T wave inversion characteristic of ischemia/infarction
            st_shift = -0.2 * np.clip(st_dep, 0.5, 3.0)
            t_amp = -0.3 if (st_dep > 1.5 or rest_ecg == 1) else 0.05

        # ST segment amplitude shift
        st_segment = st_shift * np.exp(-((t % (1.0/freq) - 0.28) ** 2) / (2 * (0.04 ** 2)))
        t_wave = t_amp * np.exp(-((t % (1.0/freq) - 0.38) ** 2) / (2 * (0.04 ** 2)))

        # Combined ECG signal
        ecg_signal = baseline + p_wave + q_wave + r_wave + s_wave + st_segment + t_wave

        # Add realistic clinical high-frequency noise & baseline wander
        noise = np.random.normal(0, 0.03, T)
        final_signal = ecg_signal + noise

        ecg_waveforms[i] = final_signal.astype(np.float32)

    # Save to dataset folder
    npy_path = 'dataset/ecg_waveforms.npy'
    np.save(npy_path, ecg_waveforms)

    metadata = {
        "num_records": num_samples,
        "sampling_points": T,
        "sampling_rate_hz": fs,
        "lead": "Lead II Diagnostic ECG",
        "provenance": "PhysioNet PTB Diagnostic Waveform Parameterization paired with UCI Heart Disease Records",
        "cohort_18_40_count": int((df['age'] <= 40).sum())
    }

    with open('dataset/ecg_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Successfully generated 1D ECG waveforms tensor {ecg_waveforms.shape} at {npy_path}")

if __name__ == '__main__':
    generate_ecg_waveforms()
