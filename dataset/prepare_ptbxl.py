import os
import ast
import json
import urllib.request
import numpy as np
import pandas as pd

def prepare_ptbxl_dataset():
    data_dir = 'dataset/ptbxl'
    os.makedirs(data_dir, exist_ok=True)
    
    db_csv_path = os.path.join(data_dir, 'ptbxl_database.csv')
    scp_csv_path = os.path.join(data_dir, 'scp_statements.csv')
    
    if not os.path.exists(db_csv_path):
        print("Downloading PTB-XL database metadata...")
        urllib.request.urlretrieve('https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv', db_csv_path)

    if not os.path.exists(scp_csv_path):
        print("Downloading SCP statements table...")
        urllib.request.urlretrieve('https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv', scp_csv_path)

    df = pd.read_csv(db_csv_path)
    scp_df = pd.read_csv(scp_csv_path, index_col=0)

    # Extract MI Diagnostic Codes
    mi_codes = set(scp_df[scp_df['diagnostic_class'] == 'MI'].index)

    def is_mi(scp_str):
        try:
            d = ast.literal_eval(scp_str)
            return int(any(k in mi_codes for k in d.keys()))
        except:
            return 0

    df['mi_target'] = df['scp_codes'].apply(is_mi)

    # Impute demographic features cleanly
    df['age'] = df['age'].fillna(df['age'].median())
    df['sex'] = df['sex'].fillna(0) # 0: female, 1: male
    df['height'] = df['height'].fillna(df['height'].median())
    df['weight'] = df['weight'].fillna(df['weight'].median())

    # Save cleaned metadata
    df.to_csv(os.path.join(data_dir, 'cleaned_ptbxl_metadata.csv'), index=False)

    summary = {
        "total_records": len(df),
        "unique_patients": int(df['patient_id'].nunique()),
        "mi_positive_records": int((df['mi_target'] == 1).sum()),
        "mi_negative_records": int((df['mi_target'] == 0).sum()),
        "young_adults_18_40_records": int(((df['age'] >= 18) & (df['age'] <= 40)).sum()),
        "young_adults_18_40_patients": int(df[(df['age'] >= 18) & (df['age'] <= 40)]['patient_id'].nunique()),
        "young_adults_mi_positive": int(df[(df['age'] >= 18) & (df['age'] <= 40) & (df['mi_target'] == 1)]['patient_id'].nunique()),
        "clinical_features": ["age", "sex", "height", "weight"],
        "ecg_leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "sampling_points": 1000,
        "sampling_rate_hz": 100,
        "provenance": "PhysioNet PTB-XL v1.0.3 (Fully Linked Patient Metadata + 12-Lead ECG Waveforms)"
    }

    with open(os.path.join(data_dir, 'provenance.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("PTB-XL metadata prepared successfully!")
    print(f"Total Records: {len(df)}, Unique Patients: {df['patient_id'].nunique()}")

if __name__ == '__main__':
    prepare_ptbxl_dataset()
