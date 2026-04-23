from src.data_processing import load_and_clean
from src.data_processing import add_features
from src.data_processing import split_data
from src.model_training import train_and_save
from src.inference import predict_patient

import os
import pandas as pd

MODE = "inference"   # change to "train", "inference", or "custom_patient"

DATA_PATH = "data/oasis_longitudinal_demographics.xlsx"


df = load_and_clean(DATA_PATH)
df = add_features(df)
X,y,X_train, X_test, y_train,y_test = split_data(df)

if MODE == "train":
    print("Training models...")
    train_and_save(X, X_train, X_test, y_train, y_test)
    print("\nModels trained and saved in /models folder")


elif MODE == "inference":
    print("Running inference...\n")

    # SINGLE prediction
    sample = X.iloc[[3]]
    result = predict_patient(sample)

    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%\n")
    
    print("Top Contributing Factors:")
    for signal in result['top_signals']:
        print(f"  • {signal}")
    
    print("\nOther Contributing Factors:")
    for factor in result['other_factors']:
        print(f"  • {factor}")


elif MODE == "custom_patient":
    print("Running inference on custom patient data...\n")
    
    patient_data = {
        'Visit': 1,                  # Visit number
        'MR Delay': 0,               # Days since last visit (0 for first visit)
        'M/F': 1,                    # 1=Male, 0=Female
        'Age': 75,                   # Patient age in years
        'EDUC': 18,                  # Years of education
        'SES': 2,                    # Socioeconomic status (1-5)
        'MMSE': 28,                  # Mini-mental state exam score (0-30)
        'eTIV': 1506.0,              # Estimated total intracranial volume
        'nWBV': 0.709,               # Normalized whole brain volume
        'ASF': 1.207,                # Atlas scaling factor
    }
    

    custom_sample = pd.DataFrame([patient_data])
    
    custom_sample['Brain_Ratio'] = custom_sample['nWBV'] / custom_sample['eTIV']
    custom_sample['Age_nWBV'] = custom_sample['Age'] * custom_sample['nWBV']
    custom_sample['nWBV_diff'] = 0  # Longitudinal features (0 for single sample)
    custom_sample['MMSE_diff'] = 0
    
    custom_sample = custom_sample[X.columns]
    
    result = predict_patient(custom_sample)

    print(f"Patient Profile:")
    print(f"  Age: {patient_data['Age']} years")
    print(f"  Gender: {'Male' if patient_data['M/F'] == 1 else 'Female'}")
    print(f"  MMSE Score: {patient_data['MMSE']}/30")
    
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%\n")
    
    print("Top Contributing Factors:")
    for signal in result['top_signals']:
        print(f"  - {signal}")
    
    print("\nOther Contributing Factors:")
    for factor in result['other_factors']:
        print(f"  - {factor}")


else:
    print("Invalid MODE. Use 'train', 'inference', or 'custom_patient'")