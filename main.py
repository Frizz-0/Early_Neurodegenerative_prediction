from src.data_processing import load_and_clean
from src.data_processing import add_features
from src.data_processing import split_data
from src.model_training import train_and_save
from src.inference import predict_patient

import os

MODE = "inference"   # change to "train" or "inference"

DATA_PATH = "data/oasis_longitudinal_demographics.xlsx"


df = load_and_clean(DATA_PATH)
df = add_features(df)
X,y,X_train, X_test, y_train,y_test = split_data(df)

if MODE == "train":
    print("Training models...")
    train_and_save(X,X_train, y_train)
    print("Models trained and saved in /models folder")


elif MODE == "inference":
    print("Running inference...\n")

    # SINGLE prediction
    sample = X.iloc[[0]]
    result = predict_patient(sample)

    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%\n")
    
    print("Top Contributing Factors:")
    for signal in result['top_signals']:
        print(f"  • {signal}")
    
    print("\nOther Contributing Factors:")
    for factor in result['other_factors']:
        print(f"  • {factor}")


else:
    print("Invalid MODE. Use 'train' or 'inference'")