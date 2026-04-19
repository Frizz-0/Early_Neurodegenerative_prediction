from src.data_processing import load_and_clean
from src.data_processing import add_features
from src.data_processing import split_data
from src.model_training import train_and_save
from src.inference import predict_patient, predict_batch

import os

# =========================
# CONFIG
# =========================
MODE = "inference"   # change to "train" or "inference"

DATA_PATH = "data/oasis_longitudinal_demographics.xlsx"


# =========================
# LOAD DATA
# =========================
df = load_and_clean(DATA_PATH)
df = add_features(df)
X,y,X_train, X_test, y_train,y_test = split_data(df)

# =========================
# TRAIN MODE
# =========================
if MODE == "train":
    print("Training models...")
    train_and_save(X,X_train, y_train)
    print("Models trained and saved in /models folder")


# =========================
# INFERENCE MODE
# =========================
elif MODE == "inference":
    print("Running inference...\n")

    # SINGLE prediction
    sample = X.iloc[[0]]
    result = predict_patient(sample)

    # print("=" * 60)
    # print("SINGLE PREDICTION WITH SHAP EXPLANATIONS")
    print("=" * 60)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%\n")
    
    print("Top Contributing Factors:")
    for signal in result['top_signals']:
        print(f"  • {signal}")
    
    print("\nOther Contributing Factors:")
    for factor in result['other_factors']:
        print(f"  • {factor}")
    
    # print("\n" + "=" * 60)
    # print("BATCH PREDICTIONS (first 5):")
    # print("=" * 60)
    # preds = predict_batch(X)
    # for i, pred in enumerate(preds[:5]):
    #     print(f"  Sample {i}: {pred}")


else:
    print("Invalid MODE. Use 'train' or 'inference'")