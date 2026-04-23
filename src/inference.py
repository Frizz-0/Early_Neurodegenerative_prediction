import pandas as pd
import numpy as np
import joblib
import shap
import os
from pathlib import Path

# Global model cache
_models_cache = {}

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

def _load_models():
    """Lazy load models on first use."""
    global _models_cache
    
    if _models_cache:
        return _models_cache
    
    required_files = [
        MODELS_DIR / "model1.pkl",
        MODELS_DIR / "model2.pkl",
        MODELS_DIR / "imputer.pkl",
        MODELS_DIR / "scaler.pkl",
        MODELS_DIR / "le.pkl"
    ]
    
    missing = [str(f) for f in required_files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Models not found: {missing}\n"
            "Please run: python -c \"from src.data_processing import load_and_clean, add_features; "
            "from src.model_training import train_and_save; "
            "df = load_and_clean('data/oasis_longitudinal_demographics.xlsx'); "
            "df = add_features(df); "
            "X = df.drop(columns=['Group', 'Subject ID']); "
            "y = df['Group']; "
            "train_and_save(X, y)\""
        )
    
    _models_cache = {
        'model1': joblib.load(str(MODELS_DIR / "model1.pkl")),
        'model2': joblib.load(str(MODELS_DIR / "model2.pkl")),
        'imputer': joblib.load(str(MODELS_DIR / "imputer.pkl")),
        'scaler': joblib.load(str(MODELS_DIR / "scaler.pkl")),
        'le': joblib.load(str(MODELS_DIR / "le.pkl")),
        'explainer1': shap.Explainer(joblib.load(str(MODELS_DIR / "model1.pkl"))),
        'explainer2': shap.Explainer(joblib.load(str(MODELS_DIR / "model2.pkl")))
    }

    return _models_cache

# FEATURE MEANING MAP 
feature_meaning = {
    "MMSE": "cognitive score",
    "nWBV": "brain volume",
    "eTIV": "intracranial volume",
    "Age": "age",
    "M/F": "gender",
    "Brain_Ratio": "brain-to-skull ratio",
    "Age_nWBV": "age × brain volume",
    "nWBV_diff": "brain volume change",
    "MMSE_diff": "cognitive decline"
}


# SHAP
def generate_text_explanation(shap_values, input_df, feature_names, class_idx, top_n=5):
    
    if isinstance(shap_values, list):
        shap_vals = shap_values[class_idx][0]
    else:
        sv = shap_values.values
        
        if len(sv.shape) == 3:
            shap_vals = sv[0, :, class_idx]
        elif len(sv.shape) == 2:
            if sv.shape[1] == len(feature_names):
                shap_vals = sv[0]
            else:
                shap_vals = sv[0, class_idx]
        else:
            shap_vals = sv[0]

    feature_data = []
    for i, feature in enumerate(feature_names):
        feature_data.append({
            "feature": feature,
            "value": input_df.iloc[0, i],
            "impact": shap_vals[i]
        })

    # Sort by absolute importance
    feature_data = sorted(feature_data, key=lambda x: abs(x["impact"]), reverse=True)

    # Get top features
    top_features = feature_data[:3]
    other_features = feature_data[3:top_n]

    def format_feature(f):
        name = feature_meaning.get(f["feature"], f["feature"])
        value = round(f['value'], 2)

        if f["impact"] > 0:
            phrase = "-> higher risk"
        else:
            phrase = "-> lower risk"

        return f"{name} ({value}) {phrase}"

    top_sentences = [format_feature(f) for f in top_features]
    other_sentences = [format_feature(f) for f in other_features]

    return top_sentences, other_sentences

def preprocess(df):
    models = _load_models()
    X = models['imputer'].transform(df)
    X = models['scaler'].transform(X)

    return pd.DataFrame(X, columns=df.columns)

def predict_patient(df):
    models = _load_models()
    X = preprocess(df)
    feature_names = X.columns.tolist()

    # Stage 1: Binary Model
    prob1 = models['model1'].predict_proba(X)[0]
    pred1 = np.argmax(prob1)
    confidence1 = round(prob1[pred1] * 100, 2)

    shap_vals1 = models['explainer1'](X)

    # If Nondemented
    if pred1 == 0:
        top, others = generate_text_explanation(
            shap_vals1, df, feature_names, class_idx=1
        )

        return {
            "prediction": "Nondemented",
            "confidence": confidence1,
            "top_signals": top,
            "other_factors": others
        }


    # Stage 2: Multiclass Model
    prob2 = models['model2'].predict_proba(X)[0]
    pred2 = np.argmax(prob2)
    confidence2 = round(prob2[pred2] * 100, 2)

    final_label = models['le'].inverse_transform([pred2])[0]

    shap_vals2 = models['explainer2'](X)

    top, others = generate_text_explanation(
        shap_vals2, df, feature_names, class_idx=pred2
    )

    return {
        "prediction": final_label,
        "confidence": confidence2,
        "top_signals": top,
        "other_factors": others
    }

