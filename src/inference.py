import pandas as pd
import numpy as np
import joblib
import shap
import os

# Global model cache
_models_cache = {}

def _load_models():
    """Lazy load models on first use."""
    global _models_cache
    
    if _models_cache:
        return _models_cache
    
    required_files = ["models/model1.pkl", "models/model2.pkl", "models/imputer.pkl", 
                     "models/scaler.pkl", "models/le.pkl"]
    
    missing = [f for f in required_files if not os.path.exists(f)]
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
        'model1': joblib.load("models/model1.pkl"),
        'model2': joblib.load("models/model2.pkl"),
        'imputer': joblib.load("models/imputer.pkl"),
        'scaler': joblib.load("models/scaler.pkl"),
        'le': joblib.load("models/le.pkl"),
        'explainer1': shap.Explainer(joblib.load("models/model1.pkl")),
        'explainer2': shap.Explainer(joblib.load("models/model2.pkl"))
    }

    return _models_cache

# =========================
# FEATURE MEANING MAP
# =========================
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

# =========================
# SHAP → TEXT EXPLANATION
# =========================
def generate_text_explanation(shap_values, input_df, feature_names, class_idx, top_n=5):
    """Convert SHAP values to human-readable explanations."""
    
    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        # Multiclass: list of arrays
        shap_vals = shap_values[class_idx][0]
    else:
        # Get the SHAP values - handle both binary and multiclass
        sv = shap_values.values
        
        if len(sv.shape) == 3:
            # Multiclass format: (n_samples, n_features, n_classes)
            shap_vals = sv[0, :, class_idx]
        elif len(sv.shape) == 2:
            # Could be (n_samples, n_classes) or (n_samples, n_features)
            if sv.shape[1] == len(feature_names):
                # (n_samples, n_features) - binary classification
                shap_vals = sv[0]
            else:
                # (n_samples, n_classes)
                shap_vals = sv[0, class_idx]
        else:
            # (n_samples,) or (n_features,)
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
            phrase = "→ higher risk"
        else:
            phrase = "→ lower risk"

        return f"{name} ({value}) {phrase}"

    top_sentences = [format_feature(f) for f in top_features]
    other_sentences = [format_feature(f) for f in other_features]

    return top_sentences, other_sentences

def preprocess(df):
    models = _load_models()
    X = models['imputer'].transform(df)
    X = models['scaler'].transform(X)

    return pd.DataFrame(X, columns=df.columns)

# 🔹 SINGLE PREDICTION WITH SHAP
def predict_patient(df):
    models = _load_models()
    X = preprocess(df)
    feature_names = X.columns.tolist()

    # =====================
    # Stage 1: Binary Model
    # =====================
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

    # =====================
    # Stage 2: Multiclass Model
    # =====================
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

# 🔹 BATCH PREDICTIONS
def predict_batch(df):
    models = _load_models()
    X = preprocess(df)

    prob1 = models['model1'].predict_proba(X)
    pred1 = np.argmax(prob1, axis=1)

    prob2 = models['model2'].predict_proba(X)
    pred2 = np.argmax(prob2, axis=1)

    results = []
    for i in range(len(df)):
        if pred1[i] == 0:
            results.append("Nondemented")
        else:
            results.append(models['le'].inverse_transform([pred2[i]])[0])

    return results

