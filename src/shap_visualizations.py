
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Get project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

def load_shap_data(model_dir="models"):
    
    # Convert to absolute path if relative
    if not os.path.isabs(model_dir):
        model_dir = PROJECT_ROOT / model_dir
    else:
        model_dir = Path(model_dir)
    
    # Allow either legacy names (model2.pkl) or stage naming (model_stage2.pkl)
    model2_path = model_dir / "model2.pkl"
    if not model2_path.exists():
        stage2_path = model_dir / "model_stage2.pkl"
        if stage2_path.exists():
            model2_path = stage2_path

    required_files = [model2_path, model_dir / "imputer.pkl", model_dir / "scaler.pkl", model_dir / "le.pkl"]
    missing = [str(f) for f in required_files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")
    
    model2 = joblib.load(str(model2_path))
    imputer = joblib.load(str(model_dir / "imputer.pkl"))
    scaler = joblib.load(str(model_dir / "scaler.pkl"))
    le = joblib.load(str(model_dir / "le.pkl"))
    
    return model2, imputer, scaler, le


def prepare_shap_data(X_df, imputer, scaler):
    """Preprocess data for SHAP analysis."""
    X_processed = imputer.transform(X_df)
    X_processed = scaler.transform(X_processed)
    return pd.DataFrame(X_processed, columns=X_df.columns)


def get_shap_explainer(model, X_processed):
    """Create SHAP explainer."""
    return shap.Explainer(model), model.predict(X_processed)


def plot_force_plot(explainer, shap_values, X_processed, sample_idx=0):
    """Plot SHAP force plot for a single sample."""
    return shap.force_plot(
        explainer.expected_value,
        shap_values.values[sample_idx],
        X_processed.iloc[sample_idx],
        matplotlib=True
    )


def plot_summary_plot(explainer, shap_values, X_processed, plot_type="bar"):

    if plot_type == "bar":
        shap.summary_plot(shap_values, X_processed, plot_type="bar")
    else:
        shap.summary_plot(shap_values, X_processed)
    plt.tight_layout()


def plot_dependence_plot(explainer, shap_values, X_processed, feature_name):
    shap.dependence_plot(feature_name, shap_values.values, X_processed)
    plt.tight_layout()


def plot_waterfall_plot(explainer, shap_values, X_processed, sample_idx=0):
    return shap.plots.waterfall(
        shap.Explanation(
            values=shap_values.values[sample_idx],
            base_values=explainer.expected_value,
            data=X_processed.iloc[sample_idx],
            feature_names=X_processed.columns.tolist()
        )
    )


def get_feature_importance(shap_values, feature_names, top_n=10):
   
    # Accept SHAP as Explanation, ndarray, or list-of-arrays (multiclass TreeExplainer)
    if hasattr(shap_values, "values"):
        sv = shap_values.values
    else:
        sv = shap_values

    if isinstance(sv, list):
        # List of (n_samples, n_features) per class -> (n_samples, n_features, n_classes)
        sv = np.stack(sv, axis=-1)

    if len(sv.shape) == 3:
        # Multiclass: (n_samples, n_features, n_classes)
        # Take mean absolute value across all samples and classes
        mean_shap = np.abs(sv).mean(axis=(0, 2))
    else:
        # Binary: (n_samples, n_features)
        mean_shap = np.abs(sv).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)


def analyze_feature_contribution(explainer, shap_values, X_processed, feature_name):
   
    feature_idx = X_processed.columns.tolist().index(feature_name)
    # Accept SHAP as Explanation, ndarray, or list-of-arrays (multiclass TreeExplainer)
    if hasattr(shap_values, "values"):
        sv = shap_values.values
    else:
        sv = shap_values

    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    
    if len(sv.shape) == 3:
        # Multiclass: (n_samples, n_features, n_classes)
        # Take mean across classes for each sample
        contributions = np.abs(sv[:, feature_idx, :]).mean(axis=1)
    else:
        # Binary: (n_samples, n_features)
        contributions = sv[:, feature_idx]
    
    return {
        'feature': feature_name,
        'mean_contribution': np.mean(contributions),
        'median_contribution': np.median(contributions),
        'std_contribution': np.std(contributions),
        'max_positive': np.max(contributions),
        'max_negative': np.min(contributions),
        'positive_count': np.sum(contributions > 0),
        'negative_count': np.sum(contributions < 0)
    }
