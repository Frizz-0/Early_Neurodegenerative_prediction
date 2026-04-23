#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from data_processing import load_and_clean, add_features
from shap_visualizations import (
    load_shap_data,
    prepare_shap_data,
    get_shap_explainer,
    get_feature_importance,
    analyze_feature_contribution
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Create outputs directory
os.makedirs("outputs", exist_ok=True)

print("\nSHAP MODEL INTERPRETABILITY ANALYSIS - COMPLETE SUITE")

print("\n Loading data and models...")
# Resolve data path relative to project root so running from any CWD works.
PROJECT_ROOT = Path(__file__).parent.parent
data_path = PROJECT_ROOT / "data" / "oasis_longitudinal_demographics.xlsx"
df = load_and_clean(str(data_path))
df = add_features(df)

X = df.drop(columns=["Group", "Subject ID"])
y = df["Group"]

print(f"    Data shape: {X.shape}")
print(f"    Classes: {y.unique().tolist()}")

# Load models
model2, imputer, scaler, le = load_shap_data(model_dir="models")
X_processed = prepare_shap_data(X, imputer, scaler)

print(f"    Model2 (Loaded): {type(model2).__name__}")
print(f"    Preprocessed shape: {X_processed.shape}")

expected_classes = sorted(y.unique().tolist())
loaded_n_classes = None
try:
    loaded_n_classes = int(getattr(model2, "n_classes_", model2.predict_proba(X_processed.iloc[[0]]).shape[1]))
except Exception:
    loaded_n_classes = None

# If the loaded model isn't truly multiclass, train a multiclass model for SHAP so the
# feature-importance plot matches the 3-class stacked bar style (as in the reference image).
model_for_shap = model2
le_for_shap = le
if loaded_n_classes is not None and loaded_n_classes != len(expected_classes):
    print(f"    Note: loaded model predicts {loaded_n_classes} classes, but data has {len(expected_classes)} classes.")
    print("    Training a multiclass XGBoost model for SHAP plots...")

    le_for_shap = LabelEncoder()
    y_enc = le_for_shap.fit_transform(y)

    model_for_shap = XGBClassifier(
        objective="multi:softprob",
        num_class=len(le_for_shap.classes_),
        eval_metric="mlogloss",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model_for_shap.fit(X_processed, y_enc)

print(f"    SHAP model: {type(model_for_shap).__name__} (classes={len(getattr(le_for_shap, 'classes_', []))})")


print("\n Computing SHAP values (this may take a moment)...")
explainer, predictions = get_shap_explainer(model_for_shap, X_processed)
shap_values = explainer(X_processed)

print(f"   - SHAP values computed: {shap_values.values.shape}")

# For multiclass models, SHAP may come back as 2D depending on explainer.
# To match the stacked-per-class bar plot (like the reference image),
# recompute per-class SHAP values when possible.
shap_values_for_bar = shap_values
shap_values_for_importance = shap_values
try:
    if hasattr(model2, "predict_proba") and getattr(model2, "n_classes_", 0) > 2 and shap_values.values.ndim == 2:
        tree_explainer = shap.TreeExplainer(model2)
        per_class = tree_explainer.shap_values(X_processed)  # list-of-arrays or (n, f, c)
        shap_values_for_bar = per_class
        shap_values_for_importance = per_class
        if isinstance(per_class, list):
            print(f"   - Recomputed per-class SHAP values: ({per_class[0].shape[0]}, {per_class[0].shape[1]}, {len(per_class)})")
        else:
            print(f"   - Recomputed per-class SHAP values: {per_class.shape}")
except Exception as e:
    print(f"   - Note: per-class SHAP recomputation skipped ({type(e).__name__}: {e})")


print("\n Analyzing feature importance...")

importance_df = get_feature_importance(
    shap_values_for_importance,
    X_processed.columns.tolist(),
    top_n=15
)

print("\n   Top 15 Features by Importance:")
print("   " + "-" * 50)
for idx, row in importance_df.iterrows():
    print(f"   {idx+1:2d}. {row['feature']:15s} -> {row['importance']:.4f}")

# Save to CSV
importance_df.to_csv("outputs/shap_feature_importance.csv", index=False)
print(f"\n    Saved to: outputs/shap_feature_importance.csv")

print("\n Generating visualizations...")
print("   1. Creating feature importance bar plot...")

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_for_bar, X_processed, plot_type="bar", show=False)
plt.title("Feature Importance - Mean Absolute SHAP Values", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/01_shap_importance_bar.png", dpi=300, bbox_inches='tight')
plt.close()
print("      - Saved: outputs/01_shap_importance_bar.png")

# Get top 4 features for later use
top_4_features = importance_df.head(4)['feature'].tolist()


print("   2. Creating SHAP values beeswarm plot...")

plt.figure(figsize=(14, 10))
# SHAP can be either 2D (n_samples, n_features) or 3D (n_samples, n_features, n_classes)
sv = shap_values.values
if sv.ndim == 3:
    shap_values_summed = np.abs(sv).mean(axis=2)  # Average across classes
else:
    shap_values_summed = sv
shap.summary_plot(
    shap.Explanation(values=shap_values_summed, base_values=np.mean(shap_values_summed),
                    data=X_processed.values, feature_names=X_processed.columns.tolist()),
    X_processed,
    plot_type="dot",
    show=False
)
plt.title("SHAP Value Distribution by Feature (Mean Across Classes)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/02_shap_values_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("      - Saved: outputs/02_shap_values_distribution.png")

print("   3. Creating dependence plots for top 4 features...")

# For multiclass, average SHAP values across classes; otherwise just use absolute SHAP values
if sv.ndim == 3:
    shap_values_mean = np.abs(sv).mean(axis=2)
else:
    shap_values_mean = np.abs(sv)
shap_values_mean_df = pd.DataFrame(shap_values_mean, columns=X_processed.columns)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, feature in enumerate(top_4_features):
    axes[idx].scatter(X_processed[feature], shap_values_mean_df[feature], alpha=0.6, s=50)
    axes[idx].set_xlabel(feature, fontweight='bold')
    axes[idx].set_ylabel('Mean |SHAP value|', fontweight='bold')
    axes[idx].set_title(f"Dependence: {feature}", fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle("Feature Dependence Plots - Top 4 Features", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/03_shap_dependence_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print("      - Saved: outputs/03_shap_dependence_plots.png")


print("   4. Creating sample predictions with SHAP details...")

# Get predictions for different classes
class_labels = list(getattr(le_for_shap, "classes_", []))
fig, axes = plt.subplots(1, len(class_labels), figsize=(max(16, 5 * len(class_labels)), 5))
if len(class_labels) == 1:
    axes = [axes]

class_indices = []
for class_label in class_labels:
    # Find first prediction of each class
    preds = model_for_shap.predict(X_processed)
    pred_classes = le_for_shap.inverse_transform(preds)
    idx = np.where(pred_classes == class_label)[0]
    if len(idx) > 0:
        class_indices.append(idx[0])

for ax, sample_idx in zip(axes, class_indices):
    pred_class = le_for_shap.inverse_transform(
        [np.argmax(model_for_shap.predict_proba(X_processed.iloc[[sample_idx]]))]
    )[0]
    
    # Get top 10 SHAP values for this sample
    sv_abs = np.abs(shap_values.values[sample_idx]).mean(axis=-1) if len(shap_values.values.shape) == 3 else np.abs(shap_values.values[sample_idx])
    top_indices = np.argsort(sv_abs)[-10:][::-1]
    
    top_features = X_processed.columns[top_indices].tolist()
    top_values = sv_abs[top_indices]
    
    ax.barh(range(len(top_features)), top_values, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'Sample {sample_idx}: {pred_class}', fontweight='bold')
    ax.invert_yaxis()

plt.suptitle("Top 10 SHAP Values for Sample Predictions", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/04_shap_sample_explanations.png", dpi=300, bbox_inches='tight')
plt.close()
print("       - Saved: outputs/04_shap_sample_explanations.png")


print("\n Generating detailed feature analysis...")

top_10_features = importance_df.head(10)['feature'].tolist()
analysis_data = []

for feature in top_10_features:
    stats = analyze_feature_contribution(explainer, shap_values, X_processed, feature)
    analysis_data.append(stats)

analysis_df = pd.DataFrame(analysis_data)
analysis_df.to_csv("outputs/shap_feature_contribution_stats.csv", index=False)

print("\n   Feature Contribution Statistics (Top 10):")

print(f"   {'Feature':<15} {'Mean':<10} {'Median':<10} {'Std':<10} {'Range':<25} {'Dir':<15}")

for _, row in analysis_df.iterrows():
    range_str = f"[{row['max_negative']:.3f}, {row['max_positive']:.3f}]"
    direction = f"{row['positive_count']}/{row['negative_count']}"
    print(f"   {row['feature']:<15} {row['mean_contribution']:<10.4f} "
          f"{row['median_contribution']:<10.4f} {row['std_contribution']:<10.4f} "
          f"{range_str:<25} {direction:<15}")

print("\n Saved: outputs/shap_feature_contribution_stats.csv")

print(" ANALYSIS SUMMARY")

print(f"""
KEY FINDINGS:
      
1. TOP 5 MOST IMPORTANT FEATURES:
   {chr(10).join([f'   {i+1}. {row["feature"]:<15} (importance: {row["importance"]:.4f})' 
                  for i, (_, row) in enumerate(importance_df.head(5).iterrows())])}

2. FEATURE IMPACT INTERPRETATION:
   - Positive SHAP values  - Increase prediction likelihood
   - Negative SHAP values  - Decrease prediction likelihood
   - Larger magnitude      - Stronger influence on prediction

3. MODEL INSIGHTS:
   - Total samples analyzed:    {X_processed.shape[0]}
   - Total features:            {X_processed.shape[1]}
   - Number of classes:         {len(class_labels)}
   - Classes:                   {', '.join(class_labels)}

""")

print("SHAP ANALYSIS COMPLETE!")


