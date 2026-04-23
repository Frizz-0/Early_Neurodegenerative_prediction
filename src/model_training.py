import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, classification_report)
from xgboost import XGBClassifier
import os

def train_and_save(X, X_train, X_test, y_train, y_test):
    
    # Ensure models and outputs directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    X_train_processed = imputer.fit_transform(X_train)
    X_train_processed = scaler.fit_transform(X_train_processed)
    
    X_test_processed = imputer.transform(X_test)
    X_test_processed = scaler.transform(X_test_processed)

    X_train_df = pd.DataFrame(X_train_processed, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_processed, columns=X.columns)

    # STAGE 1: DEMENTED vs REST (Converted + Nondemented)
    print("\nSTAGE 1: DEMENTED vs REST")
    print("_"*30)
    
    # Binary: Demented (1) vs Rest (0)
    y_train_stage1 = np.where(y_train == "Demented", 1, 0)
    y_test_stage1 = np.where(y_test == "Demented", 1, 0)

    model_stage1 = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model_stage1.fit(X_train_df, y_train_stage1)
    y_pred_stage1_train = model_stage1.predict(X_train_df)
    y_pred_stage1_test = model_stage1.predict(X_test_df)
    
    # Evaluate Stage 1
    evaluate_model(y_test_stage1, y_pred_stage1_test, "Stage 1 (Demented vs Rest)", 
                   ["Rest", "Demented"], "binary")

    # STAGE 2: CONVERTED vs NONDEMENTED
    # (Only on non-demented samples from training)
    print("\nSTAGE 2: CONVERTED vs NONDEMENTED")
    print("_"*30)

    
    # Get indices where training labels are not "Demented"
    train_mask = (y_train != "Demented").values
    X_train_stage2 = X_train_df[train_mask].reset_index(drop=True)
    y_train_stage2_raw = y_train[train_mask].reset_index(drop=True)
    
    # Binary labels: Converted (1) vs Nondemented (0)
    y_train_stage2 = np.where(y_train_stage2_raw == "Converted", 1, 0)
    
    # For evaluation: get test indices where true labels are not "Demented"
    test_mask = (y_test != "Demented").values
    X_test_stage2 = X_test_df[test_mask].reset_index(drop=True)
    y_test_stage2_raw = y_test[test_mask].reset_index(drop=True)
    y_test_stage2_true = np.where(y_test_stage2_raw == "Converted", 1, 0)

    model_stage2 = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model_stage2.fit(X_train_stage2, y_train_stage2)
    
    # Predict on test samples that Stage 1 classified as non-demented
    indices_for_stage2 = np.where(y_pred_stage1_test == 0)[0]
    X_test_for_stage2 = X_test_df.iloc[indices_for_stage2].reset_index(drop=True)
    y_pred_stage2 = model_stage2.predict(X_test_for_stage2)
    
    # Get the corresponding true labels for samples where Stage 1 predicted "Not Demented"
    y_test_stage2_true_for_eval = np.where(y_test.iloc[indices_for_stage2] == "Converted", 1, 0)
    
    # Evaluate Stage 2 (only on samples predicted as non-demented by Stage 1)
    if len(y_test_stage2_true_for_eval) > 0:
        evaluate_model(y_test_stage2_true_for_eval, y_pred_stage2, 
                      "Stage 2 (Converted vs Nondemented)", 
                      ["Nondemented", "Converted"], "binary")

    # FINAL PREDICTION COMBINATION
    print("\nFINAL HIERARCHICAL PREDICTIONS")
    print("_"*30)

    final_preds = np.empty(len(X_test_df), dtype=object)

    # Step 1: Assign "Demented" where Stage 1 predicts 1
    final_preds[y_pred_stage1_test == 1] = "Demented"

    # Step 2: For Stage 1 predictions of 0, use Stage 2 predictions
    for i, original_idx in enumerate(indices_for_stage2):
        if y_pred_stage2[i] == 1:
            final_preds[original_idx] = "Converted"
        else:
            final_preds[original_idx] = "Nondemented"

    # Evaluate final predictions
    print(f"\nFinal Classification Report (3-class):\n")
    print(classification_report(y_test, final_preds, zero_division=0))
    
    # Confusion matrix for final predictions
    cm_final = confusion_matrix(y_test, final_preds, labels=["Nondemented", "Converted", "Demented"])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Nondemented", "Converted", "Demented"],
                yticklabels=["Nondemented", "Converted", "Demented"],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Final Hierarchical Predictions', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    filename = "outputs/confusion_matrix_final_hierarchical.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nFinal confusion matrix saved to outputs/")
    plt.close()

    # SAVE MODELS AND PREPROCESSORS
    joblib.dump(model_stage1, "models/model_stage1.pkl")
    joblib.dump(model_stage2, "models/model_stage2.pkl")
    joblib.dump(imputer, "models/imputer.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

    print("\nTraining complete and models saved.")
    print("_"*30)
    
    return model_stage1, model_stage2, imputer, scaler


def evaluate_model(y_true, y_pred, model_name, class_labels, model_type):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{model_name} - Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels, 
                                zero_division=0))
    
    # confusion matrix visualization
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    filename = f"outputs/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to outputs/")
    plt.close()

