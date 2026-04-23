#!/usr/bin/env python

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, recall_score, precision_score, 
                            f1_score, accuracy_score, classification_report)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing import load_and_clean, add_features


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return metrics."""
    
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, 
            random_state=42, eval_metric='mlogloss', verbose=0
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'KNN (k=10)': KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f" Training {name}...")
        
        try:
            # Train
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            cm = confusion_matrix(y_test, y_pred)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'confusion_matrix': cm,
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'accuracy': accuracy
            }
            
            print(f"{name} trained successfully")
            print(f"\n Metrics for {name}:")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   F1-Score:  {f1:.4f}")
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
    
    return results


def plot_confusion_matrices(results, y_test_labels):
    """Plot confusion matrices for all models."""
    n_models = len(results)
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(results.items()):
        cm = data['confusion_matrix']
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
            xticklabels=y_test_labels, yticklabels=y_test_labels
        )
        axes[idx].set_title(f"{name}\nRecall: {data['recall']:.3f} | Acc: {data['accuracy']:.3f}")
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('outputs/model_comparison_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrices saved to outputs/model_comparison_confusion_matrices.png")
    plt.show()


def plot_metrics_comparison(results):
    """Plot comparison of key metrics across models."""
    metrics_data = {
        'Model': list(results.keys()),
        'Accuracy': [data['accuracy'] for data in results.values()],
        'Recall': [data['recall'] for data in results.values()],
        'Precision': [data['precision'] for data in results.values()],
        'F1-Score': [data['f1'] for data in results.values()]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['Accuracy', 'Recall', 'Precision', 'F1-Score']
    for idx, (ax, metric) in enumerate(zip(axes.flatten(), metrics)):
        ax.barh(metrics_df['Model'], metrics_df[metric], color='steelblue')
        ax.set_xlabel(metric)
        ax.set_xlim([0, 1])
        
        # Add value labels
        for i, v in enumerate(metrics_df[metric]):
            ax.text(v + 0.02, i, f'{v:.3f}', va='center')
        
        ax.set_title(f'{metric} Comparison')
    
    plt.tight_layout()
    plt.savefig('outputs/model_comparison_metrics.png', dpi=300, bbox_inches='tight')
    print("Metrics comparison saved to outputs/model_comparison_metrics.png")
    plt.show()
    
    return metrics_df


def print_summary_table(results):
    """Print summary table of all models."""
    summary_data = []
    
    for name, data in results.items():
        summary_data.append({
            'Model': name,
            'Accuracy': f"{data['accuracy']:.4f}",
            'Recall': f"{data['recall']:.4f}",
            'Precision': f"{data['precision']:.4f}",
            'F1-Score': f"{data['f1']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\n" + "="*80)
    print("[*] MODEL COMPARISON SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    return summary_df


def main():
    print("\n" + "="*80)
    print("[ML] MODEL COMPARISON SUITE")
    print("="*80)
    
    # Load and prepare data
    print("\n[*] Loading data...")
    df = load_and_clean("data/oasis_longitudinal_demographics.xlsx")
    df = add_features(df)
    
    X = df.drop(columns=["Group", "Subject ID"])
    y = df["Group"]
    
    print(f"   Data shape: {X.shape}")
    print(f"   Classes: {y.unique()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocess
    print("\n[*] Preprocessing...")
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()
    
    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=X.columns)
    
    X_test = imputer.transform(X_test)
    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Train and evaluate models
    print("\n" + "="*80)
    print("[*] TRAINING MODELS")
    print("="*80)
    
    results = train_and_evaluate_models(
        X_train, X_test, y_train_encoded, y_test_encoded
    )
    
    # Generate reports
    print("\n" + "="*80)
    print("[*] GENERATING REPORTS")
    print("="*80)
    
    # Summary table
    summary_df = print_summary_table(results)
    
    # Save summary to CSV
    summary_df.to_csv('outputs/model_comparison_summary.csv', index=False)
    print("\n[OK] Summary saved to outputs/model_comparison_summary.csv")
    
    # Confusion matrices
    print("\n[*] Plotting confusion matrices...")
    plot_confusion_matrices(results, le.classes_)
    
    # Metrics comparison
    print("\n[*] Plotting metrics comparison...")
    plot_metrics_comparison(results)
    
    # Detailed classification reports
    print("\n" + "="*80)
    print("[*] DETAILED CLASSIFICATION REPORTS")
    print("="*80)
    
    for name, data in results.items():
        print(f"\n{name}:")
        print(classification_report(y_test_encoded, data['predictions'], 
                                   target_names=le.classes_))
    
    print("\n" + "="*80)
    print("[OK] MODEL COMPARISON COMPLETE")
    print("="*80)
    print("\n[Output] Output files saved to outputs/:")
    print("   - model_comparison_summary.csv")
    print("   - model_comparison_confusion_matrices.png")
    print("   - model_comparison_metrics.png")
    print("\n[!] Key Findings:")
    
    best_model = max(results.items(), key=lambda x: x[1]['f1'])[0]
    best_recall = max(results.items(), key=lambda x: x[1]['recall'])[0]
    
    print(f"   - Best F1-Score: {best_model}")
    print(f"   - Best Recall: {best_recall}")
    print(f"   • XGBoost F1-Score: {results['XGBoost']['f1']:.4f}")
    print(f"   • XGBoost Recall: {results['XGBoost']['recall']:.4f}")


if __name__ == "__main__":
    main()
