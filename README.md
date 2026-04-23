# Dementia Prediction Model with SHAP Explainability

A machine learning pipeline for dementia prediction and classification using XGBoost with SHAP-based model interpretability.

## Overview

This project implements a two-stage cascade classifier:

1. **Binary Stage**: Classifies patients as Demented or Nondemented
2. **Multiclass Stage**: For demented patients, distinguishes between Demented and Converted (cognitive decline)

The model includes SHAP (SHapley Additive exPlanations) integration for full model interpretability and prediction explanations.

## Project Structure

```
├── main.py                      # Entry point (train/inference)
├── train_models.py              # Standalone model training
├── shap_analysis.py             # SHAP analysis and visualizations
├── model_comparison.py          # Compare 7 ML algorithms
├── src/
│   ├── data_processing.py       # Data loading and feature engineering
│   ├── model_training.py        # XGBoost training
│   ├── inference.py             # Production predictions with SHAP
│   └── shap_visualizations.py   # SHAP utility functions
├── data/                        # Dataset directory
├── models/                      # Trained models
└── outputs/                     # Generated visualizations and reports
```

## Requirements

- Python 3.12
- XGBoost, scikit-learn, pandas, numpy
- SHAP, matplotlib, seaborn
- joblib

## Quick Start

```python
pip install -r requirements.txt
```

### Train Models

```python
python train_models.py
```

Trains and saves XGBoost models to `models/` directory.

### Generate SHAP Analysis

```python
python shap_analysis.py
```

Generates:

- 4 publication-quality visualizations (PNG)
- Feature importance rankings (CSV)
- Detailed contribution statistics (CSV)

### Run Inference

Set `MODE = "inference"` to get predictions with SHAP explanations for sample data.

```python
MODE = "inference"   # change to "train" or "inference"
```

```python
python main.py
```

### Compare ML Models

```python
python model_comparison.py
```

Compares XGBoost against 6 alternative algorithms with confusion matrices and metrics.

## Key Features

- **Two-stage cascade**: Optimized binary + multiclass classification
- **SHAP explainability**: Understanding why predictions are made
- **Model comparisons**: Validates XGBoost choice over alternatives
- **Lazy model loading**: Efficient memory usage
- **Absolute path handling**: Works from any directory

## Output Files

Generated in `outputs/` directory:

- `01_shap_importance_bar.png` - Feature importance rankings
- `02_shap_values_distribution.png` - SHAP value distributions
- `03_shap_dependence_plots.png` - Feature-SHAP relationships
- `04_shap_sample_explanations.png` - Individual prediction explanations
- `shap_feature_importance.csv` - Feature rankings
- `shap_feature_contribution_stats.csv` - Detailed statistics
