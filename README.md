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

### Run the UI (recommended)

Launch the Streamlit app:

```python
python -m streamlit run app.py
```

This gives you buttons to:

- Train models
- Run inference on a dataset row
- Predict a custom patient

### 1.Train Models

```python
MODE = "train"   # change to "train" or "inference" or "custom_patient"
```

then run the main

```python
python main.py
```

Trains and saves XGBoost models to `models/` directory.

### 2.Generate SHAP Analysis

```python
cd src
```

```python
python shap_analysis.py
```

Generates:

- 4 publication-quality visualizations (PNG)
- Feature importance rankings (CSV)
- Detailed contribution statistics (CSV)

### 3.Run Inference

Set `MODE = "inference"` to get predictions with SHAP explanations for sample data.

```python
MODE = "inference"   # change to "train" or "inference" or "custom_patient"
```

then run the main

```python
python main.py
```

### 4.Test custom patient data

```python
MODE = "custom_patient"   # change to "train" or "inference" or "custom_patient"
```

fill the patient details , then run

```python
python main.py
```

Patient Data example

patient_data = {
'Visit': 1, # Visit number
'MR Delay': 0, # Days since last visit (0 for first visit)
'M/F': 1, # 1=Male, 0=Female
'Age': 75, # Patient age in years
'EDUC': 18, # Years of education
'SES': 2, # Socioeconomic status (1-5)
'MMSE': 28, # Mini-mental state exam score (0-30)
'eTIV': 1506.0, # Estimated total intracranial volume
'nWBV': 0.709, # Normalized whole brain volume
'ASF': 1.207, # Atlas scaling factor
}

### 5.Compare ML Models

```python
cd src
```

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
