import pandas as pd
import numpy as np
import joblib
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import os


def train_and_save(X,X_train, y_train):
    """Train and save binary and multiclass XGBoost models with preprocessing."""
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)

    X_train_df = pd.DataFrame(X_train, columns=X.columns)

    # =========================
    # MODEL 1 (Binary)
    # =========================
    y_train_binary = np.where(y_train == "Nondemented", 0, 1)

    model1 = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )

    model1.fit(X_train_df, y_train_binary)

    # =========================
    # MODEL 2 (Multiclass)
    # =========================
    le = LabelEncoder()
    y_train_multi = le.fit_transform(y_train)

    model2 = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric="mlogloss"
    )

    model2.fit(X_train_df, y_train_multi)

    # =========================
    # SAVE EVERYTHING
    # =========================
    # ===== SAVE MODELS AND PREPROCESSORS =====
    joblib.dump(model1, "models/model1.pkl")
    joblib.dump(model2, "models/model2.pkl")
    joblib.dump(imputer, "models/imputer.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(le, "models/le.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

    print("Training complete and models saved.")
    return model1, model2, imputer, scaler, le
