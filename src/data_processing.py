import pandas as pd
from sklearn.model_selection import train_test_split

# Data Cleaning

def load_and_clean(path):
    df = pd.read_excel(path)

    df = df.drop(columns=["MRI ID"])
    df = df.drop(columns=["CDR"])

    df["M/F"] = df["M/F"].map({"M": 1, "F": 0})

    if "Hand" in df.columns and df["Hand"].nunique() == 1:
        df = df.drop(columns=["Hand"])

    return df

# Feature Engineering

def add_features(df):
    df["Brain_Ratio"] = df["nWBV"] / df["eTIV"]
    df["Age_nWBV"] = df["Age"] * df["nWBV"]

    df = df.sort_values(["Subject ID", "Visit"])
    df["nWBV_diff"] = df.groupby("Subject ID")["nWBV"].diff().fillna(0)
    df["MMSE_diff"] = df.groupby("Subject ID")["MMSE"].diff().fillna(0)

    return df

def split_data(df):
    X = df.drop(columns=["Group", "Subject ID"])
    y = df["Group"]

    print(f"\nData shape: {X.shape}")
    print(f"Classes: {", ".join(y.unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X,y,X_train, X_test, y_train, y_test

