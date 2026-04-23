from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_processing import add_features, load_and_clean, split_data
from src.inference import predict_patient
from src.model_training import train_and_save


PROJECT_ROOT = Path(__file__).parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "oasis_longitudinal_demographics.xlsx"


st.set_page_config(
    page_title="Dementia Prediction Pipeline",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_dataset(data_path: str) -> pd.DataFrame:
    df = load_and_clean(data_path)
    df = add_features(df)
    return df


def _feature_columns_from_models() -> list[str] | None:
    try:
        import joblib

        cols_path = PROJECT_ROOT / "models" / "feature_columns.pkl"
        if cols_path.exists():
            cols = joblib.load(str(cols_path))
            if isinstance(cols, list) and all(isinstance(c, str) for c in cols):
                return cols
    except Exception:
        return None
    return None


def _render_prediction(result: dict) -> None:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("Prediction", str(result.get("prediction", "")))
    with c2:
        st.metric("Confidence", f"{result.get('confidence', 0):.3f}%")

    st.subheader("Top contributing factors")
    for s in result.get("top_signals", []):
        st.write(f"- {s}")

    st.subheader("Other contributing factors")
    for s in result.get("other_factors", []):
        st.write(f"- {s}")


st.title("Dementia Prediction Model — Training & Inference UI")
st.caption("Run training, sample inference, or custom patient predictions without editing `main.py`.")

with st.sidebar:
    st.subheader("Dataset")
    data_path = st.text_input("Excel path", value=str(DEFAULT_DATA_PATH))
    st.write("If you move the dataset, update this path.")

    st.subheader("Models")
    st.write("Training writes to `models/` and `outputs/`.")


tabs = st.tabs(["Train models", "Inference (sample)", "Custom patient"])


with tabs[0]:
    st.subheader("Train models")
    st.write("This trains the two-stage cascade and saves artifacts into `models/`.")

    if not Path(data_path).exists():
        st.error(f"Dataset not found at: {data_path}")
    else:
        df = load_dataset(data_path)
        X, y, X_train, X_test, y_train, y_test = split_data(df)

        st.write("Dataset preview")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Train and save models", type="primary"):
            with st.spinner("Training models..."):
                train_and_save(X, X_train, X_test, y_train, y_test)
            st.success("Training complete. Models saved to `models/` and plots to `outputs/`.")


with tabs[1]:
    st.subheader("Inference on a dataset row")
    st.write("Pick a row index from the dataset and run the model inference with SHAP explanations.")

    if not Path(data_path).exists():
        st.error(f"Dataset not found at: {data_path}")
    else:
        df = load_dataset(data_path)
        X = df.drop(columns=["Group", "Subject ID"])

        idx = st.slider("Row index", min_value=0, max_value=len(X) - 1, value=min(3, len(X) - 1))
        st.write("Selected row (features)")
        st.dataframe(X.iloc[[idx]], use_container_width=True)

        if st.button("Run inference", type="primary"):
            with st.spinner("Predicting..."):
                result = predict_patient(X.iloc[[idx]])
            _render_prediction(result)


with tabs[2]:
    st.subheader("Custom patient")
    st.write("Enter patient measurements. Derived features are computed automatically.")

    cols = _feature_columns_from_models()
    if cols is None:
        st.info("Could not read `models/feature_columns.pkl` yet. Train models first for best compatibility.")

    with st.form("custom_patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            visit = st.number_input("Visit", min_value=1, value=1, step=1)
            mr_delay = st.number_input("MR Delay", min_value=0, value=0, step=1)
            gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
            age = st.number_input("Age", min_value=18, max_value=120, value=75, step=1)
        with c2:
            educ = st.number_input("EDUC (years)", min_value=0, max_value=40, value=18, step=1)
            ses = st.number_input("SES (1-5)", min_value=1, max_value=5, value=2, step=1)
            mmse = st.number_input("MMSE (0-30)", min_value=0, max_value=30, value=28, step=1)
            etiv = st.number_input("eTIV", min_value=0.0, value=1506.0, step=1.0)
        with c3:
            nwbv = st.number_input("nWBV", min_value=0.0, value=0.709, step=0.001, format="%.3f")
            asf = st.number_input("ASF", min_value=0.0, value=1.207, step=0.001, format="%.3f")

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        patient_data = {
            "Visit": int(visit),
            "MR Delay": int(mr_delay),
            "M/F": 1 if gender == "Male" else 0,
            "Age": float(age),
            "EDUC": float(educ),
            "SES": float(ses),
            "MMSE": float(mmse),
            "eTIV": float(etiv),
            "nWBV": float(nwbv),
            "ASF": float(asf),
        }

        custom_sample = pd.DataFrame([patient_data])
        custom_sample["Brain_Ratio"] = custom_sample["nWBV"] / custom_sample["eTIV"]
        custom_sample["Age_nWBV"] = custom_sample["Age"] * custom_sample["nWBV"]
        custom_sample["nWBV_diff"] = 0.0
        custom_sample["MMSE_diff"] = 0.0

        if cols is not None:
            missing = [c for c in cols if c not in custom_sample.columns]
            if missing:
                st.error(f"Custom patient is missing required features: {missing}")
            else:
                custom_sample = custom_sample[cols]
                with st.spinner("Predicting..."):
                    result = predict_patient(custom_sample)
                _render_prediction(result)
        else:
            with st.spinner("Predicting..."):
                result = predict_patient(custom_sample)
            _render_prediction(result)

