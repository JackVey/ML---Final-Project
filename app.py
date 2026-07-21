import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os


st.set_page_config(
    page_title="Equipment Health Monitoring",
    page_icon="🤖",
    layout="wide"
)



@st.cache_resource
def load_all_artifacts():
    with open('saved_artifacts/pipeline_metadata.json', 'r') as f:
        metadata = json.load(f)

    scaler_fd001 = joblib.load('saved_artifacts/scaler_fd001.joblib')
    xgb_win_model = joblib.load('saved_artifacts/xgb_win_model.joblib')
    calibrated_models = joblib.load('saved_artifacts/calibrated_models.joblib')
    tuned_thresholds = joblib.load('saved_artifacts/tuned_thresholds.joblib')
    models_unsupervised = joblib.load('saved_artifacts/models_unsupervised.joblib')

    return metadata, scaler_fd001, xgb_win_model, calibrated_models, tuned_thresholds, models_unsupervised


try:
    metadata, scaler_fd001, xgb_model, cls_models, thresholds, unsup_models = load_all_artifacts()
    st.sidebar.success("All models and parameters loaded successfully.")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


st.sidebar.title("System Settings")
dataset_type = st.sidebar.selectbox("Select Dataset:", ["FD001", "FD002"])

st.sidebar.markdown("---")
st.sidebar.subheader("Pipeline Parameters")
st.sidebar.write(f"• Window Size (W): **{metadata['window_size_W']}**")
st.sidebar.write(f"• RUL Threshold: **{metadata['rul_threshold']}**")
st.sidebar.write(f"• Active Sensors Count: **{len(metadata['active_sensors'])}**")


st.title("Remaining Useful Life (RUL) Prediction and Anomaly Detection System")
st.markdown("This dashboard receives turbine sensor data and displays estimated remaining useful life and risk levels.")

st.markdown("---")

tab1, tab2 = st.tabs(["Scenario Test (Sample Data)", "Upload CSV File"])

with tab1:
    st.subheader("Run Quick Prediction on Sensor Window")

    if st.button("Run Prediction on Simulated Data", type="primary"):

        n_features = xgb_model.n_features_in_
        sample_input = np.random.randn(1, n_features)

        predicted_rul = float(xgb_model.predict(sample_input)[0])

        q95 = metadata["conformal_prediction_q95"]
        if predicted_rul <= 30:
            err = q95["near_failure"]
        elif predicted_rul <= 80:
            err = q95["mid_life"]
        else:
            err = q95["early_life"]

        lower_bound = max(0.0, predicted_rul - err)
        upper_bound = predicted_rul + err

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Remaining Useful Life (RUL)",
                value=f"{predicted_rul:.1f} cycles"
            )

        with col2:
            st.metric(
                label="Confidence Interval (95%)",
                value=f"[{lower_bound:.1f} to {upper_bound:.1f}]"
            )

        with col3:
            if predicted_rul <= metadata["rul_threshold"]:
                st.error("Status: Immediate overhaul required (High Risk)")
            else:
                st.success("Status: Optimal and normal operation")

        st.markdown("---")

        st.subheader("Sensor Health and Anomaly Analysis")

        col_a, col_b = st.columns(2)

        with col_a:
            if "isolation_forest" in unsup_models:
                iso_score = unsup_models["isolation_forest"].predict(sample_input)[0]
                if iso_score == -1:
                    st.warning("Anomaly: Unusual sensor behavior pattern detected (Isolation Forest).")
                else:
                    st.info("Sensor behavior is completely normal (Isolation Forest).")

        with col_b:
            if "lof" in unsup_models:
                lof_score = unsup_models["lof"].predict(sample_input)[0]
                if lof_score == -1:
                    st.warning("Local outlier anomaly identified (LOF).")
                else:
                    st.info("No local anomalies observed (LOF).")

with tab2:
    st.subheader("Upload CSV File Containing Sensor Data")
    uploaded_file = st.file_uploader("Drop your test data file here", type=["csv"])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df_uploaded.head(10))
        st.info("Batch processing code for the uploaded file will execute upon column matching.")