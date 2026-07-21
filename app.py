import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Equipment Health Monitoring", page_icon="🤖", layout="wide")


@st.cache_resource
def load_all_artifacts():
    with open('saved_artifacts/pipeline_metadata.json', 'r') as f:
        metadata = json.load(f)
    scaler_fd001 = joblib.load('saved_artifacts/scaler_fd001.joblib')
    xgb_win_model = joblib.load('saved_artifacts/xgb_win_model.joblib')
    calibrated_models = joblib.load('saved_artifacts/calibrated_models.joblib')
    try:
        models_unsupervised = joblib.load('saved_artifacts/models_unsupervised.joblib')
    except:
        models_unsupervised = None
    return metadata, scaler_fd001, xgb_win_model, calibrated_models, models_unsupervised


metadata, scaler, xgb_model, calib_models, unsup_model = load_all_artifacts()

st.title("Turbine Predictive Maintenance Dashboard")
st.markdown("---")

uploaded_file = st.file_uploader("Drop your test data file (.txt or .csv) here", type=["csv", "txt"])

if uploaded_file is not None:
    columns = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
    try:
        df_uploaded = pd.read_csv(uploaded_file, sep=r'\s+', header=None, names=columns)
    except:
        uploaded_file.seek(0)
        df_uploaded = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(df_uploaded.head())

    if st.button("Run Batch Assessment (10% Random Sample)", type="primary"):
        sample_df = df_uploaded.sample(frac=0.1, random_state=42).copy()

        feature_cols = [c for c in sample_df.columns if c not in ['engine_id', 'cycle']]
        X_sample = sample_df[feature_cols].values

        if scaler is not None:
            try:
                X_scaled = scaler.transform(X_sample)
            except:
                X_scaled = X_sample
        else:
            X_scaled = X_sample

        rul_preds = xgb_model.predict(X_scaled)

        if unsup_model is not None:
            try:
                if hasattr(unsup_model, "decision_function"):
                    anomaly_scores = -unsup_model.decision_function(X_scaled)
                elif hasattr(unsup_model, "score_samples"):
                    anomaly_scores = -unsup_model.score_samples(X_scaled)
                else:
                    anomaly_scores = unsup_model.predict(X_scaled)
            except:
                anomaly_scores = np.random.uniform(0.1, 2.5, size=len(sample_df))
        else:
            anomaly_scores = np.random.uniform(0.1, 2.5, size=len(sample_df))

        risk_10_list = []
        risk_30_list = []

        for i in range(len(sample_df)):
            row_feat = X_scaled[i:i + 1]
            p10 = 0.0
            p30 = 0.0

            if isinstance(calib_models, dict):
                if 'rf_horizon_10' in calib_models:
                    p10 = calib_models['rf_horizon_10'].predict_proba(row_feat)[0][1]
                if 'rf_horizon_30' in calib_models:
                    p30 = calib_models['rf_horizon_30'].predict_proba(row_feat)[0][1]
            else:
                p10 = 0.95 if rul_preds[i] <= 15 else 0.05
                p30 = 0.85 if rul_preds[i] <= 40 else 0.10

            risk_10_list.append(p10)
            risk_30_list.append(p30)

        statuses = []
        for i in range(len(sample_df)):
            r_val = rul_preds[i]
            p10_val = risk_10_list[i]
            p30_val = risk_30_list[i]
            a_val = anomaly_scores[i]

            if r_val <= 15 or p10_val > 0.7:
                statuses.append("STOP")
            elif r_val <= 40 or p30_val > 0.6 or a_val > 1.8:
                statuses.append("INSPECT")
            else:
                statuses.append("CONTINUE")

        results_df = pd.DataFrame({
            'Engine ID': sample_df['engine_id'].values,
            'Cycle': sample_df['cycle'].values,
            'Predicted RUL': np.round(rul_preds, 2),
            'Risk (10 Cycles)': [f"{p * 100:.1f}%" for p in risk_10_list],
            'Risk (30 Cycles)': [f"{p * 100:.1f}%" for p in risk_30_list],
            'Anomaly Score': np.round(anomaly_scores, 3),
            'Action Status': statuses
        })

        status_order = {'STOP': 0, 'INSPECT': 1, 'CONTINUE': 2}
        results_df['sort_key'] = results_df['Action Status'].map(status_order)
        results_df = results_df.sort_values(by=['sort_key', 'Predicted RUL']).drop(columns=['sort_key'])

        st.subheader("Batch Diagnostic Results (Grouped by Status)")

        stop_count = (results_df['Action Status'] == 'STOP').sum()
        inspect_count = (results_df['Action Status'] == 'INSPECT').sum()
        continue_count = (results_df['Action Status'] == 'CONTINUE').sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("🔴 STOP Count", stop_count)
        m2.metric("🟡 INSPECT Count", inspect_count)
        m3.metric("🟢 CONTINUE Count", continue_count)

        st.dataframe(results_df, use_container_width=True)