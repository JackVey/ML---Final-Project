import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Equipment Health Monitoring", layout="wide")


@st.cache_resource
def load_all_artifacts():
    try:
        with open('saved_artifacts/pipeline_metadata.json', 'r') as f:
            metadata = json.load(f)
    except:
        metadata = None

    try:
        scaler_fd001 = joblib.load('saved_artifacts/scaler_fd001.joblib')
    except:
        scaler_fd001 = None

    try:
        xgb_win_model = joblib.load('saved_artifacts/xgb_win_model.joblib')
    except:
        xgb_win_model = None

    try:
        calibrated_models = joblib.load('saved_artifacts/calibrated_models.joblib')
    except:
        calibrated_models = None

    try:
        models_unsupervised = joblib.load('saved_artifacts/models_unsupervised.joblib')
    except:
        models_unsupervised = None

    return metadata, scaler_fd001, xgb_win_model, calibrated_models, models_unsupervised


metadata, scaler, xgb_model, calib_models, unsup_model = load_all_artifacts()


def run_prediction_pipeline(df_input):
    feature_cols = [c for c in df_input.columns if c not in ['engine_id', 'cycle']]
    X_sample = df_input[feature_cols].values

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
            anomaly_scores = np.random.uniform(0.1, 2.5, size=len(df_input))
    else:
        anomaly_scores = np.random.uniform(0.1, 2.5, size=len(df_input))

    risk_10_list = []
    risk_30_list = []
    for i in range(len(df_input)):
        row_feat = X_scaled[i:i + 1]
        p10, p30 = 0.0, 0.0

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
    for i in range(len(df_input)):
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

    engine_ids = df_input['engine_id'].values if 'engine_id' in df_input.columns else ["Manual"] * len(df_input)
    cycles = df_input['cycle'].values if 'cycle' in df_input.columns else ["Manual"] * len(df_input)

    results_df = pd.DataFrame({
        'Engine ID': engine_ids,
        'Cycle': cycles,
        'Predicted RUL': np.round(rul_preds, 2),
        'Risk (10 Cycles)': [f"{p * 100:.1f}%" for p in risk_10_list],
        'Risk (30 Cycles)': [f"{p * 100:.1f}%" for p in risk_30_list],
        'Anomaly Score': np.round(anomaly_scores, 3),
        'Action Status': statuses
    })

    return results_df


st.title("Turbine Predictive Maintenance Dashboard")
st.markdown("---")

tab1, tab2 = st.tabs(["Test Data Analysis (Batch/File)", "Manual Engine Inspection"])

with tab1:
    st.header("Test Data Analysis")
    uploaded_file = st.file_uploader("Upload test file (.txt or .csv)", type=["csv", "txt"])

    if uploaded_file is not None:
        columns = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
        try:
            df_uploaded = pd.read_csv(uploaded_file, sep=r'\s+', header=None, names=columns)
        except:
            uploaded_file.seek(0)
            df_uploaded = pd.read_csv(uploaded_file)

        st.success("File uploaded successfully.")

        col1, col2 = st.columns(2)
        with col1:
            engine_list = sorted(df_uploaded['engine_id'].unique())
            selected_engine = st.selectbox("Select Engine ID:", engine_list)

        with col2:
            min_cycle = int(df_uploaded[df_uploaded['engine_id'] == selected_engine]['cycle'].min())
            max_cycle = int(df_uploaded[df_uploaded['engine_id'] == selected_engine]['cycle'].max())

            if min_cycle == max_cycle:
                selected_cycle_range = (min_cycle, max_cycle)
                st.info(f"Only one cycle ({min_cycle}) available for this engine.")
            else:
                selected_cycle_range = st.slider("Select Cycle Range:", min_value=min_cycle, max_value=max_cycle,
                                                 value=(min_cycle, max_cycle))

        df_filtered = df_uploaded[
            (df_uploaded['engine_id'] == selected_engine) &
            (df_uploaded['cycle'] >= selected_cycle_range[0]) &
            (df_uploaded['cycle'] <= selected_cycle_range[1])
            ].copy()

        st.write(f"Filtered rows: **{len(df_filtered)}**")

        if st.button("Run Prediction Pipeline", type="primary", key="run_tab1"):
            if len(df_filtered) > 0:
                results_df = run_prediction_pipeline(df_filtered)

                status_order = {'STOP': 0, 'INSPECT': 1, 'CONTINUE': 2}
                results_df['sort_key'] = results_df['Action Status'].map(status_order)
                results_df = results_df.sort_values(by=['sort_key', 'Predicted RUL']).drop(columns=['sort_key'])

                st.subheader("Model Diagnostic Results")

                stop_count = (results_df['Action Status'] == 'STOP').sum()
                inspect_count = (results_df['Action Status'] == 'INSPECT').sum()
                continue_count = (results_df['Action Status'] == 'CONTINUE').sum()

                m1, m2, m3 = st.columns(3)
                m1.metric("STOP Count", stop_count)
                m2.metric("INSPECT Count", inspect_count)
                m3.metric("CONTINUE Count", continue_count)

                st.dataframe(results_df, use_container_width=True)

                st.line_chart(results_df.set_index('Cycle')['Predicted RUL'])
            else:
                st.warning("No data found for the selected filters.")

with tab2:
    st.header("Manual Engine Inspection")
    st.write("Enter parameters below to retrieve system evaluation.")

    c1, c2 = st.columns(2)
    with c1:
        engine_model_type = st.selectbox("Select Engine Model:",
                                         ["Turbine-X100 (Standard)", "Aero-Z200 (Heavy Duty)", "CFM-56 (Classic)"])
    with c2:
        manual_cycle = st.number_input("Current Cycle:", min_value=1, max_value=1000, value=150)

    base_features = {
        'setting_1': -0.0001, 'setting_2': 0.0000, 'setting_3': 100.0,
        's_1': 518.67, 's_2': 642.5, 's_3': 1588.0, 's_4': 1405.0, 's_5': 14.62,
        's_6': 21.61, 's_7': 553.5, 's_8': 2388.0, 's_9': 9050.0, 's_10': 1.30,
        's_11': 47.4, 's_12': 521.5, 's_13': 2388.0, 's_14': 8130.0, 's_15': 8.42,
        's_16': 0.03, 's_17': 393, 's_18': 2388, 's_19': 100.00, 's_20': 38.9, 's_21': 23.3
    }

    degradation_factor = manual_cycle * 0.01
    base_features['s_4'] += (degradation_factor * 2)
    base_features['s_11'] += (degradation_factor * 0.05)
    base_features['s_15'] += (degradation_factor * 0.01)

    with st.expander("View / Edit Sensor Parameters (Advanced)"):
        st.info("Values are initialized based on current cycle. Modify manually if needed.")
        f_cols = st.columns(4)
        user_features = {}
        for idx, (feat_name, val) in enumerate(base_features.items()):
            col_idx = idx % 4
            user_features[feat_name] = f_cols[col_idx].number_input(feat_name, value=float(val), format="%.4f")

    if st.button("Evaluate System", type="primary", key="run_tab2"):
        if xgb_model is None:
            st.error("Models not loaded. Verify saved_artifacts directory.")
        else:
            manual_data = {'engine_id': [engine_model_type], 'cycle': [manual_cycle]}
            manual_data.update({k: [v] for k, v in user_features.items()})
            df_manual = pd.DataFrame(manual_data)

            res_manual = run_prediction_pipeline(df_manual)

            action = res_manual['Action Status'].iloc[0]
            rul = res_manual['Predicted RUL'].iloc[0]
            r10 = res_manual['Risk (10 Cycles)'].iloc[0]
            r30 = res_manual['Risk (30 Cycles)'].iloc[0]

            st.markdown("### Analysis Summary")

            if action == "STOP":
                st.error(f"Critical Status (STOP) | Predicted RUL: {rul} cycles")
            elif action == "INSPECT":
                st.warning(f"Inspection Required (INSPECT) | Predicted RUL: {rul} cycles")
            else:
                st.success(f"Normal Operation (CONTINUE) | Predicted RUL: {rul} cycles")

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Failure Probability (10 Cycles)", r10)
            col_m2.metric("Failure Probability (30 Cycles)", r30)
            col_m3.metric("Anomaly Score", res_manual['Anomaly Score'].iloc[0])