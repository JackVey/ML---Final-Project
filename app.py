import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import random

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Jet Engine Early Warning System",
    page_icon="🛩️",
    layout="wide"
)


@st.cache_resource
def load_artifacts():
    artifacts = {}

    with open('saved_artifacts/available_datasets.json', 'r') as f:
        artifacts['available_datasets'] = json.load(f)

    for dataset in ['FD001', 'FD002']:
        artifacts[dataset] = {}
        ds_info = artifacts['available_datasets'][dataset]

        artifacts[dataset]['scaler'] = joblib.load(f'saved_artifacts/{ds_info["scaler"]}')
        artifacts[dataset]['xgb_model'] = joblib.load(f'saved_artifacts/{ds_info["xgb_model"]}')
        artifacts[dataset]['calibrated_models'] = joblib.load(f'saved_artifacts/{ds_info["calibrated_models"]}')
        artifacts[dataset]['anomaly_models'] = joblib.load(f'saved_artifacts/{ds_info["anomaly_models"]}')
        artifacts[dataset]['feature_info'] = joblib.load(f'saved_artifacts/{ds_info["feature_info"]}')
        artifacts[dataset]['window_info'] = joblib.load(f'saved_artifacts/{ds_info["window_info"]}')
        artifacts[dataset]['conformal_params'] = joblib.load(f'saved_artifacts/{ds_info["conformal_params"]}')
        artifacts[dataset]['tuned_thresholds'] = joblib.load(f'saved_artifacts/{ds_info["tuned_thresholds"]}')
        artifacts[dataset]['decision_params'] = joblib.load(f'saved_artifacts/{ds_info["decision_params"]}')
        artifacts[dataset]['rul_params'] = joblib.load(f'saved_artifacts/{ds_info["rul_params"]}')
        artifacts[dataset]['pct_scores_test'] = joblib.load(f'saved_artifacts/{ds_info["pct_scores_test"]}')
        artifacts[dataset]['pct_scores_val'] = joblib.load(f'saved_artifacts/{ds_info["pct_scores_val"]}')

        if 'feature_names' in ds_info:
            artifacts[dataset]['feature_names'] = joblib.load(f'saved_artifacts/{ds_info["feature_names"]}')
        else:
            artifacts[dataset]['feature_names'] = None

        with open(f'saved_artifacts/{ds_info["metadata"]}', 'r') as f:
            artifacts[dataset]['metadata'] = json.load(f)

        if dataset == 'FD002':
            artifacts[dataset]['scaler_dict'] = joblib.load(f'saved_artifacts/{ds_info["scaler_dict"]}')
            artifacts[dataset]['kmeans'] = joblib.load(f'saved_artifacts/{ds_info["kmeans"]}')

    return artifacts


@st.cache_data
def load_raw_data(dataset):
    col_names = ['engine_id', 'cycle'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    test_df = pd.read_csv(f'data/test_{dataset}.txt', sep=r'\s+', header=None, names=col_names)
    rul_df = pd.read_csv(f'data/RUL_{dataset}.txt', sep=r'\s+', header=None, names=['RUL_final'])
    return test_df, rul_df


@st.cache_data
def load_preprocessed_fd001():
    try:
        df = pd.read_csv('data/test_window_fd001_preprocessed.csv.gz', compression='gzip')
        rul_df = pd.read_csv('data/rul_final_fd001.csv.gz', compression='gzip')
        return df, rul_df
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading preprocessed data: {e}")
        return None, None


@st.cache_data
def load_preprocessed_fd002():
    try:
        df = pd.read_csv('data/test_window_fd002_preprocessed.csv.gz', compression='gzip')
        rul_df = pd.read_csv('data/rul_final_fd002.csv.gz', compression='gzip')
        return df, rul_df
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading preprocessed data: {e}")
        return None, None


def extract_multi_window_features_single_engine(engine_df, window_info, feature_cols):
    window_sizes = window_info['window_sizes']
    df_out = engine_df.copy()

    if len(df_out) == 0:
        return df_out

    grouped = df_out.groupby('engine_id')

    for W in window_sizes:
        for col in feature_cols:
            if col not in df_out.columns:
                continue

            rolling_obj = grouped[col].rolling(window=W, min_periods=1)

            df_out[f'{col}_roll_mean_W{W}'] = rolling_obj.mean().reset_index(level=0, drop=True)
            df_out[f'{col}_roll_std_W{W}'] = rolling_obj.std().reset_index(level=0, drop=True).fillna(0)
            df_out[f'{col}_roll_min_W{W}'] = rolling_obj.min().reset_index(level=0, drop=True)
            df_out[f'{col}_roll_max_W{W}'] = rolling_obj.max().reset_index(level=0, drop=True)

            slope_col = grouped[col].rolling(window=W, min_periods=2).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0,
                raw=True
            )
            df_out[f'{col}_slope_W{W}'] = slope_col.reset_index(level=0, drop=True).fillna(0)

    return df_out


def get_features_for_prediction(processed_df, selected_engine, selected_cycle, selected_dataset, artifacts):
    current_row = processed_df[(processed_df['engine_id'] == selected_engine) &
                               (processed_df['cycle'] == selected_cycle)]

    if len(current_row) == 0:
        return None

    feature_names = artifacts[selected_dataset]['feature_names']
    expected_features = feature_names['all_features']

    features = []
    for col in expected_features:
        if col in current_row.columns:
            val = current_row[col].values[0]
            if pd.isna(val):
                val = 0.0
            features.append(float(val))
        else:
            features.append(0.0)

    return np.array(features)


def predict_rul_fd001(features, dataset, artifacts):
    ds_artifacts = artifacts[dataset]
    model = ds_artifacts['xgb_model']
    conformal_params = ds_artifacts['conformal_params']

    feature_names = ds_artifacts['feature_names']
    expected_features = feature_names['all_features']

    if len(features) != len(expected_features):
        if len(features) < len(expected_features):
            padded = np.zeros(len(expected_features))
            padded[:len(features)] = features
            features = padded
        else:
            features = features[:len(expected_features)]

    pred = model.predict(features.reshape(1, -1))[0]
    rul_cap = ds_artifacts['rul_params']['rul_cap']
    pred_capped = np.clip(pred, None, rul_cap)

    if pred_capped <= 50:
        q = conformal_params['q_95_near_failure']
    elif pred_capped <= 100:
        q = conformal_params['q_95_mid_life']
    else:
        q = conformal_params['q_95_early_life']

    lower = max(0, pred_capped - q)
    upper = pred_capped + q

    return pred_capped, lower, upper

def predict_rul_fd002(engine_id, cycle, preprocessed_df, artifacts):
    row = preprocessed_df[(preprocessed_df['engine_id'] == engine_id) & (preprocessed_df['cycle'] == cycle)]

    if len(row) == 0:
        return None, None, None

    model = artifacts['FD002']['xgb_model']
    conformal_params = artifacts['FD002']['conformal_params']
    feature_names = artifacts['FD002']['feature_names']

    expected_features = feature_names['all_features']
    features = []
    for col in expected_features:
        if col in row.columns:
            val = row[col].values[0]
            if pd.isna(val):
                val = 0.0
            features.append(float(val))
        else:
            features.append(0.0)

    features = np.array(features).reshape(1, -1)

    pred = model.predict(features)[0]
    pred_capped = np.clip(pred, None, 125)

    if pred_capped <= 50:
        q = conformal_params['q_95_near_failure']
    elif pred_capped <= 100:
        q = conformal_params['q_95_mid_life']
    else:
        q = conformal_params['q_95_early_life']

    lower = max(0, pred_capped - q)
    upper = pred_capped + q

    return pred_capped, lower, upper


def predict_failure_risk(features, dataset, artifacts):
    ds_artifacts = artifacts[dataset]
    calibrated_models = ds_artifacts['calibrated_models']
    tuned_thresholds = ds_artifacts['tuned_thresholds']
    horizons = [10, 20, 30]

    feature_names = ds_artifacts['feature_names']
    expected_features = feature_names['all_features']

    if len(features) != len(expected_features):
        if len(features) < len(expected_features):
            padded = np.zeros(len(expected_features))
            padded[:len(features)] = features
            features = padded
        else:
            features = features[:len(expected_features)]

    risks = {}
    for h in horizons:
        model = calibrated_models[h]['XGBoost']
        prob = model.predict_proba(features.reshape(1, -1))[0, 1]

        if str(h) in tuned_thresholds:
            threshold = tuned_thresholds[str(h)]['XGBoost']
        elif h in tuned_thresholds:
            threshold = tuned_thresholds[h]['XGBoost']
        else:
            threshold = 0.05

        risks[f'h{h}'] = {
            'probability': prob,
            'threshold': threshold,
            'alert': prob >= threshold
        }

    return risks


def predict_anomaly(features, dataset, artifacts):
    ds_artifacts = artifacts[dataset]
    anomaly_models = ds_artifacts['anomaly_models']
    pct_scores_test = ds_artifacts['pct_scores_test']

    feature_names = ds_artifacts['feature_names']
    expected_features = feature_names['all_features']

    if len(features) != len(expected_features):
        if len(features) < len(expected_features):
            padded = np.zeros(len(expected_features))
            padded[:len(features)] = features
            features = padded
        else:
            features = features[:len(expected_features)]

    scores = {}
    for name, model in anomaly_models.items():
        try:
            if name == 'PCA':
                transformed = model.transform(features.reshape(1, -1))
                reconstructed = model.inverse_transform(transformed)
                raw_score = np.mean((features.reshape(1, -1) - reconstructed) ** 2, axis=1)[0]
            else:
                raw_score = -model.decision_function(features.reshape(1, -1))[0]

            if name in pct_scores_test:
                ref_scores = pct_scores_test[name]
                if ref_scores is not None and len(ref_scores) > 0:
                    if raw_score >= np.max(ref_scores):
                        percentile = 100.0
                    elif raw_score <= np.min(ref_scores):
                        percentile = 0.0
                    else:
                        percentile = float(
                            np.interp(raw_score, np.sort(ref_scores), np.linspace(0, 100, len(ref_scores))))
                else:
                    percentile = 50.0
            else:
                percentile = 50.0

            scores[name] = {
                'raw_score': float(raw_score),
                'percentile': float(percentile),
                'alert': percentile >= 95
            }
        except Exception:
            scores[name] = {
                'raw_score': 0.0,
                'percentile': 50.0,
                'alert': False
            }

    return scores


def make_recommendation(rul_pred, rul_lower, rul_upper, failure_risks, anomaly_scores, dataset, artifacts):
    prob_h30 = failure_risks['h30']['probability']
    anomaly_score = anomaly_scores['OCSVM']['percentile']
    interval_width = rul_upper - rul_lower

    decision_params = artifacts[dataset]['decision_params']
    final_thresholds = decision_params['final_thresholds']

    if (rul_pred <= final_thresholds['rul_stop']) or \
            (rul_lower <= 30) or \
            ((prob_h30 > final_thresholds['prob_stop']) and (anomaly_score > final_thresholds['anomaly_stop'])):

        triggers = []
        if rul_pred <= final_thresholds['rul_stop']:
            triggers.append(f"RUL prediction ({rul_pred:.0f}) below STOP threshold ({final_thresholds['rul_stop']})")
        if rul_lower <= 30:
            triggers.append(f"RUL lower bound ({rul_lower:.0f}) below critical threshold (30)")
        if (prob_h30 > final_thresholds['prob_stop']) and (anomaly_score > final_thresholds['anomaly_stop']):
            triggers.append(f"Failure probability ({prob_h30:.1%}) and Anomaly ({anomaly_score:.1f}) above thresholds")

        return {
            'action': 'STOP',
            'color': 'red',
            'triggers': triggers,
            'confidence': 'HIGH' if len(triggers) >= 2 else 'MEDIUM'
        }

    elif (rul_pred <= final_thresholds['rul_inspect']) or \
            (rul_lower <= final_thresholds['rul_inspect_lower']) or \
            ((prob_h30 > final_thresholds['prob_inspect']) and (anomaly_score > final_thresholds['anomaly_inspect'])):

        triggers = []
        if rul_pred <= final_thresholds['rul_inspect']:
            triggers.append(
                f"RUL prediction ({rul_pred:.0f}) below INSPECT threshold ({final_thresholds['rul_inspect']})")
        if rul_lower <= final_thresholds['rul_inspect_lower']:
            triggers.append(
                f"RUL lower bound ({rul_lower:.0f}) below inspect threshold ({final_thresholds['rul_inspect_lower']})")
        if (prob_h30 > final_thresholds['prob_inspect']) and (anomaly_score > final_thresholds['anomaly_inspect']):
            triggers.append(f"Failure probability ({prob_h30:.1%}) and Anomaly ({anomaly_score:.1f}) above thresholds")

        return {
            'action': 'INSPECT',
            'color': 'orange',
            'triggers': triggers,
            'confidence': 'MEDIUM'
        }

    elif (prob_h30 > final_thresholds['monitor_prob']) or (anomaly_score > 85):
        triggers = []
        if prob_h30 > final_thresholds['monitor_prob']:
            triggers.append(
                f"Failure probability ({prob_h30:.1%}) above monitor threshold ({final_thresholds['monitor_prob']:.0%})")
        if anomaly_score > 85:
            triggers.append(f"Anomaly score ({anomaly_score:.1f}) above warning threshold (85)")

        return {
            'action': 'MONITOR',
            'color': 'gold',
            'triggers': triggers,
            'confidence': 'LOW'
        }

    else:
        return {
            'action': 'CONTINUE',
            'color': 'green',
            'triggers': ['All parameters within normal range'],
            'confidence': 'HIGH'
        }


def get_dataset_description(dataset):
    descriptions = {
        'FD001': '1 condition, 1 fault mode',
        'FD002': '6 conditions, 1 fault mode'
    }
    return descriptions.get(dataset, '')


def initialize_session_state():
    if 'prediction_done' not in st.session_state:
        st.session_state.prediction_done = False
    if 'rul_pred' not in st.session_state:
        st.session_state.rul_pred = None
    if 'rul_lower' not in st.session_state:
        st.session_state.rul_lower = None
    if 'rul_upper' not in st.session_state:
        st.session_state.rul_upper = None
    if 'risks' not in st.session_state:
        st.session_state.risks = None
    if 'anomaly_scores' not in st.session_state:
        st.session_state.anomaly_scores = None
    if 'recommendation' not in st.session_state:
        st.session_state.recommendation = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'selected_cycle' not in st.session_state:
        st.session_state.selected_cycle = None
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = None
    if 'artifacts' not in st.session_state:
        st.session_state.artifacts = None
    if 'selected_engine' not in st.session_state:
        st.session_state.selected_engine = None


def main():
    initialize_session_state()

    st.title("Jet Engine Early Warning System")
    st.caption("Predictive Maintenance Dashboard for NASA C-MAPSS Turbofan Engines")

    with st.spinner("Loading model artifacts..."):
        artifacts = load_artifacts()
        preprocessed_fd001, rul_fd001 = load_preprocessed_fd001()
        preprocessed_fd002, rul_fd002 = load_preprocessed_fd002()

    with st.sidebar:
        st.header("Engine Configuration")

        available_datasets = ['FD001', 'FD002']
        selected_dataset = st.selectbox(
            "Select Dataset",
            available_datasets,
            format_func=lambda x: f"{x} - {get_dataset_description(x)}"
        )

        with st.spinner(f"Loading {selected_dataset} data..."):
            if selected_dataset == 'FD001':
                test_df = preprocessed_fd001
                rul_df = rul_fd001
            else:
                test_df = preprocessed_fd002
                rul_df = rul_fd002

            engines = sorted(test_df['engine_id'].unique())
            selected_engine = st.selectbox(
                "Select Engine ID",
                engines,
                format_func=lambda x: f"Engine #{x}"
            )
            st.session_state.selected_engine = selected_engine
            engine_data = test_df[test_df['engine_id'] == selected_engine]
            cycles = sorted(engine_data['cycle'].unique())
            selected_cycle = st.slider(
                "Select Cycle",
                min_value=min(cycles),
                max_value=max(cycles),
                value=max(cycles),
                step=1
            )
            processed_df = test_df

        predict_button = st.button("Run Prediction", type="primary", use_container_width=True)


    if predict_button:
        st.session_state.prediction_done = True

        if selected_dataset == 'FD001':
            features = get_features_for_prediction(processed_df, selected_engine, selected_cycle, selected_dataset, artifacts)
            if features is None:
                st.error("Could not extract features for prediction")
                return
            rul_pred, rul_lower, rul_upper = predict_rul_fd001(features, selected_dataset, artifacts)
        else:
            rul_pred, rul_lower, rul_upper = predict_rul_fd002(selected_engine, selected_cycle, processed_df, artifacts)
            if rul_pred is None:
                st.error("Invalid selection!")
                return

            feature_names = artifacts['FD002']['feature_names']
            expected_features = feature_names['all_features']
            current_row = processed_df[
                (processed_df['engine_id'] == selected_engine) & (processed_df['cycle'] == selected_cycle)]
            features = []
            for col in expected_features:
                if col in current_row.columns:
                    features.append(float(current_row[col].values[0]))
                else:
                    features.append(0.0)
            features = np.array(features)

        with st.spinner("Making predictions..."):
            risks = predict_failure_risk(features, selected_dataset, artifacts)
            anomaly_scores = predict_anomaly(features, selected_dataset, artifacts)
            recommendation = make_recommendation(
                rul_pred, rul_lower, rul_upper,
                risks, anomaly_scores, selected_dataset, artifacts
            )

            st.session_state.rul_pred = rul_pred
            st.session_state.rul_lower = rul_lower
            st.session_state.rul_upper = rul_upper
            st.session_state.risks = risks
            st.session_state.anomaly_scores = anomaly_scores
            st.session_state.recommendation = recommendation
            st.session_state.processed_df = processed_df
            st.session_state.selected_cycle = selected_cycle
            st.session_state.selected_dataset = selected_dataset
            st.session_state.artifacts = artifacts

    if st.session_state.prediction_done and st.session_state.rul_pred is not None:
        rul_pred = st.session_state.rul_pred
        rul_lower = st.session_state.rul_lower
        rul_upper = st.session_state.rul_upper
        risks = st.session_state.risks
        anomaly_scores = st.session_state.anomaly_scores
        recommendation = st.session_state.recommendation
        processed_df = st.session_state.processed_df
        selected_cycle = st.session_state.selected_cycle
        selected_dataset = st.session_state.selected_dataset
        artifacts = st.session_state.artifacts

        st.subheader("Current Engine Status")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Remaining Useful Life",
                f"{rul_pred:.0f} cycles",
                delta=f"95% CI: [{rul_lower:.0f}, {rul_upper:.0f}]"
            )

        with col2:
            prob_h30 = risks['h30']['probability']
            st.metric(
                "Failure Risk (30 cycles)",
                f"{prob_h30:.1%}",
                delta=f"Threshold: {risks['h30']['threshold']:.2f}"
            )

        with col3:
            anomaly_score = anomaly_scores['OCSVM']['percentile']
            st.metric(
                "Anomaly Score",
                f"{anomaly_score:.1f}th percentile",
                delta="Critical > 95%"
            )

        with col4:
            color = recommendation['color']
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; background-color: {color}; text-align: center;">
                <h2 style="color: white; margin: 0; font-size: 24px;">{recommendation['action']}</h2>
                <p style="color: white; margin: 5px 0 0 0; font-size: 14px;">Confidence: {recommendation['confidence']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Failure Risk by Horizon")

        col1, col2, col3 = st.columns(3)
        for i, h in enumerate([10, 20, 30]):
            with [col1, col2, col3][i]:
                prob = risks[f'h{h}']['probability']
                alert = risks[f'h{h}']['alert']
                st.metric(
                    f"Risk in {h} cycles",
                    f"{prob:.1%}",
                    delta="ALERT" if alert else "Normal"
                )

        st.subheader("Anomaly Detection Results")

        anomaly_data = []
        for name, scores in anomaly_scores.items():
            anomaly_data.append({
                'Method': name,
                'Score': f"{scores['percentile']:.1f}th percentile",
                'Status': 'ALERT' if scores['alert'] else 'Normal'
            })
        st.dataframe(pd.DataFrame(anomaly_data), hide_index=True, use_container_width=True)

        st.subheader("Decision Triggers")

        triggers = recommendation['triggers']
        if len(triggers) > 1:
            st.warning("Active triggers:")
            for trigger in triggers:
                st.write(f"- {trigger}")
        else:
            st.success(triggers[0])

        st.subheader("Engine Health Timeline")

        dropped_sensors = artifacts[selected_dataset]['metadata'].get('dropped_sensors', [])

        # برای پلات سنسورها از داده‌های خام استفاده کن
        raw_test_df, _ = load_raw_data(selected_dataset)
        raw_engine_data = raw_test_df[raw_test_df['engine_id'] == st.session_state.selected_engine]

        sensor_cols = [col for col in raw_engine_data.columns if col.startswith('sensor_')]
        sensor_cols = [col for col in sensor_cols if col not in dropped_sensors]

        col1, col2 = st.columns([2, 1])
        with col1:
            selected_sensor = st.selectbox(
                "Select Sensor to Visualize",
                sensor_cols if sensor_cols else ['sensor_2'],
                format_func=lambda x: x
            )
        with col2:
            show_health = st.checkbox("Show Health Features", value=False)

        rul_data = processed_df[processed_df['engine_id'] == st.session_state.selected_engine]

        if show_health:
            fig = make_subplots(rows=2, cols=1, subplot_titles=("RUL Over Time", "Anomaly Score Over Time"),
                                vertical_spacing=0.15)

            fig.add_trace(
                go.Scatter(x=raw_engine_data['cycle'], y=rul_data['RUL'], mode='lines', name='True RUL',
                           line=dict(color='green', width=2)),
                row=1, col=1
            )
            fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Critical", row=1, col=1)

            anomaly_col = 'OCSVM_Anomaly_Score'
            if anomaly_col in processed_df.columns:
                anomaly_data = processed_df[processed_df['engine_id'] == st.session_state.selected_engine]
                fig.add_trace(
                    go.Scatter(x=anomaly_data['cycle'], y=anomaly_data[anomaly_col], mode='lines',
                               name='Anomaly Score',
                               line=dict(color='orange', width=2)),
                    row=2, col=1
                )
                fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Critical", row=2, col=1)
                fig.add_hline(y=90, line_dash="dot", line_color="orange", annotation_text="Warning", row=2, col=1)

            fig.update_layout(height=500, showlegend=True)

        else:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(x=raw_engine_data['cycle'], y=raw_engine_data[selected_sensor], mode='lines',
                           name=selected_sensor,
                           line=dict(color='blue', width=2))
            )

            fig.add_trace(
                go.Scatter(x=rul_data['cycle'], y=rul_data['RUL'], mode='lines', name='RUL',
                           line=dict(color='green', width=2, dash='dot'), yaxis='y2')
            )

            fig.update_layout(
                yaxis=dict(title=selected_sensor),
                yaxis2=dict(title='RUL', overlaying='y', side='right'),
                height=400,
                showlegend=True
            )

        fig.add_vline(x=selected_cycle, line_dash="dash", line_color="red", annotation_text="Current Cycle",
                      annotation_position="top")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Model Metadata"):
            metadata = artifacts[selected_dataset]['metadata']
            rul_params = artifacts[selected_dataset]['rul_params']
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Dataset Information**")
                st.write(f"- Dataset: {metadata.get('dataset', 'N/A')}")
                st.write(f"- Description: {metadata.get('description', 'N/A')}")
                st.write(f"- Training Date: {metadata.get('training_date', 'N/A')}")
                st.write(f"- Author: {metadata.get('author', 'N/A')}")
            with col2:
                st.write("**Model Configuration**")
                st.write(f"- Model Version: {metadata.get('model_version', 'N/A')}")
                window_sizes = metadata.get('window_sizes', [])
                if isinstance(window_sizes, list):
                    st.write(f"- Window Sizes: {', '.join(map(str, window_sizes))} cycles")
                else:
                    st.write(f"- Window Size: {window_sizes} cycles")
                st.write(f"- RUL Cap: {rul_params.get('rul_cap', 125)} cycles")
                st.write(f"- Total Features: {metadata.get('total_features', 'N/A')}")
                if selected_dataset == 'FD002':
                    st.write(f"- Number of Regimes: {metadata.get('num_regimes', 'N/A')}")


if __name__ == "__main__":
    main()