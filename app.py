import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings

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

        # # ====== DEBUG: Check loaded decision_params ======
        # if dataset == 'FD001':
        #     st.write(f"### DEBUG: Loaded decision_params for {dataset}")
        #     st.json(artifacts[dataset]['decision_params'])
        # # ====== END DEBUG ======

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
    col_names = ['engine_id', 'cycle'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in
                                                                                     range(1, 22)]

    train_df = pd.read_csv(f'data/train_{dataset}.txt', sep=r'\s+', header=None, names=col_names)
    test_df = pd.read_csv(f'data/test_{dataset}.txt', sep=r'\s+', header=None, names=col_names)
    rul_df = pd.read_csv(f'data/RUL_{dataset}.txt', sep=r'\s+', header=None, names=['RUL_final'])

    return train_df, test_df, rul_df


def extract_window_features(df, window_info, feature_cols):
    W = window_info['window_size']
    df_out = df.copy()
    grouped = df_out.groupby('engine_id')

    for col in feature_cols:
        if col not in df.columns:
            continue

        rolling_obj = grouped[col].rolling(window=W, min_periods=1)
        df_out[f'{col}_roll_mean'] = rolling_obj.mean().reset_index(level=0, drop=True)
        df_out[f'{col}_roll_std'] = rolling_obj.std().reset_index(level=0, drop=True).fillna(0)
        df_out[f'{col}_roll_min'] = rolling_obj.min().reset_index(level=0, drop=True)
        df_out[f'{col}_roll_max'] = rolling_obj.max().reset_index(level=0, drop=True)
        df_out[f'{col}_slope'] = df_out.groupby('engine_id')[col].transform(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0
        )

        df_out[f'{col}_ewma'] = grouped[col].transform(
            lambda x: x.ewm(span=W, adjust=False).mean()
        )

        df_out[f'{col}_diff'] = grouped[col].diff().fillna(0)

    return df_out


def preprocess_data(dataset, test_df, rul_df, artifacts):
    ds_artifacts = artifacts[dataset]

    test_max_cycle = test_df.groupby('engine_id')['cycle'].max().to_dict()
    rul_mapping = {engine: rul_df.iloc[i, 0] for i, engine in enumerate(test_df['engine_id'].unique())}

    test_df['max_cycle'] = test_df['engine_id'].map(test_max_cycle)
    test_df['RUL_final'] = test_df['engine_id'].map(rul_mapping)
    test_df['RUL'] = test_df['max_cycle'] - test_df['cycle'] + test_df['RUL_final']

    rul_cap = 125
    test_df['RUL_capped'] = test_df['RUL'].clip(upper=rul_cap)

    test_df_raw = test_df.copy()

    feature_info = ds_artifacts['feature_info']
    features_to_scale = feature_info['all_features']
    scaler = ds_artifacts['scaler']
    dropped_sensors = ds_artifacts['feature_info'].get('dropped_sensors', [])

    if dataset == 'FD001':
        dropped_sensors = artifacts['FD001']['metadata']['dropped_sensors']
    else:
        dropped_sensors = artifacts['FD002']['metadata']['dropped_sensors']

    if dropped_sensors:
        test_df = test_df.drop(columns=dropped_sensors, errors='ignore')
        test_df_raw = test_df_raw.drop(columns=dropped_sensors, errors='ignore')

    sensor_cols = [col for col in test_df.columns if col.startswith('sensor_')]
    if dataset == 'FD001':
        test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])
    else:
        op_settings = feature_info['op_settings']
        test_df[op_settings] = scaler.transform(test_df[op_settings])

        sensor_cols_scaled = feature_info['active_sensors']
        scaler_dict = ds_artifacts['scaler_dict']
        kmeans = ds_artifacts['kmeans']

        test_df['regime'] = kmeans.predict(test_df[op_settings])

        for col in sensor_cols_scaled:
            test_df[col] = test_df[col].astype(float)

        for r in range(6):
            regime_mask = test_df['regime'] == r
            if regime_mask.sum() > 0 and r in scaler_dict:
                test_df.loc[regime_mask, sensor_cols_scaled] = scaler_dict[r].transform(
                    test_df.loc[regime_mask, sensor_cols_scaled])

    window_info = ds_artifacts['window_info']
    feature_cols = window_info['feature_cols']

    active_cols = [col for col in feature_cols if col in test_df.columns]
    test_df = extract_window_features(test_df, window_info, active_cols)

    for col in sensor_cols:
        if col in test_df_raw.columns:
            test_df[col + '_raw'] = test_df_raw[col]

    return test_df


def predict_rul(features, dataset, artifacts):
    ds_artifacts = artifacts[dataset]
    model = ds_artifacts['xgb_model']
    conformal_params = ds_artifacts['conformal_params']

    if ds_artifacts['feature_names'] is not None:
        expected_features = ds_artifacts['feature_names']['all_features']
        if len(features) != len(expected_features):
            features = features[:len(expected_features)]

    pred = model.predict(features.reshape(1, -1))[0]
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

    if ds_artifacts['feature_names'] is not None:
        expected_features = ds_artifacts['feature_names']['all_features']
        if len(features) != len(expected_features):
            features = features[:len(expected_features)]

    risks = {}
    for h in horizons:
        model = calibrated_models[h]['XGBoost']
        prob = model.predict_proba(features.reshape(1, -1))[0, 1]
        threshold = tuned_thresholds[h]['XGBoost']
        risks[f'h{h}'] = {
            'probability': prob,
            'threshold': threshold,
            'alert': prob >= threshold
        }

    return risks


# def predict_anomaly(features, dataset, artifacts):
#     ds_artifacts = artifacts[dataset]
#     anomaly_models = ds_artifacts['anomaly_models']
#
#     if ds_artifacts['feature_names'] is not None:
#         expected_features = ds_artifacts['feature_names']['all_features']
#         if len(features) != len(expected_features):
#             features = features[:len(expected_features)]
#
#     scores = {}
#     for name, model in anomaly_models.items():
#         if name == 'PCA':
#             reconstructed = model.inverse_transform(model.transform(features.reshape(1, -1)))
#             raw_score = np.mean((features.reshape(1, -1) - reconstructed) ** 2, axis=1)[0]
#         else:
#             raw_score = -model.decision_function(features.reshape(1, -1))[0]
#
#         threshold = 95
#
#         scores[name] = {
#             'raw_score': raw_score,
#             'percentile': raw_score,
#             'alert': raw_score >= threshold
#         }
#
#     return scores


# def make_recommendation(rul_pred, rul_lower, rul_upper, failure_risks, anomaly_scores, dataset, artifacts):
#     ds_artifacts = artifacts[dataset]
#     decision_params = ds_artifacts['decision_params']
#
#     prob_h30 = failure_risks['h30']['probability']
#     anomaly_score = anomaly_scores['OCSVM']['percentile']
#     interval_width = rul_upper - rul_lower
#
#     stop_rul_threshold = decision_params['stop_rules'].get('rul_lower_bound', 20)
#     stop_prob_threshold = decision_params['stop_rules'].get('failure_prob_threshold', 0.6)
#     stop_anomaly_threshold = decision_params['stop_rules'].get('anomaly_threshold', 95)
#
#     inspect_rul_threshold = decision_params['inspect_rules'].get('rul_lower_bound', 30)
#     inspect_prob_threshold = decision_params['inspect_rules'].get('failure_prob_threshold', 0.3)
#     inspect_anomaly_threshold = decision_params['inspect_rules'].get('anomaly_threshold', 90)
#     inspect_uncertainty_threshold = decision_params['inspect_rules'].get('uncertainty_threshold', 50)
#
#     if (rul_lower < stop_rul_threshold or
#             prob_h30 > stop_prob_threshold or
#             anomaly_score > stop_anomaly_threshold):
#
#         triggers = []
#         if rul_lower < stop_rul_threshold:
#             triggers.append(f"RUL lower bound ({rul_lower:.0f}) below critical threshold ({stop_rul_threshold})")
#         if prob_h30 > stop_prob_threshold:
#             triggers.append(
#                 f"Failure probability ({prob_h30:.1%}) above critical threshold ({stop_prob_threshold:.0%})")
#         if anomaly_score > stop_anomaly_threshold:
#             triggers.append(f"Anomaly score ({anomaly_score:.1f}) above critical threshold ({stop_anomaly_threshold})")
#
#         return {
#             'action': 'STOP',
#             'color': 'red',
#             'triggers': triggers,
#             'confidence': 'HIGH' if len(triggers) >= 2 else 'MEDIUM'
#         }
#
#     elif (rul_lower < inspect_rul_threshold or
#           prob_h30 > inspect_prob_threshold or
#           anomaly_score > inspect_anomaly_threshold or
#           interval_width > inspect_uncertainty_threshold):
#
#         triggers = []
#         if rul_lower < inspect_rul_threshold:
#             triggers.append(f"RUL lower bound ({rul_lower:.0f}) below inspect threshold ({inspect_rul_threshold})")
#         if prob_h30 > inspect_prob_threshold:
#             triggers.append(
#                 f"Failure probability ({prob_h30:.1%}) above inspect threshold ({inspect_prob_threshold:.0%})")
#         if anomaly_score > inspect_anomaly_threshold:
#             triggers.append(
#                 f"Anomaly score ({anomaly_score:.1f}) above inspect threshold ({inspect_anomaly_threshold})")
#         if interval_width > inspect_uncertainty_threshold:
#             triggers.append(
#                 f"Uncertainty width ({interval_width:.0f}) above inspect threshold ({inspect_uncertainty_threshold})")
#
#         return {
#             'action': 'INSPECT',
#             'color': 'orange',
#             'triggers': triggers,
#             'confidence': 'MEDIUM'
#         }
#
#     else:
#         return {
#             'action': 'CONTINUE',
#             'color': 'green',
#             'triggers': ['All parameters within normal range'],
#             'confidence': 'HIGH'
#         }
# def make_recommendation(rul_pred, rul_lower, rul_upper, failure_risks, anomaly_scores, dataset, artifacts):
#     ds_artifacts = artifacts[dataset]
#     decision_params = ds_artifacts['decision_params']
#
#     prob_h30 = failure_risks['h30']['probability']
#     anomaly_score = anomaly_scores['OCSVM']['percentile']
#     interval_width = rul_upper - rul_lower
#
#     stop_rul_threshold = decision_params['stop_rules'].get('rul_lower_bound', 20)
#     stop_prob_threshold = decision_params['stop_rules'].get('failure_prob_threshold', 0.6)
#     stop_anomaly_threshold = decision_params['stop_rules'].get('anomaly_threshold', 95)
#
#     inspect_rul_threshold = decision_params['inspect_rules'].get('rul_lower_bound', 30)
#     inspect_prob_threshold = decision_params['inspect_rules'].get('failure_prob_threshold', 0.3)
#     inspect_anomaly_threshold = decision_params['inspect_rules'].get('anomaly_threshold', 90)
#     inspect_uncertainty_threshold = decision_params['inspect_rules'].get('uncertainty_threshold', 50)
#
#     # # ====== DEBUG SECTION ======
#     # st.write("### Debug: Decision Values")
#     # debug_data = {
#     #     'Parameter': [
#     #         'RUL Lower Bound',
#     #         'Failure Probability (h30)',
#     #         'Anomaly Score (OCSVM)',
#     #         'Interval Width',
#     #         'STOP - RUL threshold',
#     #         'STOP - Prob threshold',
#     #         'STOP - Anomaly threshold',
#     #         'INSPECT - RUL threshold',
#     #         'INSPECT - Prob threshold',
#     #         'INSPECT - Anomaly threshold',
#     #         'INSPECT - Uncertainty threshold'
#     #     ],
#     #     'Value': [
#     #         f"{rul_lower:.0f}",
#     #         f"{prob_h30:.1%}",
#     #         f"{anomaly_score:.1f}",
#     #         f"{interval_width:.0f}",
#     #         f"{stop_rul_threshold}",
#     #         f"{stop_prob_threshold:.0%}",
#     #         f"{stop_anomaly_threshold}",
#     #         f"{inspect_rul_threshold}",
#     #         f"{inspect_prob_threshold:.0%}",
#     #         f"{inspect_anomaly_threshold}",
#     #         f"{inspect_uncertainty_threshold}"
#     #     ],
#     #     'Status': [
#     #         'OK' if rul_lower >= stop_rul_threshold else '🚨 LOW',
#     #         'OK' if prob_h30 <= stop_prob_threshold else '🚨 HIGH',
#     #         'OK' if anomaly_score <= stop_anomaly_threshold else '🚨 HIGH',
#     #         'OK' if interval_width <= inspect_uncertainty_threshold else '⚠️ WIDE',
#     #         '-',
#     #         '-',
#     #         '-',
#     #         '-',
#     #         '-',
#     #         '-',
#     #         '-'
#     #     ]
#     # }
#     # st.dataframe(pd.DataFrame(debug_data), hide_index=True, use_container_width=True)
#     #
#     # st.write("### Decision Logic")
#     # st.write(
#     #     f"**STOP condition:** ({rul_lower:.0f} < {stop_rul_threshold}) or ({prob_h30:.1%} > {stop_prob_threshold:.0%}) or ({anomaly_score:.1f} > {stop_anomaly_threshold})")
#     # st.write(
#     #     f"**INSPECT condition:** ({rul_lower:.0f} < {inspect_rul_threshold}) or ({prob_h30:.1%} > {inspect_prob_threshold:.0%}) or ({anomaly_score:.1f} > {inspect_anomaly_threshold}) or ({interval_width:.0f} > {inspect_uncertainty_threshold})")
#     # # ====== END DEBUG ======
#
#     if (rul_lower < stop_rul_threshold or
#             prob_h30 > stop_prob_threshold or
#             anomaly_score > stop_anomaly_threshold):
#
#         triggers = []
#         if rul_lower < stop_rul_threshold:
#             triggers.append(f"RUL lower bound ({rul_lower:.0f}) below critical threshold ({stop_rul_threshold})")
#         if prob_h30 > stop_prob_threshold:
#             triggers.append(
#                 f"Failure probability ({prob_h30:.1%}) above critical threshold ({stop_prob_threshold:.0%})")
#         if anomaly_score > stop_anomaly_threshold:
#             triggers.append(f"Anomaly score ({anomaly_score:.1f}) above critical threshold ({stop_anomaly_threshold})")
#
#         return {
#             'action': 'STOP',
#             'color': 'red',
#             'triggers': triggers,
#             'confidence': 'HIGH' if len(triggers) >= 2 else 'MEDIUM'
#         }
#
#     elif (rul_lower < inspect_rul_threshold or
#           prob_h30 > inspect_prob_threshold or
#           anomaly_score > inspect_anomaly_threshold or
#           interval_width > inspect_uncertainty_threshold):
#
#         triggers = []
#         if rul_lower < inspect_rul_threshold:
#             triggers.append(f"RUL lower bound ({rul_lower:.0f}) below inspect threshold ({inspect_rul_threshold})")
#         if prob_h30 > inspect_prob_threshold:
#             triggers.append(
#                 f"Failure probability ({prob_h30:.1%}) above inspect threshold ({inspect_prob_threshold:.0%})")
#         if anomaly_score > inspect_anomaly_threshold:
#             triggers.append(
#                 f"Anomaly score ({anomaly_score:.1f}) above inspect threshold ({inspect_anomaly_threshold})")
#         if interval_width > inspect_uncertainty_threshold:
#             triggers.append(
#                 f"Uncertainty width ({interval_width:.0f}) above inspect threshold ({inspect_uncertainty_threshold})")
#
#         return {
#             'action': 'INSPECT',
#             'color': 'orange',
#             'triggers': triggers,
#             'confidence': 'MEDIUM'
#         }
#
#     else:
#         return {
#             'action': 'CONTINUE',
#             'color': 'green',
#             'triggers': ['All parameters within normal range'],
#             'confidence': 'HIGH'
#         }

def predict_anomaly(features, dataset, artifacts):
    ds_artifacts = artifacts[dataset]
    anomaly_models = ds_artifacts['anomaly_models']

    if ds_artifacts['feature_names'] is not None:
        expected_features = ds_artifacts['feature_names']['all_features']
        if len(features) != len(expected_features):
            features = features[:len(expected_features)]

    scores = {}
    for name, model in anomaly_models.items():
        if name == 'PCA':
            reconstructed = model.inverse_transform(model.transform(features.reshape(1, -1)))
            raw_score = np.mean((features.reshape(1, -1) - reconstructed) ** 2, axis=1)[0]
        else:
            raw_score = -model.decision_function(features.reshape(1, -1))[0]

        threshold = 95

        # ====== FIX: Use raw_score as percentile (temporarily) ======
        # در حالت ایده‌آل، باید از pct_scores_test استفاده کنید
        # ولی برای تست، raw_score را به percentiles محدود می‌کنیم
        if name == 'OCSVM':
            percentile = max(0, min(100, raw_score + 50))  # تبدیل تقریبی
        else:
            percentile = max(0, min(100, raw_score))
        # ====== END FIX ======

        scores[name] = {
            'raw_score': raw_score,
            'percentile': percentile,
            'alert': raw_score >= threshold
        }

    return scores

def make_recommendation(rul_pred, rul_lower, rul_upper, failure_risks, anomaly_scores, dataset, artifacts):
    prob_h30 = failure_risks['h30']['probability']
    anomaly_score = anomaly_scores['OCSVM']['percentile']
    interval_width = rul_upper - rul_lower

    # ====== HARDCODED THRESHOLDS FOR TESTING ======
    if dataset == 'FD001':
        stop_rul_threshold = 15
        stop_prob_threshold = 0.6
        stop_anomaly_threshold = 97

        inspect_rul_threshold = 25
        inspect_prob_threshold = 0.4
        inspect_anomaly_threshold = 92
        inspect_uncertainty_threshold = 60
    else:
        stop_rul_threshold = 15
        stop_prob_threshold = 0.6
        stop_anomaly_threshold = 97

        inspect_rul_threshold = 25
        inspect_prob_threshold = 0.4
        inspect_anomaly_threshold = 92
        inspect_uncertainty_threshold = 75
    # ====== END HARDCODED ======

    # ====== DEBUG SECTION ======
    st.write("### Debug: Decision Values")
    debug_data = {
        'Parameter': [
            'RUL Lower Bound',
            'Failure Probability (h30)',
            'Anomaly Score (OCSVM)',
            'Interval Width',
            'STOP - RUL threshold',
            'STOP - Prob threshold',
            'STOP - Anomaly threshold',
            'INSPECT - RUL threshold',
            'INSPECT - Prob threshold',
            'INSPECT - Anomaly threshold',
            'INSPECT - Uncertainty threshold'
        ],
        'Value': [
            f"{rul_lower:.0f}",
            f"{prob_h30:.1%}",
            f"{anomaly_score:.1f}",
            f"{interval_width:.0f}",
            f"{stop_rul_threshold}",
            f"{stop_prob_threshold:.0%}",
            f"{stop_anomaly_threshold}",
            f"{inspect_rul_threshold}",
            f"{inspect_prob_threshold:.0%}",
            f"{inspect_anomaly_threshold}",
            f"{inspect_uncertainty_threshold}"
        ],
        'Status': [
            'OK' if rul_lower >= stop_rul_threshold else 'LOW',
            'OK' if prob_h30 <= stop_prob_threshold else 'HIGH',
            'OK' if anomaly_score <= stop_anomaly_threshold else 'HIGH',
            'OK' if interval_width <= inspect_uncertainty_threshold else 'WIDE',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-'
        ]
    }
    st.dataframe(pd.DataFrame(debug_data), hide_index=True, use_container_width=True)

    st.write("### Decision Logic")
    st.write(
        f"**STOP condition:** ({rul_lower:.0f} < {stop_rul_threshold}) or ({prob_h30:.1%} > {stop_prob_threshold:.0%}) or ({anomaly_score:.1f} > {stop_anomaly_threshold})")
    st.write(
        f"**INSPECT condition:** ({rul_lower:.0f} < {inspect_rul_threshold}) or ({prob_h30:.1%} > {inspect_prob_threshold:.0%}) or ({anomaly_score:.1f} > {inspect_anomaly_threshold}) or ({interval_width:.0f} > {inspect_uncertainty_threshold})")
    # ====== END DEBUG ======

    if (rul_lower < stop_rul_threshold or
            prob_h30 > stop_prob_threshold or
            anomaly_score > stop_anomaly_threshold):

        triggers = []
        if rul_lower < stop_rul_threshold:
            triggers.append(f"RUL lower bound ({rul_lower:.0f}) below critical threshold ({stop_rul_threshold})")
        if prob_h30 > stop_prob_threshold:
            triggers.append(
                f"Failure probability ({prob_h30:.1%}) above critical threshold ({stop_prob_threshold:.0%})")
        if anomaly_score > stop_anomaly_threshold:
            triggers.append(f"Anomaly score ({anomaly_score:.1f}) above critical threshold ({stop_anomaly_threshold})")

        return {
            'action': 'STOP',
            'color': 'red',
            'triggers': triggers,
            'confidence': 'HIGH' if len(triggers) >= 2 else 'MEDIUM'
        }

    elif (rul_lower < inspect_rul_threshold or
          prob_h30 > inspect_prob_threshold or
          anomaly_score > inspect_anomaly_threshold or
          interval_width > inspect_uncertainty_threshold):

        triggers = []
        if rul_lower < inspect_rul_threshold:
            triggers.append(f"RUL lower bound ({rul_lower:.0f}) below inspect threshold ({inspect_rul_threshold})")
        if prob_h30 > inspect_prob_threshold:
            triggers.append(
                f"Failure probability ({prob_h30:.1%}) above inspect threshold ({inspect_prob_threshold:.0%})")
        if anomaly_score > inspect_anomaly_threshold:
            triggers.append(
                f"Anomaly score ({anomaly_score:.1f}) above inspect threshold ({inspect_anomaly_threshold})")
        if interval_width > inspect_uncertainty_threshold:
            triggers.append(
                f"Uncertainty width ({interval_width:.0f}) above inspect threshold ({inspect_uncertainty_threshold})")

        return {
            'action': 'INSPECT',
            'color': 'orange',
            'triggers': triggers,
            'confidence': 'MEDIUM'
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


def main():
    st.title("Jet Engine Early Warning System")
    st.caption("Predictive Maintenance Dashboard for NASA C-MAPSS Turbofan Engines")

    with st.spinner("Loading model artifacts..."):
        artifacts = load_artifacts()

    with st.sidebar:
        st.header("Engine Configuration")

        available_datasets = ['FD001', 'FD002']
        selected_dataset = st.selectbox(
            "Select Dataset",
            available_datasets,
            format_func=lambda x: f"{x} - {get_dataset_description(x)}"
        )

        with st.spinner(f"Loading {selected_dataset} data..."):
            train_df, test_df, rul_df = load_raw_data(selected_dataset)
            processed_df = preprocess_data(selected_dataset, test_df, rul_df, artifacts)

        engines = sorted(processed_df['engine_id'].unique())
        selected_engine = st.selectbox(
            "Select Engine ID",
            engines,
            format_func=lambda x: f"Engine #{x}"
        )

        engine_data = processed_df[processed_df['engine_id'] == selected_engine]
        cycles = sorted(engine_data['cycle'].unique())
        selected_cycle = st.slider(
            "Select Cycle",
            min_value=min(cycles),
            max_value=max(cycles),
            value=max(cycles),
            step=1
        )

        predict_button = st.button("Run Prediction", type="primary", use_container_width=True)

    if predict_button or st.session_state.get('prediction_done', False):
        if predict_button:
            st.session_state.prediction_done = True

        current_row = engine_data[engine_data['cycle'] == selected_cycle]
        if len(current_row) == 0:
            st.error("Invalid selection! Please choose a valid cycle.")
            return

        if artifacts[selected_dataset]['feature_names'] is not None:
            expected_cols = artifacts[selected_dataset]['feature_names']['all_features']
            available_expected = [col for col in expected_cols if col in processed_df.columns]
            feature_cols = [col for col in available_expected if col in processed_df.columns]
        else:
            feature_cols = [col for col in processed_df.columns
                            if col not in ['engine_id', 'cycle', 'RUL', 'RUL_capped', 'max_cycle', 'RUL_final']]
            if 'regime' in processed_df.columns:
                feature_cols = [col for col in feature_cols if col != 'regime']

        features = current_row[feature_cols].values.flatten()

        if features.dtype == 'object':
            try:
                features = features.astype(float)
            except:
                features = np.array([float(x) if isinstance(x, (int, float)) else 0.0 for x in features])

        if predict_button:
            with st.spinner("Making predictions..."):
                rul_pred, rul_lower, rul_upper = predict_rul(features, selected_dataset, artifacts)
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
                st.session_state.engine_data = engine_data
                st.session_state.selected_cycle = selected_cycle
                st.session_state.selected_dataset = selected_dataset
                st.session_state.artifacts = artifacts

        if st.session_state.get('prediction_done', False):
            rul_pred = st.session_state.rul_pred
            rul_lower = st.session_state.rul_lower
            rul_upper = st.session_state.rul_upper
            risks = st.session_state.risks
            anomaly_scores = st.session_state.anomaly_scores
            recommendation = st.session_state.recommendation
            processed_df = st.session_state.processed_df
            engine_data = st.session_state.engine_data
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

            sensor_cols = [col for col in processed_df.columns if col.endswith('_raw') and 'sensor_' in col]
            sensor_cols = [col for col in sensor_cols if col.replace('_raw', '') not in dropped_sensors]

            col1, col2 = st.columns([2, 1])
            with col1:
                selected_sensor = st.selectbox(
                    "Select Sensor to Visualize",
                    sensor_cols if sensor_cols else ['sensor_2_raw'],
                    format_func=lambda x: x.replace('_raw', '')
                )
            with col2:
                show_health = st.checkbox("Show Health Features", value=False)

            if show_health:
                fig = make_subplots(rows=2, cols=1, subplot_titles=("RUL Over Time", "Anomaly Score Over Time"),
                                    vertical_spacing=0.15)

                fig.add_trace(
                    go.Scatter(x=engine_data['cycle'], y=engine_data['RUL'], mode='lines', name='True RUL',
                               line=dict(color='green', width=2)),
                    row=1, col=1
                )
                fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Critical", row=1, col=1)

                anomaly_col = 'OCSVM_Anomaly_Score'
                if anomaly_col in engine_data.columns:
                    fig.add_trace(
                        go.Scatter(x=engine_data['cycle'], y=engine_data[anomaly_col], mode='lines',
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
                    go.Scatter(x=engine_data['cycle'], y=engine_data[selected_sensor], mode='lines',
                               name=selected_sensor.replace('_raw', ''),
                               line=dict(color='blue', width=2))
                )

                fig.add_trace(
                    go.Scatter(x=engine_data['cycle'], y=engine_data['RUL'], mode='lines', name='RUL',
                               line=dict(color='green', width=2, dash='dot'), yaxis='y2')
                )

                fig.update_layout(
                    yaxis=dict(title=selected_sensor.replace('_raw', '')),
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
                    st.write(f"- Window Size: {metadata.get('window_size', 'N/A')} cycles")
                    st.write(f"- RUL Cap: {rul_params.get('rul_cap', 125)} cycles")
                    st.write(f"- Total Features: {metadata.get('total_features', 'N/A')}")
                    if selected_dataset == 'FD002':
                        st.write(f"- Number of Regimes: {metadata.get('num_regimes', 'N/A')}")


if __name__ == "__main__":
    main()