import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Jet Engine Early Warning System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }
    .main {
        background-color: #121212;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
        border-right: 1px solid #2a2a2a;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #b3b3b3;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    .stSelectbox > div > div {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #3a3a3a !important;
    }
    .stSelectbox > div > div:hover {
        border-color: #1db954 !important;
    }

    .stSlider > div > div > div {
        # background: linear-gradient(90deg, #1db954, #1ed760) !important;
    }
    
    .stSlider > div > div > div > div > div {
        color: #1db954 !important;
    }
    # .stSlider > div > div > div > div[data-baseweb="slider"] {
    #     background: linear-gradient(90deg, #1db954, #1ed760) !important;
    # }
    .stSlider > div > div > div > div > div[role="slider"] {
        background-color: #1db954 !important;
        border-color: #1db954 !important;
    }
    .stSlider > div > div > div > div > div[role="slider"]:focus-visible {
        outline: 2px solid rgba(29, 185, 84, 0.5) !important;
        outline-offset: 2px !important;
    }
    .stSlider > div > div > div > div > div[role="slider"]:hover {
        box-shadow: 0 0 0 4px rgba(29, 185, 84, 0.3) !important;
    }
    .stSlider > div > div > div > div > div[role="slider"]:active {
        box-shadow: 0 0 0 6px rgba(29, 185, 84, 0.4) !important;
    }

    # .stSlider [data-testid="stSliderTickBar"] {
    #     # color: #1db954 !important;
    # }
    # .stSlider .stSliderTick {
    #     # color: #1db954 !important;
    # }
    # .stSlider .stSliderNumber {
    #     # color: #1db954 !important;
    # }
    # .stSlider .stSliderNumber > div {
    #     # color: #1db954 !important;
    # }
    # .stSlider .stSliderNumber > div > div {
    #     # color: #1db954 !important;
    # }

    .stButton > button {
        background: linear-gradient(135deg, #1db954, #1ed760) !important;
        color: #121212 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3) !important;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 25px rgba(29, 185, 84, 0.5) !important;
    }

    .metric-card {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 20px;
        border: 2px solid #2a2a2a;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        cursor: default;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #1db954, #1ed760);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #1db954;
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }
    .metric-card .label {
        font-size: 13px;
        color: #b3b3b3;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .metric-card .value {
        font-size: 32px;
        font-weight: 700;
        color: #ffffff;
        margin: 8px 0 4px 0;
    }
    .metric-card .sub {
        font-size: 12px;
        color: #666;
    }

    .metric-card.green::before { background: linear-gradient(90deg, #1db954, #1ed760); }
    .metric-card.green .value { color: #1db954; }
    .metric-card.green:hover { border-color: #1db954; box-shadow: 0 8px 30px rgba(29, 185, 84, 0.3) !important;}

    .metric-card.orange::before { background: linear-gradient(90deg, #fb8c00, #e65100); }
    .metric-card.orange .value { color: #ff6b35; }
    .metric-card.orange:hover { border-color: #ff6b35; box-shadow: 0 8px 30px rgba(251, 140, 0, 0.3) !important;}

    .metric-card.red::before { background: linear-gradient(90deg, #e53935, #c62828); }
    .metric-card.red .value { color: #e53935; }
    .metric-card.red:hover { border-color: #e53935; box-shadow: 0 8px 30px rgba(229, 57, 53, 0.3) !important;}

    .metric-card.purple::before { background: linear-gradient(90deg, #9b59b6, #c39bd3); }
    .metric-card.purple .value { color: #9b59b6; }
    .metric-card.purple:hover { border-color: #9b59b6; }

    .metric-card.blue::before { background: linear-gradient(90deg, #3498db, #5dade2); }
    .metric-card.blue .value { color: #3498db; }
    .metric-card.blue:hover { border-color: #3498db; box-shadow: 0 8px 30px rgba(66, 165, 245, 0.15) !important;}

    .metric-card.rec-card {
        border-color: #2a2a2a;
    }
    .metric-card.rec-card:hover {
        border-color: #1db954 !important;
    }

    .rec-btn {
        display: inline-block;
        padding: 10px 24px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 18px;
        text-align: center;
        letter-spacing: 0.5px;
        margin-top: 8px;
        transition: all 0.3s ease;
        border: none;
        width: 100%;
    }
    .rec-btn.stop {
        background: linear-gradient(135deg, #c62828, #e53935);
        color: #ffffff;
        box-shadow: 0 4px 20px rgba(229, 57, 53, 0.3);
    }
    .rec-btn.inspect {
        background: linear-gradient(135deg, #e65100, #fb8c00);
        color: #ffffff;
        box-shadow: 0 4px 20px rgba(251, 140, 0, 0.3);
    }
    .rec-btn.monitor {
        background: linear-gradient(135deg, #f9a825, #fdd835);
        color: #121212;
        box-shadow: 0 4px 20px rgba(253, 216, 53, 0.3);
    }
    .rec-btn.continue {
        background: linear-gradient(135deg, #1db954, #1ed760);
        color: #121212;
        box-shadow: 0 4px 20px rgba(29, 185, 84, 0.3);
    }
    .rec-btn:hover {
        transform: scale(1.03);
    }

    .risk-box {
        background: #1a1a1a;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #2a2a2a;
        transition: all 0.3s ease;
    }
    .risk-box:hover {
        border-color: #1db954;
    }
    .risk-box .horizon {
        font-size: 13px;
        color: #b3b3b3;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .risk-box .prob {
        font-size: 28px;
        font-weight: 700;
        margin: 8px 0;
    }
    .risk-box .status {
        font-size: 13px;
        font-weight: 500;
    }

    .anomaly-table-container {
        background: #1a1a1a;
        border-radius: 10px;
        border: 1px solid #2a2a2a;
        overflow: hidden;
        padding: 0;
        width: 100%;
    }
    .anomaly-table-container table {
        width: 100%;
        border-collapse: collapse;
        margin: 0;
    }
    .anomaly-table-container th {
        background-color: #2a2a2a;
        color: #ffffff;
        padding: 10px 16px;
        text-align: left;
        font-weight: 600;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 2px solid #3a3a3a;
    }
    .anomaly-table-container td {
        padding: 10px 16px;
        color: #b3b3b3;
        border-bottom: 1px solid #2a2a2a;
        font-size: 14px;
    }
    .anomaly-table-container tr:last-child td {
        border-bottom: none;
    }
    .anomaly-table-container .status-normal {
        color: #1db954;
        font-weight: 500;
    }
    .anomaly-table-container .status-alert {
        color: #e53935;
        font-weight: 500;
    }
    .anomaly-table-container tr:hover td {
        background-color: #252525;
    }

    .streamlit-expanderHeader {
        background-color: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: #1db954 !important;
    }
    .streamlit-expanderContent {
        background-color: #121212 !important;
        border: 1px solid #2a2a2a !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }

    .dataframe {
        background-color: #1a1a1a !important;
        border-radius: 8px !important;
    }
    .dataframe thead tr th {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
    }
    .dataframe tbody tr td {
        color: #b3b3b3 !important;
    }

    hr {
        border-color: #2a2a2a !important;
        margin: 20px 0 !important;
    }

    .risk-section {
        margin-top: 5px;
    }

    .stAlert {
        border-radius: 8px !important;
        border-left: 4px solid #1db954 !important;
    }
    .stAlert > div {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }

    .metadata-label {
        color: #b3b3b3;
        font-size: 13px;
        font-weight: 500;
    }
    .metadata-value {
        color: #ffffff;
        font-size: 14px;
    }

    .header-icon {
        display: inline-block;
        padding: 1px 1px !important;
        border-radius: 8px;
        font-size: 28px;
        line-height: 1;
    }
</style>
""", unsafe_allow_html=True)


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
    col_names = ['engine_id', 'cycle'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in
                                                                                     range(1, 22)]
    test_df = pd.read_csv(f'data/test_{dataset}.txt', sep=r'\s+', header=None, names=col_names)
    rul_df = pd.read_csv(f'data/RUL_{dataset}.txt', sep=r'\s+', header=None, names=['RUL_final'])
    return test_df, rul_df


@st.cache_data
def load_preprocessed_fd001():
    try:
        df = pd.read_csv('data/test_window_fd001_preprocessed.csv.gz', compression='gzip')
        rul_df = pd.read_csv('data/rul_final_fd001.csv.gz', compression='gzip')
        return df, rul_df
    except:
        return None, None


@st.cache_data
def load_preprocessed_fd002():
    try:
        df = pd.read_csv('data/test_window_fd002_preprocessed.csv.gz', compression='gzip')
        rul_df = pd.read_csv('data/rul_final_fd002.csv.gz', compression='gzip')
        return df, rul_df
    except:
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

    st.markdown("""
    <div style="padding: 20px 0 10px 0;">
        <h1 style="font-size: 36px; font-weight: 700; margin: 0; display: flex; align-items: center; gap: 12px;">
            <span class="header-icon">✈️</span>
            Jet Engine Early Warning System
        </h1>
        <p style="color: #b3b3b3; font-size: 16px; margin: 8px 0 0 0; padding-left: 4px;">
            Predictive Maintenance Dashboard for NASA C-MAPSS Turbofan Engines
        </p>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    with st.spinner("Loading model artifacts..."):
        artifacts = load_artifacts()
        preprocessed_fd001, rul_fd001 = load_preprocessed_fd001()
        preprocessed_fd002, rul_fd002 = load_preprocessed_fd002()

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")

        available_datasets = ['FD001', 'FD002']
        selected_dataset = st.selectbox(
            "📁 Dataset",
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
                "🔧 Engine ID",
                engines,
                format_func=lambda x: f"Engine #{x}"
            )
            st.session_state.selected_engine = selected_engine

            engine_data = test_df[test_df['engine_id'] == selected_engine]
            cycles = sorted(engine_data['cycle'].unique())

            st.markdown("### 🔄 Cycle Selection")
            selected_cycle = st.slider(
                "Cycle",
                min_value=min(cycles),
                max_value=max(cycles),
                value=max(cycles),
                step=1,
                label_visibility="collapsed"
            )
            processed_df = test_df

        st.markdown("---")
        predict_button = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

    if predict_button:
        st.session_state.prediction_done = True

        if selected_dataset == 'FD001':
            features = get_features_for_prediction(processed_df, selected_engine, selected_cycle, selected_dataset,
                                                   artifacts)
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

        st.markdown("### 📊 Current Engine Status")
        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card blue">
                <div class="label">Remaining Useful Life</div>
                <div class="value">{rul_pred:.0f}</div>
                <div class="sub">95% CI: [{rul_lower:.0f}, {rul_upper:.0f}]</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            prob_h30 = risks['h30']['probability']
            color = "green" if prob_h30 < 0.3 else "orange" if prob_h30 < 0.6 else "red"
            st.markdown(f"""
            <div class="metric-card {color}">
                <div class="label">Failure Risk (30 cycles)</div>
                <div class="value">{prob_h30:.1%}</div>
                <div class="sub">Threshold: {risks['h30']['threshold']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            anomaly_score = anomaly_scores['OCSVM']['percentile']
            color = "green" if anomaly_score < 90 else "orange" if anomaly_score < 95 else "red"
            st.markdown(f"""
            <div class="metric-card {color}">
                <div class="label">Anomaly Score</div>
                <div class="value">{anomaly_score:.1f}</div>
                <div class="sub">Critical > 95%</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            action = recommendation['action']
            confidence = recommendation['confidence']

            # ============================================================
            # 🎨 تعیین رنگ‌ها بر اساس action (بدون MONITOR)
            # ============================================================
            if action == 'STOP':
                color = '#e53935'
                color_dark = '#c62828'
                glow_color = 'rgba(229, 57, 53, 0.3)'
                text_color = '#ffffff'
            elif action == 'INSPECT':
                color = '#fb8c00'
                color_dark = '#e65100'
                glow_color = 'rgba(251, 140, 0, 0.3)'
                text_color = '#ffffff'
            else:  # CONTINUE
                color = '#1db954'
                color_dark = '#1ed760'
                glow_color = 'rgba(29, 185, 84, 0.3)'
                text_color = '#121212'

            st.markdown(f"""
            <style>
                .rec-card-{action.lower()} {{
                    border: 2px solid transparent !important;
                    transition: all 0.3s ease !important;
                    position: relative !important;
                    overflow: hidden !important;
                }}
                .rec-card-{action.lower()}::before {{
                    content: '' !important;
                    position: absolute !important;
                    top: 0 !important;
                    left: 0 !important;
                    right: 0 !important;
                    height: 3px !important;
                    background: linear-gradient(90deg, {color}, {color_dark}) !important;
                    z-index: 2 !important;
                }}
                .rec-card-{action.lower()}::before {{
                    background: linear-gradient(90deg, {color}, {color_dark}) !important;
                }}
                .rec-card-{action.lower()}:hover {{
                    border-color: {color} !important;
                    box-shadow: 0 8px 30px {glow_color} !important;
                    transform: translateY(-4px);
                }}
                .metric-card.rec-card-{action.lower()}:hover {{
                    border-color: {color} !important;
                }}
                .rec-btn-{action.lower()} {{
                    background: linear-gradient(135deg, {color}, {color_dark}) !important;
                    color: {text_color} !important;
                    box-shadow: 0 4px 20px {glow_color} !important;
                }}
                .rec-btn-{action.lower()}:hover {{
                    transform: scale(1.03);
                }}
            </style>
            <div class="metric-card rec-card rec-card-{action.lower()}">
                <div class="label">Recommendation</div>
                <div class="rec-btn rec-btn-{action.lower()}">{action}</div>
                <div class="sub" style="margin-top: 4px;">Confidence: {confidence}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="risk-section">', unsafe_allow_html=True)
        st.markdown("### ⏱️ Failure Risk by Horizon")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        thresholds = {
            10: risks['h10']['threshold'],
            20: risks['h20']['threshold'],
            30: risks['h30']['threshold']
        }

        for i, h in enumerate([10, 20, 30]):
            with [col1, col2, col3][i]:
                prob = risks[f'h{h}']['probability']
                alert = risks[f'h{h}']['alert']
                threshold = thresholds[h]

                if prob >= 0.6:
                    color = "#e53935"  # قرمز
                    status_text = "HIGH RISK"
                    status_icon = "🔴"
                elif prob >= 0.3:
                    color = "#ff6b35"  # نارنجی
                    status_text = "MED RISK"
                    status_icon = "🟠"
                else:
                    color = "#1db954"  # سبز
                    status_text = "LOW RISK"
                    status_icon = "🟢"

                if alert:
                    status_text = "ALERT"
                    status_icon = "🔴"
                    color = "#e53935"


                st.markdown(f"""
                <div class="risk-box">
                    <div class="horizon">{h} Cycles</div>
                    <div class="horizon"><span style="color: #666; font-size: 11px;">(Threshold: {threshold:.2f})</span></div>
                    <div class="prob" style="color: {color};">{prob:.1%}</div>
                    <div class="status" style="color: {color};">{status_icon} {status_text}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### 🔍 Anomaly Detection Results")
        st.markdown("---")

        anomaly_data = []
        for name, scores in anomaly_scores.items():
            raw_score = scores['raw_score']
            percentile = scores['percentile']
            is_alert = scores['alert']

            # تعیین وضعیت بر اساس درصد
            if percentile >= 95:
                status = 'CRITICAL'
                icon = '🔴'
            elif percentile >= 50:
                status = 'WARNING'
                icon = '🟡'
            else:
                status = 'NORMAL'
                icon = '🟢'

            anomaly_data.append({
                'Method': name,
                'Raw Score': f"{raw_score:.4f}",
                'Percentile': f"{percentile:.1f}th",
                'Status': status,
                'Icon': icon
            })

        anomaly_df = pd.DataFrame(anomaly_data)

        # ============================================================
        # ✅ تابع رنگ‌بندی با پشتیبانی از هر دو نسخه Pandas
        # ============================================================
        def color_status(val):
            if val == 'CRITICAL':
                return 'color: #e53935; font-weight: 600;'
            elif val == 'WARNING':
                return 'color: #f9a825; font-weight: 600;'
            else:
                return 'color: #1db954; font-weight: 600;'

        # بررسی نسخه Pandas و استفاده از متد مناسب
        try:
            # برای Pandas >= 2.1.0
            styled_df = anomaly_df.style.map(color_status, subset=['Status'])
        except AttributeError:
            # برای Pandas < 2.1.0
            styled_df = anomaly_df.style.applymap(color_status, subset=['Status'])

        st.dataframe(
            styled_df,
            column_config={
                "Method": "Method",
                "Raw Score": "Raw Score",
                "Percentile": "Percentile",
                "Icon": "Status",
                "Status": None
            },
            hide_index=True,
            use_container_width=True
        )

        st.markdown("### 🚨 Decision Triggers")
        st.markdown("---")

        triggers = recommendation['triggers']
        if len(triggers) > 1:
            for trigger in triggers:
                st.warning(f"⚠️ {trigger}")
        else:
            st.success(f"✅ {triggers[0]}")

        st.markdown("### 📈 Engine Health Timeline")
        st.markdown("---")

        dropped_sensors = artifacts[selected_dataset]['metadata'].get('dropped_sensors', [])
        raw_test_df, _ = load_raw_data(selected_dataset)
        raw_engine_data = raw_test_df[raw_test_df['engine_id'] == st.session_state.selected_engine]

        sensor_cols = [col for col in raw_engine_data.columns if col.startswith('sensor_')]
        sensor_cols = [col for col in sensor_cols if col not in dropped_sensors]

        col1, col2 = st.columns([2, 1])
        with col1:
            selected_sensor = st.selectbox(
                "📊 Select Sensor to Visualize",
                sensor_cols if sensor_cols else ['sensor_2'],
                format_func=lambda x: x
            )
        with col2:
            show_health = st.checkbox("📊 Show Health Features", value=False)

        rul_data = processed_df[processed_df['engine_id'] == st.session_state.selected_engine]

        if show_health:
            fig = make_subplots(rows=2, cols=1,
                                subplot_titles=("🟢 RUL Over Time", "🔶 Anomaly Score Over Time"),
                                vertical_spacing=0.15)

            fig.add_trace(
                go.Scatter(x=raw_engine_data['cycle'], y=rul_data['RUL'], mode='lines', name='True RUL',
                           line=dict(color='#1db954', width=3)),
                row=1, col=1
            )
            fig.add_hline(y=50, line_dash="dash", line_color="#e53935", annotation_text="Critical", row=1, col=1)

            anomaly_col = 'OCSVM_Anomaly_Score'
            if anomaly_col in processed_df.columns:
                anomaly_data = processed_df[processed_df['engine_id'] == st.session_state.selected_engine]
                fig.add_trace(
                    go.Scatter(x=anomaly_data['cycle'], y=anomaly_data[anomaly_col], mode='lines',
                               name='Anomaly Score',
                               line=dict(color='#ff6b35', width=3)),
                    row=2, col=1
                )
                fig.add_hline(y=95, line_dash="dash", line_color="#e53935", annotation_text="Critical", row=2, col=1)
                fig.add_hline(y=90, line_dash="dot", line_color="#ff6b35", annotation_text="Warning", row=2, col=1)

            fig.update_layout(
                height=500,
                showlegend=True,
                template="plotly_dark",
                paper_bgcolor='#121212',
                plot_bgcolor='#121212',
                font=dict(color='#ffffff')
            )

        else:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(x=raw_engine_data['cycle'], y=raw_engine_data[selected_sensor], mode='lines',
                           name=selected_sensor,
                           line=dict(color='#3498db', width=3))
            )

            fig.add_trace(
                go.Scatter(x=rul_data['cycle'], y=rul_data['RUL'], mode='lines', name='RUL',
                           line=dict(color='#1db954', width=2, dash='dot'), yaxis='y2')
            )

            fig.update_layout(
                yaxis=dict(title=selected_sensor, color='#3498db', gridcolor='#2a2a2a'),
                yaxis2=dict(title='RUL', overlaying='y', side='right', color='#1db954', gridcolor='#2a2a2a'),
                height=400,
                showlegend=True,
                template="plotly_dark",
                paper_bgcolor='#121212',
                plot_bgcolor='#121212',
                font=dict(color='#ffffff'),
                hovermode="x unified"
            )

        fig.add_vline(x=selected_cycle, line_dash="dash", line_color="#e53935",
                      annotation_text="Current Cycle", annotation_position="top")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Model Metadata", expanded=False):
            metadata = artifacts[selected_dataset]['metadata']
            rul_params = artifacts[selected_dataset]['rul_params']
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📁 Dataset Information**")
                st.write(f"- Dataset: {metadata.get('dataset', 'N/A')}")
                st.write(f"- Description: {metadata.get('description', 'N/A')}")
                st.write(f"- Training Date: {metadata.get('training_date', 'N/A')}")
                st.write(f"- Author: {metadata.get('author', 'N/A')}")
            with col2:
                st.markdown("**⚙️ Model Configuration**")
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