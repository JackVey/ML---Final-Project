# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import json
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from pathlib import Path
# import warnings
#
# warnings.filterwarnings('ignore')
#
# st.set_page_config(
#     page_title="Jet Engine Early Warning System",
#     page_icon="🛩️",
#     layout="wide"
# )
#
#
# @st.cache_resource
# def load_artifacts():
#     artifacts = {}
#
#     with open('saved_artifacts/available_datasets.json', 'r') as f:
#         artifacts['available_datasets'] = json.load(f)
#
#     for dataset in ['FD001', 'FD002']:
#         artifacts[dataset] = {}
#         ds_info = artifacts['available_datasets'][dataset]
#
#         artifacts[dataset]['scaler'] = joblib.load(f'saved_artifacts/{ds_info["scaler"]}')
#         artifacts[dataset]['xgb_model'] = joblib.load(f'saved_artifacts/{ds_info["xgb_model"]}')
#         artifacts[dataset]['calibrated_models'] = joblib.load(f'saved_artifacts/{ds_info["calibrated_models"]}')
#         artifacts[dataset]['anomaly_models'] = joblib.load(f'saved_artifacts/{ds_info["anomaly_models"]}')
#         artifacts[dataset]['feature_info'] = joblib.load(f'saved_artifacts/{ds_info["feature_info"]}')
#         artifacts[dataset]['window_info'] = joblib.load(f'saved_artifacts/{ds_info["window_info"]}')
#         artifacts[dataset]['conformal_params'] = joblib.load(f'saved_artifacts/{ds_info["conformal_params"]}')
#         artifacts[dataset]['tuned_thresholds'] = joblib.load(f'saved_artifacts/{ds_info["tuned_thresholds"]}')
#         artifacts[dataset]['decision_params'] = joblib.load(f'saved_artifacts/{ds_info["decision_params"]}')
#         artifacts[dataset]['rul_params'] = joblib.load(f'saved_artifacts/{ds_info["rul_params"]}')
#         artifacts[dataset]['pct_scores_test'] = joblib.load(f'saved_artifacts/{ds_info["pct_scores_test"]}')
#         artifacts[dataset]['pct_scores_val'] = joblib.load(f'saved_artifacts/{ds_info["pct_scores_val"]}')
#
#         if 'feature_names' in ds_info:
#             artifacts[dataset]['feature_names'] = joblib.load(f'saved_artifacts/{ds_info["feature_names"]}')
#         else:
#             artifacts[dataset]['feature_names'] = None
#
#         with open(f'saved_artifacts/{ds_info["metadata"]}', 'r') as f:
#             artifacts[dataset]['metadata'] = json.load(f)
#
#         if dataset == 'FD002':
#             artifacts[dataset]['scaler_dict'] = joblib.load(f'saved_artifacts/{ds_info["scaler_dict"]}')
#             artifacts[dataset]['kmeans'] = joblib.load(f'saved_artifacts/{ds_info["kmeans"]}')
#
#     return artifacts
#
#
# @st.cache_data
# def load_raw_data(dataset):
#     col_names = ['engine_id', 'cycle'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in
#                                                                                      range(1, 22)]
#
#     train_df = pd.read_csv(f'data/train_{dataset}.txt', sep=r'\s+', header=None, names=col_names)
#     test_df = pd.read_csv(f'data/test_{dataset}.txt', sep=r'\s+', header=None, names=col_names)
#     rul_df = pd.read_csv(f'data/RUL_{dataset}.txt', sep=r'\s+', header=None, names=['RUL_final'])
#
#     return train_df, test_df, rul_df
#
#
# def extract_window_features(df, window_info, feature_cols):
#     W = window_info['window_size']
#     df_out = df.copy()
#     grouped = df_out.groupby('engine_id')
#
#     for col in feature_cols:
#         if col not in df.columns:
#             continue
#
#         rolling_obj = grouped[col].rolling(window=W, min_periods=1)
#         df_out[f'{col}_roll_mean'] = rolling_obj.mean().reset_index(level=0, drop=True)
#         df_out[f'{col}_roll_std'] = rolling_obj.std().reset_index(level=0, drop=True).fillna(0)
#         df_out[f'{col}_roll_min'] = rolling_obj.min().reset_index(level=0, drop=True)
#         df_out[f'{col}_roll_max'] = rolling_obj.max().reset_index(level=0, drop=True)
#         df_out[f'{col}_slope'] = df_out.groupby('engine_id')[col].transform(
#             lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0
#         )
#
#         df_out[f'{col}_ewma'] = grouped[col].transform(
#             lambda x: x.ewm(span=W, adjust=False).mean()
#         )
#
#         df_out[f'{col}_diff'] = grouped[col].diff().fillna(0)
#
#     return df_out
#
#
# def preprocess_data(dataset, test_df, rul_df, artifacts):
#     ds_artifacts = artifacts[dataset]
#
#     test_max_cycle = test_df.groupby('engine_id')['cycle'].max().to_dict()
#     rul_mapping = {engine: rul_df.iloc[i, 0] for i, engine in enumerate(test_df['engine_id'].unique())}
#
#     test_df['max_cycle'] = test_df['engine_id'].map(test_max_cycle)
#     test_df['RUL_final'] = test_df['engine_id'].map(rul_mapping)
#     test_df['RUL'] = test_df['max_cycle'] - test_df['cycle'] + test_df['RUL_final']
#
#     rul_cap = 125
#     test_df['RUL_capped'] = test_df['RUL'].clip(upper=rul_cap)
#
#     test_df_raw = test_df.copy()
#
#     feature_info = ds_artifacts['feature_info']
#     features_to_scale = feature_info['all_features']
#     scaler = ds_artifacts['scaler']
#     dropped_sensors = ds_artifacts['feature_info'].get('dropped_sensors', [])
#
#     if dataset == 'FD001':
#         dropped_sensors = artifacts['FD001']['metadata']['dropped_sensors']
#     else:
#         dropped_sensors = artifacts['FD002']['metadata']['dropped_sensors']
#
#     if dropped_sensors:
#         test_df = test_df.drop(columns=dropped_sensors, errors='ignore')
#         test_df_raw = test_df_raw.drop(columns=dropped_sensors, errors='ignore')
#
#     sensor_cols = [col for col in test_df.columns if col.startswith('sensor_')]
#     if dataset == 'FD001':
#         test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])
#     else:
#         op_settings = feature_info['op_settings']
#         test_df[op_settings] = scaler.transform(test_df[op_settings])
#
#         sensor_cols_scaled = feature_info['active_sensors']
#         scaler_dict = ds_artifacts['scaler_dict']
#         kmeans = ds_artifacts['kmeans']
#
#         test_df['regime'] = kmeans.predict(test_df[op_settings])
#
#         for col in sensor_cols_scaled:
#             test_df[col] = test_df[col].astype(float)
#
#         for r in range(6):
#             regime_mask = test_df['regime'] == r
#             if regime_mask.sum() > 0 and r in scaler_dict:
#                 test_df.loc[regime_mask, sensor_cols_scaled] = scaler_dict[r].transform(
#                     test_df.loc[regime_mask, sensor_cols_scaled])
#
#     window_info = ds_artifacts['window_info']
#     feature_cols = window_info['feature_cols']
#
#     active_cols = [col for col in feature_cols if col in test_df.columns]
#     test_df = extract_window_features(test_df, window_info, active_cols)
#
#     for col in sensor_cols:
#         if col in test_df_raw.columns:
#             test_df[col + '_raw'] = test_df_raw[col]
#
#     return test_df
#
#
# def predict_rul(features, dataset, artifacts):
#     ds_artifacts = artifacts[dataset]
#     model = ds_artifacts['xgb_model']
#     conformal_params = ds_artifacts['conformal_params']
#
#     if ds_artifacts['feature_names'] is not None:
#         expected_features = ds_artifacts['feature_names']['all_features']
#         if len(features) != len(expected_features):
#             features = features[:len(expected_features)]
#
#     pred = model.predict(features.reshape(1, -1))[0]
#     pred_capped = np.clip(pred, None, 125)
#
#     if pred_capped <= 50:
#         q = conformal_params['q_95_near_failure']
#     elif pred_capped <= 100:
#         q = conformal_params['q_95_mid_life']
#     else:
#         q = conformal_params['q_95_early_life']
#
#     lower = max(0, pred_capped - q)
#     upper = pred_capped + q
#
#     return pred_capped, lower, upper
#
#
# def predict_failure_risk(features, dataset, artifacts):
#     ds_artifacts = artifacts[dataset]
#     calibrated_models = ds_artifacts['calibrated_models']
#     tuned_thresholds = ds_artifacts['tuned_thresholds']
#     horizons = [10, 20, 30]
#
#     if ds_artifacts['feature_names'] is not None:
#         expected_features = ds_artifacts['feature_names']['all_features']
#         if len(features) != len(expected_features):
#             features = features[:len(expected_features)]
#
#     risks = {}
#     for h in horizons:
#         model = calibrated_models[h]['XGBoost']
#         prob = model.predict_proba(features.reshape(1, -1))[0, 1]
#         threshold = tuned_thresholds[h]['XGBoost']
#         risks[f'h{h}'] = {
#             'probability': prob,
#             'threshold': threshold,
#             'alert': prob >= threshold
#         }
#
#     return risks
#
# def predict_anomaly(features, dataset, artifacts):
#     ds_artifacts = artifacts[dataset]
#     anomaly_models = ds_artifacts['anomaly_models']
#     pct_scores_test = ds_artifacts['pct_scores_test']
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
#         if name in pct_scores_test:
#             ref_scores = pct_scores_test[name]
#             if len(ref_scores) > 0:
#                 if raw_score >= np.max(ref_scores):
#                     percentile = 100.0
#                 elif raw_score <= np.min(ref_scores):
#                     percentile = 0.0
#                 else:
#                     percentile = float(np.interp(raw_score, np.sort(ref_scores), np.linspace(0, 100, len(ref_scores))))
#             else:
#                 percentile = 0.0
#         else:
#             percentile = 0.0
#
#         scores[name] = {
#             'raw_score': float(raw_score),
#             'percentile': float(percentile),
#             'alert': raw_score >= threshold
#         }
#
#     return scores
#
# def make_recommendation(rul_pred, rul_lower, rul_upper, failure_risks, anomaly_scores, dataset, artifacts):
#     prob_h30 = failure_risks['h30']['probability']
#     anomaly_score = anomaly_scores['OCSVM']['percentile']
#     interval_width = rul_upper - rul_lower
#
#     # ====== HARDCODED THRESHOLDS FOR TESTING ======
#     if dataset == 'FD001':
#         stop_rul_threshold = 15
#         stop_prob_threshold = 0.6
#         stop_anomaly_threshold = 97
#
#         inspect_rul_threshold = 25
#         inspect_prob_threshold = 0.2
#         inspect_anomaly_threshold = 92
#         inspect_uncertainty_threshold = 60
#     else:
#         stop_rul_threshold = 15
#         stop_prob_threshold = 0.6
#         stop_anomaly_threshold = 97
#
#         inspect_rul_threshold = 25
#         inspect_prob_threshold = 0.4
#         inspect_anomaly_threshold = 92
#         inspect_uncertainty_threshold = 75
#     # ====== END HARDCODED ======
#
#     # ====== DEBUG SECTION ======
#     st.write("### Debug: Decision Values")
#     debug_data = {
#         'Parameter': [
#             'RUL Lower Bound',
#             'Failure Probability (h30)',
#             'Anomaly Score (OCSVM)',
#             'Interval Width',
#             'STOP - RUL threshold',
#             'STOP - Prob threshold',
#             'STOP - Anomaly threshold',
#             'INSPECT - RUL threshold',
#             'INSPECT - Prob threshold',
#             'INSPECT - Anomaly threshold',
#             'INSPECT - Uncertainty threshold'
#         ],
#         'Value': [
#             f"{rul_lower:.0f}",
#             f"{prob_h30:.1%}",
#             f"{anomaly_score:.1f}",
#             f"{interval_width:.0f}",
#             f"{stop_rul_threshold}",
#             f"{stop_prob_threshold:.0%}",
#             f"{stop_anomaly_threshold}",
#             f"{inspect_rul_threshold}",
#             f"{inspect_prob_threshold:.0%}",
#             f"{inspect_anomaly_threshold}",
#             f"{inspect_uncertainty_threshold}"
#         ],
#         'Status': [
#             'OK' if rul_lower >= stop_rul_threshold else 'LOW',
#             'OK' if prob_h30 <= stop_prob_threshold else 'HIGH',
#             'OK' if anomaly_score <= stop_anomaly_threshold else 'HIGH',
#             'OK' if interval_width <= inspect_uncertainty_threshold else 'WIDE',
#             '-',
#             '-',
#             '-',
#             '-',
#             '-',
#             '-',
#             '-'
#         ]
#     }
#     st.dataframe(pd.DataFrame(debug_data), hide_index=True, use_container_width=True)
#
#     st.write("### Decision Logic")
#     st.write(
#         f"**STOP condition:** ({rul_lower:.0f} < {stop_rul_threshold}) or ({prob_h30:.1%} > {stop_prob_threshold:.0%}) or ({anomaly_score:.1f} > {stop_anomaly_threshold})")
#     st.write(
#         f"**INSPECT condition:** ({rul_lower:.0f} < {inspect_rul_threshold}) or ({prob_h30:.1%} > {inspect_prob_threshold:.0%}) or ({anomaly_score:.1f} > {inspect_anomaly_threshold}) or ({interval_width:.0f} > {inspect_uncertainty_threshold})")
#     # ====== END DEBUG ======
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
#
# def get_dataset_description(dataset):
#     descriptions = {
#         'FD001': '1 condition, 1 fault mode',
#         'FD002': '6 conditions, 1 fault mode'
#     }
#     return descriptions.get(dataset, '')
#
#
# def main():
#     st.title("Jet Engine Early Warning System")
#     st.caption("Predictive Maintenance Dashboard for NASA C-MAPSS Turbofan Engines")
#
#     with st.spinner("Loading model artifacts..."):
#         artifacts = load_artifacts()
#
#     with st.sidebar:
#         st.header("Engine Configuration")
#
#         available_datasets = ['FD001', 'FD002']
#         selected_dataset = st.selectbox(
#             "Select Dataset",
#             available_datasets,
#             format_func=lambda x: f"{x} - {get_dataset_description(x)}"
#         )
#
#         with st.spinner(f"Loading {selected_dataset} data..."):
#             train_df, test_df, rul_df = load_raw_data(selected_dataset)
#             processed_df = preprocess_data(selected_dataset, test_df, rul_df, artifacts)
#
#         engines = sorted(processed_df['engine_id'].unique())
#         selected_engine = st.selectbox(
#             "Select Engine ID",
#             engines,
#             format_func=lambda x: f"Engine #{x}"
#         )
#
#         engine_data = processed_df[processed_df['engine_id'] == selected_engine]
#         cycles = sorted(engine_data['cycle'].unique())
#         selected_cycle = st.slider(
#             "Select Cycle",
#             min_value=min(cycles),
#             max_value=max(cycles),
#             value=max(cycles),
#             step=1
#         )
#
#         predict_button = st.button("Run Prediction", type="primary", use_container_width=True)
#
#     if predict_button or st.session_state.get('prediction_done', False):
#         if predict_button:
#             st.session_state.prediction_done = True
#
#         current_row = engine_data[engine_data['cycle'] == selected_cycle]
#         if len(current_row) == 0:
#             st.error("Invalid selection! Please choose a valid cycle.")
#             return
#
#         if artifacts[selected_dataset]['feature_names'] is not None:
#             expected_cols = artifacts[selected_dataset]['feature_names']['all_features']
#             available_expected = [col for col in expected_cols if col in processed_df.columns]
#             feature_cols = [col for col in available_expected if col in processed_df.columns]
#         else:
#             feature_cols = [col for col in processed_df.columns
#                             if col not in ['engine_id', 'cycle', 'RUL', 'RUL_capped', 'max_cycle', 'RUL_final']]
#             if 'regime' in processed_df.columns:
#                 feature_cols = [col for col in feature_cols if col != 'regime']
#
#         features = current_row[feature_cols].values.flatten()
#
#         if features.dtype == 'object':
#             try:
#                 features = features.astype(float)
#             except:
#                 features = np.array([float(x) if isinstance(x, (int, float)) else 0.0 for x in features])
#
#         if predict_button:
#             with st.spinner("Making predictions..."):
#                 rul_pred, rul_lower, rul_upper = predict_rul(features, selected_dataset, artifacts)
#                 risks = predict_failure_risk(features, selected_dataset, artifacts)
#                 anomaly_scores = predict_anomaly(features, selected_dataset, artifacts)
#                 recommendation = make_recommendation(
#                     rul_pred, rul_lower, rul_upper,
#                     risks, anomaly_scores, selected_dataset, artifacts
#                 )
#
#                 st.session_state.rul_pred = rul_pred
#                 st.session_state.rul_lower = rul_lower
#                 st.session_state.rul_upper = rul_upper
#                 st.session_state.risks = risks
#                 st.session_state.anomaly_scores = anomaly_scores
#                 st.session_state.recommendation = recommendation
#                 st.session_state.processed_df = processed_df
#                 st.session_state.engine_data = engine_data
#                 st.session_state.selected_cycle = selected_cycle
#                 st.session_state.selected_dataset = selected_dataset
#                 st.session_state.artifacts = artifacts
#
#         if st.session_state.get('prediction_done', False):
#             rul_pred = st.session_state.rul_pred
#             rul_lower = st.session_state.rul_lower
#             rul_upper = st.session_state.rul_upper
#             risks = st.session_state.risks
#             anomaly_scores = st.session_state.anomaly_scores
#             recommendation = st.session_state.recommendation
#             processed_df = st.session_state.processed_df
#             engine_data = st.session_state.engine_data
#             selected_cycle = st.session_state.selected_cycle
#             selected_dataset = st.session_state.selected_dataset
#             artifacts = st.session_state.artifacts
#
#             st.subheader("Current Engine Status")
#
#             col1, col2, col3, col4 = st.columns(4)
#
#             with col1:
#                 st.metric(
#                     "Remaining Useful Life",
#                     f"{rul_pred:.0f} cycles",
#                     delta=f"95% CI: [{rul_lower:.0f}, {rul_upper:.0f}]"
#                 )
#
#             with col2:
#                 prob_h30 = risks['h30']['probability']
#                 st.metric(
#                     "Failure Risk (30 cycles)",
#                     f"{prob_h30:.1%}",
#                     delta=f"Threshold: {risks['h30']['threshold']:.2f}"
#                 )
#
#             with col3:
#                 anomaly_score = anomaly_scores['OCSVM']['percentile']
#                 st.metric(
#                     "Anomaly Score",
#                     f"{anomaly_score:.1f}th percentile",
#                     delta="Critical > 95%"
#                 )
#
#             with col4:
#                 color = recommendation['color']
#                 st.markdown(f"""
#                 <div style="padding: 15px; border-radius: 10px; background-color: {color}; text-align: center;">
#                     <h2 style="color: white; margin: 0; font-size: 24px;">{recommendation['action']}</h2>
#                     <p style="color: white; margin: 5px 0 0 0; font-size: 14px;">Confidence: {recommendation['confidence']}</p>
#                 </div>
#                 """, unsafe_allow_html=True)
#
#             st.subheader("Failure Risk by Horizon")
#
#             col1, col2, col3 = st.columns(3)
#             for i, h in enumerate([10, 20, 30]):
#                 with [col1, col2, col3][i]:
#                     prob = risks[f'h{h}']['probability']
#                     alert = risks[f'h{h}']['alert']
#                     st.metric(
#                         f"Risk in {h} cycles",
#                         f"{prob:.1%}",
#                         delta="ALERT" if alert else "Normal"
#                     )
#
#             st.subheader("Anomaly Detection Results")
#
#             anomaly_data = []
#             for name, scores in anomaly_scores.items():
#                 anomaly_data.append({
#                     'Method': name,
#                     'Score': f"{scores['percentile']:.1f}th percentile",
#                     'Status': 'ALERT' if scores['alert'] else 'Normal'
#                 })
#             st.dataframe(pd.DataFrame(anomaly_data), hide_index=True, use_container_width=True)
#
#             st.subheader("Decision Triggers")
#
#             triggers = recommendation['triggers']
#             if len(triggers) > 1:
#                 st.warning("Active triggers:")
#                 for trigger in triggers:
#                     st.write(f"- {trigger}")
#             else:
#                 st.success(triggers[0])
#
#             st.subheader("Engine Health Timeline")
#
#             dropped_sensors = artifacts[selected_dataset]['metadata'].get('dropped_sensors', [])
#
#             sensor_cols = [col for col in processed_df.columns if col.endswith('_raw') and 'sensor_' in col]
#             sensor_cols = [col for col in sensor_cols if col.replace('_raw', '') not in dropped_sensors]
#
#             col1, col2 = st.columns([2, 1])
#             with col1:
#                 selected_sensor = st.selectbox(
#                     "Select Sensor to Visualize",
#                     sensor_cols if sensor_cols else ['sensor_2_raw'],
#                     format_func=lambda x: x.replace('_raw', '')
#                 )
#             with col2:
#                 show_health = st.checkbox("Show Health Features", value=False)
#
#             if show_health:
#                 fig = make_subplots(rows=2, cols=1, subplot_titles=("RUL Over Time", "Anomaly Score Over Time"),
#                                     vertical_spacing=0.15)
#
#                 fig.add_trace(
#                     go.Scatter(x=engine_data['cycle'], y=engine_data['RUL'], mode='lines', name='True RUL',
#                                line=dict(color='green', width=2)),
#                     row=1, col=1
#                 )
#                 fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Critical", row=1, col=1)
#
#                 anomaly_col = 'OCSVM_Anomaly_Score'
#                 if anomaly_col in engine_data.columns:
#                     fig.add_trace(
#                         go.Scatter(x=engine_data['cycle'], y=engine_data[anomaly_col], mode='lines',
#                                    name='Anomaly Score',
#                                    line=dict(color='orange', width=2)),
#                         row=2, col=1
#                     )
#                     fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Critical", row=2, col=1)
#                     fig.add_hline(y=90, line_dash="dot", line_color="orange", annotation_text="Warning", row=2, col=1)
#
#                 fig.update_layout(height=500, showlegend=True)
#
#             else:
#                 fig = go.Figure()
#
#                 fig.add_trace(
#                     go.Scatter(x=engine_data['cycle'], y=engine_data[selected_sensor], mode='lines',
#                                name=selected_sensor.replace('_raw', ''),
#                                line=dict(color='blue', width=2))
#                 )
#
#                 fig.add_trace(
#                     go.Scatter(x=engine_data['cycle'], y=engine_data['RUL'], mode='lines', name='RUL',
#                                line=dict(color='green', width=2, dash='dot'), yaxis='y2')
#                 )
#
#                 fig.update_layout(
#                     yaxis=dict(title=selected_sensor.replace('_raw', '')),
#                     yaxis2=dict(title='RUL', overlaying='y', side='right'),
#                     height=400,
#                     showlegend=True
#                 )
#
#             fig.add_vline(x=selected_cycle, line_dash="dash", line_color="red", annotation_text="Current Cycle",
#                           annotation_position="top")
#             st.plotly_chart(fig, use_container_width=True)
#
#             with st.expander("Model Metadata"):
#                 metadata = artifacts[selected_dataset]['metadata']
#                 rul_params = artifacts[selected_dataset]['rul_params']
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.write("**Dataset Information**")
#                     st.write(f"- Dataset: {metadata.get('dataset', 'N/A')}")
#                     st.write(f"- Description: {metadata.get('description', 'N/A')}")
#                     st.write(f"- Training Date: {metadata.get('training_date', 'N/A')}")
#                     st.write(f"- Author: {metadata.get('author', 'N/A')}")
#                 with col2:
#                     st.write("**Model Configuration**")
#                     st.write(f"- Model Version: {metadata.get('model_version', 'N/A')}")
#                     st.write(f"- Window Size: {metadata.get('window_size', 'N/A')} cycles")
#                     st.write(f"- RUL Cap: {rul_params.get('rul_cap', 125)} cycles")
#                     st.write(f"- Total Features: {metadata.get('total_features', 'N/A')}")
#                     if selected_dataset == 'FD002':
#                         st.write(f"- Number of Regimes: {metadata.get('num_regimes', 'N/A')}")
#
#
# if __name__ == "__main__":
#     main()

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
    col_names = ['engine_id', 'cycle'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in
                                                                                     range(1, 22)]
    test_df = pd.read_csv(f'data/test_{dataset}.txt', sep=r'\s+', header=None, names=col_names)
    rul_df = pd.read_csv(f'data/RUL_{dataset}.txt', sep=r'\s+', header=None, names=['RUL_final'])
    return test_df, rul_df


# def extract_multi_window_features_single_engine(engine_df, window_info, feature_cols):
#     window_sizes = window_info['window_sizes']
#     df_out = engine_df.copy()
#
#     if len(df_out) == 0:
#         return df_out
#
#     grouped = df_out.groupby('engine_id')
#
#     for W in window_sizes:
#         for col in feature_cols:
#             if col not in df_out.columns:
#                 continue
#
#             rolling_obj = grouped[col].rolling(window=W, min_periods=1)
#
#             df_out[f'{col}_roll_mean_W{W}'] = rolling_obj.mean().reset_index(level=0, drop=True)
#             df_out[f'{col}_roll_std_W{W}'] = rolling_obj.std().reset_index(level=0, drop=True).fillna(0)
#             df_out[f'{col}_roll_min_W{W}'] = rolling_obj.min().reset_index(level=0, drop=True)
#             df_out[f'{col}_roll_max_W{W}'] = rolling_obj.max().reset_index(level=0, drop=True)
#
#             slope_col = grouped[col].rolling(window=W, min_periods=2).apply(
#                 lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0,
#                 raw=True
#             )
#             df_out[f'{col}_slope_W{W}'] = slope_col.reset_index(level=0, drop=True).fillna(0)
#
#     return df_out

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

            # ========== اضافه کردن EWMA و DIFF ==========
            df_out[f'{col}_ewma_W{W}'] = grouped[col].apply(
                lambda x: x.ewm(span=W, adjust=False).mean()
            ).reset_index(level=0, drop=True)

            df_out[f'{col}_diff_W{W}'] = grouped[col].diff().fillna(0)
            # ============================================

            slope_col = grouped[col].rolling(window=W, min_periods=2).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0,
                raw=True
            )
            df_out[f'{col}_slope_W{W}'] = slope_col.reset_index(level=0, drop=True).fillna(0)

    return df_out

def preprocess_engine_data(dataset, engine_id, test_df, rul_df, artifacts):
    ds_artifacts = artifacts[dataset]

    engine_df = test_df[test_df['engine_id'] == engine_id].copy()

    if len(engine_df) == 0:
        return pd.DataFrame()

    test_max_cycle = test_df.groupby('engine_id')['cycle'].max().to_dict()
    rul_mapping = {engine: rul_df.iloc[i, 0] for i, engine in enumerate(test_df['engine_id'].unique())}

    engine_df['max_cycle'] = engine_df['engine_id'].map(test_max_cycle)
    engine_df['RUL_final'] = engine_df['engine_id'].map(rul_mapping)
    engine_df['RUL'] = engine_df['max_cycle'] - engine_df['cycle'] + engine_df['RUL_final']

    rul_cap = ds_artifacts['rul_params']['rul_cap']
    engine_df['RUL_capped'] = engine_df['RUL'].clip(upper=rul_cap)

    engine_df_raw = engine_df.copy()

    feature_info = ds_artifacts['feature_info']
    features_to_scale = feature_info['all_features']
    scaler = ds_artifacts['scaler']

    dropped_sensors = ds_artifacts['metadata']['dropped_sensors']
    if dropped_sensors:
        engine_df = engine_df.drop(columns=dropped_sensors, errors='ignore')
        engine_df_raw = engine_df_raw.drop(columns=dropped_sensors, errors='ignore')

    sensor_cols = [col for col in engine_df.columns if col.startswith('sensor_')]

    if dataset == 'FD001':
        engine_df[features_to_scale] = scaler.transform(engine_df[features_to_scale])
    else:
        op_settings = feature_info['op_settings']
        engine_df[op_settings] = scaler.transform(engine_df[op_settings])

        sensor_cols_scaled = feature_info['active_sensors']
        scaler_dict = ds_artifacts['scaler_dict']
        kmeans = ds_artifacts['kmeans']

        engine_df['regime'] = kmeans.predict(engine_df[op_settings])

        for col in sensor_cols_scaled:
            engine_df[col] = engine_df[col].astype(float)

        for r in range(6):
            regime_mask = engine_df['regime'] == r
            if regime_mask.sum() > 0 and r in scaler_dict:
                engine_df.loc[regime_mask, sensor_cols_scaled] = scaler_dict[r].transform(
                    engine_df.loc[regime_mask, sensor_cols_scaled])

    window_info = ds_artifacts['window_info']
    feature_cols = window_info['feature_cols']
    active_cols = [col for col in feature_cols if col in engine_df.columns]
    engine_df = extract_multi_window_features_single_engine(engine_df, window_info, active_cols)

    for col in sensor_cols:
        if col in engine_df_raw.columns:
            engine_df[col + '_raw'] = engine_df_raw[col]

    return engine_df


def get_features_for_prediction(processed_df, selected_cycle, selected_dataset, artifacts):
    current_row = processed_df[processed_df['cycle'] == selected_cycle]

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


# def predict_rul(features, dataset, artifacts):
#     ds_artifacts = artifacts[dataset]
#     model = ds_artifacts['xgb_model']
#     conformal_params = ds_artifacts['conformal_params']
#
#     feature_names = ds_artifacts['feature_names']
#     expected_features = feature_names['all_features']
#
#     if len(features) != len(expected_features):
#         if len(features) < len(expected_features):
#             padded = np.zeros(len(expected_features))
#             padded[:len(features)] = features
#             features = padded
#         else:
#             features = features[:len(expected_features)]
#
#     pred = model.predict(features.reshape(1, -1))[0]
#     rul_cap = ds_artifacts['rul_params']['rul_cap']
#     pred_capped = np.clip(pred, None, rul_cap)
#
#     if pred_capped <= 50:
#         q = conformal_params['q_95_near_failure']
#     elif pred_capped <= 100:
#         q = conformal_params['q_95_mid_life']
#     else:
#         q = conformal_params['q_95_early_life']
#
#     lower = max(0, pred_capped - q)
#     upper = pred_capped + q
#
#     return pred_capped, lower, upper

def predict_rul(features, dataset, artifacts):
    ds_artifacts = artifacts[dataset]
    model = ds_artifacts['xgb_model']
    conformal_params = ds_artifacts['conformal_params']

    feature_names = ds_artifacts['feature_names']
    expected_features = feature_names['all_features']

    # ========== DEBUG ==========
    st.write("### DEBUG: predict_rul")
    st.write(f"Dataset: {dataset}")
    st.write(f"Features length: {len(features)}")
    st.write(f"Expected features length: {len(expected_features)}")

    if len(features) != len(expected_features):
        st.write(f"⚠️ Feature mismatch: got {len(features)}, expected {len(expected_features)}")
        if len(features) < len(expected_features):
            padded = np.zeros(len(expected_features))
            padded[:len(features)] = features
            features = padded
            st.write(f"  Padded to {len(features)}")
        else:
            features = features[:len(expected_features)]
            st.write(f"  Truncated to {len(expected_features)}")

    # نمایش چند ویژگی اول برای بررسی
    st.write(f"First 5 features: {features[:5]}")
    st.write(f"Feature names (first 5): {expected_features[:5]}")
    # ===========================

    pred = model.predict(features.reshape(1, -1))[0]
    rul_cap = ds_artifacts['rul_params']['rul_cap']
    pred_capped = np.clip(pred, None, rul_cap)

    st.write(f"Raw prediction: {pred:.2f}")
    st.write(f"Capped prediction: {pred_capped:.2f}")

    if pred_capped <= 50:
        q = conformal_params['q_95_near_failure']
    elif pred_capped <= 100:
        q = conformal_params['q_95_mid_life']
    else:
        q = conformal_params['q_95_early_life']

    lower = max(0, pred_capped - q)
    upper = pred_capped + q

    st.write(f"q_95: {q:.2f}")
    st.write(f"95% CI: [{lower:.2f}, {upper:.2f}]")

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

    # ========== DEBUG: predict_failure_risk ==========
    st.write("### DEBUG: predict_failure_risk")
    st.write(f"Dataset: {dataset}")
    st.write(f"tuned_thresholds type: {type(tuned_thresholds)}")
    if tuned_thresholds is not None:
        st.write(f"tuned_thresholds keys: {list(tuned_thresholds.keys())}")
        for key in list(tuned_thresholds.keys())[:5]:
            st.write(f"  {key}: {tuned_thresholds[key]}")
    else:
        st.write("tuned_thresholds is None")
    st.write("---")

    risks = {}
    for h in horizons:
        model = calibrated_models[h]['XGBoost']
        prob = model.predict_proba(features.reshape(1, -1))[0, 1]

        st.write(f"Horizon {h}: prob = {prob:.4f}")

        if str(h) in tuned_thresholds:
            threshold = tuned_thresholds[str(h)]['XGBoost']
            st.write(f"  Found in str({h}): {threshold}")
        elif h in tuned_thresholds:
            threshold = tuned_thresholds[h]['XGBoost']
            st.write(f"  Found in {h}: {threshold}")
        else:
            threshold = 0.05
            st.write(f"  NOT found, using default: {threshold}")

        risks[f'h{h}'] = {
            'probability': prob,
            'threshold': threshold,
            'alert': prob >= threshold
        }

    st.write("---")
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

    # ========== DEBUG: predict_anomaly ==========
    st.write("### DEBUG: predict_anomaly")
    st.write(f"Dataset: {dataset}")
    st.write(f"anomaly_models keys: {list(anomaly_models.keys())}")
    st.write(f"pct_scores_test type: {type(pct_scores_test)}")
    if pct_scores_test is not None:
        st.write(f"pct_scores_test keys: {list(pct_scores_test.keys())}")
        for key in pct_scores_test.keys():
            if pct_scores_test[key] is not None:
                st.write(f"  {key}: length={len(pct_scores_test[key])}, min={np.min(pct_scores_test[key]):.3f}, max={np.max(pct_scores_test[key]):.3f}")
            else:
                st.write(f"  {key}: None")
    else:
        st.write("pct_scores_test is None")
    st.write("---")

    scores = {}
    for name, model in anomaly_models.items():
        try:
            if name == 'PCA':
                transformed = model.transform(features.reshape(1, -1))
                reconstructed = model.inverse_transform(transformed)
                raw_score = np.mean((features.reshape(1, -1) - reconstructed) ** 2, axis=1)[0]
            else:
                raw_score = -model.decision_function(features.reshape(1, -1))[0]

            st.write(f"Model {name}: raw_score = {raw_score:.4f}")

            if pct_scores_test is not None and name in pct_scores_test:
                ref_scores = pct_scores_test[name]
                if ref_scores is not None and len(ref_scores) > 0:
                    if raw_score >= np.max(ref_scores):
                        percentile = 100.0
                    elif raw_score <= np.min(ref_scores):
                        percentile = 0.0
                    else:
                        percentile = float(np.interp(raw_score, np.sort(ref_scores), np.linspace(0, 100, len(ref_scores))))
                    st.write(f"  percentile = {percentile:.2f} (from {len(ref_scores)} ref_scores)")
                else:
                    percentile = 50.0
                    st.write(f"  ref_scores is empty, using default: {percentile}")
            else:
                percentile = 50.0
                st.write(f"  pct_scores_test[{name}] not found, using default: {percentile}")

            scores[name] = {
                'raw_score': float(raw_score),
                'percentile': float(percentile),
                'alert': percentile >= 95
            }
        except Exception as e:
            st.write(f"  ERROR in {name}: {str(e)}")
            scores[name] = {
                'raw_score': 0.0,
                'percentile': 50.0,
                'alert': False
            }

    st.write("---")
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


def debug_engine_comparison(processed_df, engine_id, cycle, dataset, artifacts):
    """مقایسه مستقیم ویژگی‌های یک موتور خاص با نوت‌بوک"""

    st.write(f"### 🔍 Debug: Engine {engine_id}, Cycle {cycle}")

    current_row = processed_df[(processed_df['engine_id'] == engine_id) & (processed_df['cycle'] == cycle)]

    if len(current_row) == 0:
        st.error(f"Engine {engine_id}, Cycle {cycle} not found!")
        return

    feature_names = artifacts[dataset]['feature_names']
    expected_features = feature_names['all_features']

    st.write(f"**Expected features count:** {len(expected_features)}")

    # استخراج ویژگی‌ها
    features = []
    missing = []
    for col in expected_features:
        if col in current_row.columns:
            val = current_row[col].values[0]
            if pd.isna(val):
                val = 0.0
            features.append(float(val))
        else:
            features.append(0.0)
            missing.append(col)

    st.write(f"**Extracted features count:** {len(features)}")

    if missing:
        st.warning(f"⚠️ Missing {len(missing)} features: {missing[:5]}...")

    # نمایش 10 ویژگی اول
    st.write("### First 10 features (App):")
    for i, col in enumerate(expected_features[:10]):
        st.write(f"  {i}: {col} = {features[i]:.6f}")

    # مقایسه با مقادیر نوت‌بوک (Hardcoded از خروجی نوت‌بوک)
    notebook_values = {
        'op_setting_1': 0.747954,
        'op_setting_2': 0.865002,
        'op_setting_3': 0.417670,
        'sensor_1': 0.000000,
        'sensor_2': -0.020850,
        'sensor_3': 0.125843,
        'sensor_4': -0.101434,
        'sensor_5': -0.000000,
        'sensor_6': -0.174690,
        'sensor_7': 1.840167,
    }

    st.write("### Comparison with Notebook (first 10):")
    comparison = []
    for i, col in enumerate(expected_features[:10]):
        if col in notebook_values:
            app_val = features[i]
            nb_val = notebook_values[col]
            diff = abs(app_val - nb_val)
            comparison.append({
                'Feature': col,
                'Notebook': nb_val,
                'App': app_val,
                'Diff': diff,
                'Match': '✅' if diff < 0.001 else '❌'
            })

    st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)

    # پیش‌بینی RUL
    features_array = np.array(features).reshape(1, -1)
    model = artifacts[dataset]['xgb_model']
    pred = model.predict(features_array)[0]
    st.write(f"**App RUL Prediction:** {pred:.2f}")
    st.write(f"**Notebook RUL Prediction:** 113.53")
    st.write(f"**True RUL (Notebook):** 124")

    return features


def debug_scaling(processed_df, engine_id, cycle, dataset, artifacts):
    """بررسی scaling در app"""

    st.write("### 🔍 Debug: Scaling Check")

    current_row = processed_df[(processed_df['engine_id'] == engine_id) & (processed_df['cycle'] == cycle)]

    if len(current_row) == 0:
        st.error("Not found!")
        return

    test_df, rul_df = load_raw_data(dataset)
    raw_row = test_df[(test_df['engine_id'] == engine_id) & (test_df['cycle'] == cycle)]

    if len(raw_row) == 0:
        st.error("Raw data not found!")
        return

    st.write("### Raw Sensor Values (before scaling):")
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    raw_data = {}
    for col in sensor_cols:
        if col in raw_row.columns:
            raw_data[col] = raw_row[col].values[0]

    raw_df = pd.DataFrame(list(raw_data.items()), columns=['Sensor', 'Raw Value'])
    st.dataframe(raw_df, hide_index=True, use_container_width=True)

    st.write("### Scaled Sensor Values (after scaling):")
    scaled_data = {}
    for col in sensor_cols:
        if col in current_row.columns:
            scaled_data[col] = current_row[col].values[0]

    scaled_df = pd.DataFrame(list(scaled_data.items()), columns=['Sensor', 'Scaled Value'])
    st.dataframe(scaled_df, hide_index=True, use_container_width=True)

    if 'regime' in current_row.columns:
        st.write(f"**Regime:** {current_row['regime'].values[0]}")

    scaler_dict = artifacts[dataset]['scaler_dict']
    st.write(f"**Available scalers:** {list(scaler_dict.keys())}")

    notebook_values = {
        'sensor_1': 0.0,
        'sensor_2': -0.02085,
        'sensor_3': 0.125843,
        'sensor_4': -0.101434,
        'sensor_5': 0.0,
        'sensor_6': -0.17469,
        'sensor_7': 1.840167,
    }

    st.write("### Comparison with Notebook:")
    comparison = []
    for col in ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7']:
        if col in current_row.columns and col in notebook_values:
            app_val = current_row[col].values[0]
            nb_val = notebook_values[col]
            comparison.append({
                'Sensor': col,
                'Notebook': nb_val,
                'App': app_val,
                'Diff': abs(app_val - nb_val),
                'Match': '✅' if abs(app_val - nb_val) < 0.001 else '❌'
            })

    st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)

def main():
    initialize_session_state()

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
            test_df, rul_df = load_raw_data(selected_dataset)

        engines = sorted(test_df['engine_id'].unique())
        selected_engine = st.selectbox(
            "Select Engine ID",
            engines,
            format_func=lambda x: f"Engine #{x}"
        )

        with st.spinner(f"Processing engine {selected_engine} data..."):
            processed_df = preprocess_engine_data(
                selected_dataset, selected_engine, test_df, rul_df, artifacts
            )

        if len(processed_df) == 0:
            st.error(f"No data found for engine {selected_engine}")
            return

        cycles = sorted(processed_df['cycle'].unique())
        selected_cycle = st.slider(
            "Select Cycle",
            min_value=min(cycles),
            max_value=max(cycles),
            value=max(cycles),
            step=1
        )

        predict_button = st.button("Run Prediction", type="primary", use_container_width=True)

    if predict_button:
        st.session_state.prediction_done = True

        features = get_features_for_prediction(processed_df, selected_cycle, selected_dataset, artifacts)

        if features is None:
            st.error("Could not extract features for prediction")
            return

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
                go.Scatter(x=processed_df['cycle'], y=processed_df['RUL'], mode='lines', name='True RUL',
                           line=dict(color='green', width=2)),
                row=1, col=1
            )
            fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Critical", row=1, col=1)

            anomaly_col = 'OCSVM_Anomaly_Score'
            if anomaly_col in processed_df.columns:
                fig.add_trace(
                    go.Scatter(x=processed_df['cycle'], y=processed_df[anomaly_col], mode='lines',
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
                go.Scatter(x=processed_df['cycle'], y=processed_df[selected_sensor], mode='lines',
                           name=selected_sensor.replace('_raw', ''),
                           line=dict(color='blue', width=2))
            )

            fig.add_trace(
                go.Scatter(x=processed_df['cycle'], y=processed_df['RUL'], mode='lines', name='RUL',
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
                window_sizes = metadata.get('window_sizes', [])
                if isinstance(window_sizes, list):
                    st.write(f"- Window Sizes: {', '.join(map(str, window_sizes))} cycles")
                else:
                    st.write(f"- Window Size: {window_sizes} cycles")
                st.write(f"- RUL Cap: {rul_params.get('rul_cap', 125)} cycles")
                st.write(f"- Total Features: {metadata.get('total_features', 'N/A')}")
                if selected_dataset == 'FD002':
                    st.write(f"- Number of Regimes: {metadata.get('num_regimes', 'N/A')}")

    # بعد از پردازش داده‌ها و قبل از predict_button
    # در بخش main، بعد از processed_df
    with st.expander("🔧 Debug Tools"):
        debug_engine = st.number_input("Debug Engine ID", value=68, step=1)
        debug_cycle = st.number_input("Debug Cycle", value=150, step=1)
        if st.button("Run Debug Comparison"):
            debug_engine_comparison(processed_df, debug_engine, debug_cycle, selected_dataset, artifacts)

    # در بخش main، بعد از processed_df و قبل از predict_button

    with st.expander("🔧 Debug Tools"):
        st.write("Use these tools to debug feature extraction and scaling")

        debug_engine = st.number_input("Debug Engine ID", value=68, step=1, key="debug_engine")
        debug_cycle = st.number_input("Debug Cycle", value=150, step=1, key="debug_cycle")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run Scaling Debug", key="debug_scaling_btn"):
                debug_scaling(processed_df, debug_engine, debug_cycle, selected_dataset, artifacts)
        with col2:
            if st.button("Run Feature Comparison", key="debug_features_btn"):
                debug_engine_comparison(processed_df, debug_engine, debug_cycle, selected_dataset, artifacts)


if __name__ == "__main__":
    main()
