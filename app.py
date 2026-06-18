"""
dashboard/app.py — Streamlit 1 Hz real-time monitoring application for BBM92 QKD.

Architecture (§8):
    - Simulation Mode: Replays telemetry_86400.parquet at configurable speed
    - Live Mode: Accepts real telemetry via UDP socket (interface documented)

Layout (5 panels + statistics):
    ┌─────────────────────────────────────────┐
    │ Header                                  │
    ├──────────────┬──────────┬───────────────┤
    │ Live QBER    │ Gauge    │ Bell S        │
    ├──────────────┴──────────┴───────────────┤
    │ SHAP Attribution   │  Alert Log         │
    ├────────────────────┴────────────────────┤
    │ Dataset Statistics                      │
    └─────────────────────────────────────────┘

All state held in st.session_state. Updates via st.empty() placeholders.
Dark theme with professional defence-research styling.

"""

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.privacy_amplification import secure_key_rate
from ml.features import WINDOW, extract_features_single
from ml.inference import load_model, predict_single
from ml.explain import build_shap_explainer, explain_single_alert


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM THEME & CSS
# ═══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0a1628 0%, #1a1a2e 50%, #16213e 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid #1e3a5f;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 100, 255, 0.08);
    }
    .main-header h1 {
        color: #00d4ff;
        font-size: 1.8rem;
        margin-bottom: 0.3rem;
        letter-spacing: 0.5px;
    }
    .main-header p {
        color: #8899aa;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Panel containers */
    .panel-container {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
    }

    /* Status bar */
    .status-normal { color: #00ff88; font-weight: bold; }
    .status-warning { color: #ffaa00; font-weight: bold; }
    .status-critical {
        color: #ff3344;
        font-weight: bold;
        animation: pulse 1s ease-in-out infinite alternate;
    }
    @keyframes pulse {
        from { opacity: 0.7; }
        to { opacity: 1.0; }
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2a3a5a;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: bold;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #8899aa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0a1628;
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #c0d0e0;
    }

    /* Hide default Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Plotly chart containers */
    .stPlotlyChart {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL RENDERING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#1a1a2e',
    font=dict(color='#c0d0e0', size=11),
    margin=dict(l=50, r=50, t=45, b=30),
    legend=dict(
        bgcolor='rgba(0,0,0,0.3)',
        bordercolor='#333',
        borderwidth=1,
        font=dict(size=10),
    ),
)


def render_qber_panel(
    qber_history: list,
    prob_history: list,
    timestamps: list,
) -> go.Figure:
    """Panel 1: Live QBER time-series with colour-coded zones."""
    fig = go.Figure()

    # Background zones
    fig.add_hrect(y0=0, y1=0.05, fillcolor='#00ff88', opacity=0.04,
                  line_width=0, layer='below')
    fig.add_hrect(y0=0.05, y1=0.10, fillcolor='#ffaa00', opacity=0.04,
                  line_width=0, layer='below')
    fig.add_hrect(y0=0.10, y1=0.35, fillcolor='#ff3344', opacity=0.04,
                  line_width=0, layer='below')

    # QBER trace
    fig.add_trace(go.Scatter(
        x=timestamps, y=qber_history,
        mode='lines', name='QBER',
        line=dict(color='#00d4ff', width=2),
        yaxis='y1',
    ))

    # Attack probability overlay
    fig.add_trace(go.Scatter(
        x=timestamps, y=prob_history,
        mode='lines', name='Attack Prob',
        line=dict(color='#ff6b6b', width=1.5, dash='dot'),
        yaxis='y2',
    ))

    fig.update_layout(
        title=dict(text='Live QBER + Attack Probability', font=dict(size=14)),
        yaxis=dict(
            title='QBER', range=[0, 0.30],
            gridcolor='#333', zerolinecolor='#444',
        ),
        yaxis2=dict(
            title='Attack Probability',
            overlaying='y', side='right', range=[0, 1],
            gridcolor='#333',
        ),
        height=350,
        legend=dict(
            **PLOTLY_LAYOUT_DEFAULTS.get('legend', {}),
            orientation='h', y=-0.18, x=0.5, xanchor='center'
        ),
        **{k: v for k, v in PLOTLY_LAYOUT_DEFAULTS.items() if k != 'legend'}
    )
    return fig


def render_bell_panel(
    bell_S_history: list,
    timestamps: list,
) -> go.Figure:
    """Panel 2: Bell parameter (CHSH S) — entanglement health monitor."""
    fig = go.Figure()

    # Reference lines
    fig.add_hline(y=2.828, line_dash='dash', line_color='#00ff88',
                  annotation_text='Tsirelson: 2√2',
                  annotation_position='top right',
                  annotation_font_color='#00ff88',
                  annotation_font_size=9)
    fig.add_hline(y=2.0, line_dash='dot', line_color='#ff3344',
                  annotation_text='Classical: 2.0',
                  annotation_position='bottom right',
                  annotation_font_color='#ff3344',
                  annotation_font_size=9)
    fig.add_hline(y=2.5, line_dash='dash', line_color='#ffaa00',
                  annotation_text='Warning',
                  annotation_position='bottom right',
                  annotation_font_color='#ffaa00',
                  annotation_font_size=9)

    fig.add_trace(go.Scatter(
        x=timestamps, y=bell_S_history,
        mode='lines', name='Bell S (CHSH)',
        line=dict(color='#2ca02c', width=2),
        fill='tozeroy', fillcolor='rgba(44,160,44,0.03)',
    ))

    fig.update_layout(
        title=dict(text='Bell Parameter S — Entanglement Health', font=dict(size=13)),
        yaxis=dict(
            title='S parameter', range=[1.8, 2.9],
            gridcolor='#333', zerolinecolor='#444',
        ),
        height=350,
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def render_gauge_panel(
    current_prob: float,
    threshold: float,
) -> go.Figure:
    """Panel 3: Threat probability gauge indicator."""
    if current_prob >= 0.65:
        color = '#ff3344'
    elif current_prob >= threshold:
        color = '#ffaa00'
    else:
        color = '#00ff88'

    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=current_prob,
        number={'suffix': '', 'font': {'size': 32, 'color': color}},
        delta={'reference': threshold, 'decreasing': {'color': '#00ff88'},
               'increasing': {'color': '#ff3344'}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': '#666',
                     'dtick': 0.2},
            'bgcolor': '#1a1a2e',
            'borderwidth': 2,
            'bordercolor': '#333',
            'bar': {'color': color, 'thickness': 0.3},
            'steps': [
                {'range': [0, threshold * 0.7],
                 'color': 'rgba(0,255,136,0.08)'},
                {'range': [threshold * 0.7, threshold],
                 'color': 'rgba(255,170,0,0.12)'},
                {'range': [threshold, 1.0],
                 'color': 'rgba(255,51,68,0.12)'},
            ],
            'threshold': {
                'line': {'color': '#ffffff', 'width': 3},
                'thickness': 0.75,
                'value': threshold,
            },
        },
        title={
            'text': (
                f'Attack Probability<br>'
                f'<span style="font-size:11px;color:#888">'
                f'Threshold: {threshold:.2f}</span>'
            ),
            'font': {'size': 14},
        },
    ))
    fig.update_layout(
        height=280,
        **{k: v for k, v in PLOTLY_LAYOUT_DEFAULTS.items() if k != 'margin'},
        margin=dict(l=30, r=30, t=70, b=20),
    )
    return fig


def render_shap_panel(shap_data: dict) -> go.Figure:
    """Panel 4: SHAP feature attribution horizontal bar chart."""
    names = shap_data['feature_names']
    values = shap_data['shap_values']
    colors = ['#ff3344' if v > 0 else '#00d4ff' for v in values]

    fig = go.Figure(go.Bar(
        x=values[::-1], y=names[::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=[f'{v:+.4f}' for v in values[::-1]],
        textposition='outside',
        textfont=dict(size=10),
    ))
    fig.update_layout(
        title=dict(
            text='SHAP Attribution — Why This Alert Was Triggered',
            font=dict(size=13),
        ),
        xaxis_title='SHAP Value (impact on model output)',
        height=280,
        **{k: v for k, v in PLOTLY_LAYOUT_DEFAULTS.items() if k != 'margin'},
        margin=dict(l=180, r=80, t=45, b=30),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Page config ───
st.set_page_config(
    page_title='BBM92 QKD Anomaly Detector — DRDO SSPL',
    page_icon='🔐',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Inject custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─── Load models (cached) ───
@st.cache_resource
def load_models():
    """Loads trained model and SHAP explainer (cached across reruns)."""
    model_path = PROJECT_ROOT / 'models'
    data_path = PROJECT_ROOT / 'data'

    model_artifacts = load_model(str(model_path) + '/')

    # Load 24-dim feature vectors for SHAP background (not raw 6-col data)
    x_bg_path = data_path / 'X.npy'
    if x_bg_path.exists():
        X_all = np.load(str(x_bg_path))
        rng = np.random.default_rng(42)
        bg_indices = rng.choice(len(X_all), size=min(500, len(X_all)),
                                replace=False)
        X_bg = X_all[bg_indices]
    else:
        X_bg = np.zeros((500, 24))

    explainer = build_shap_explainer(model_artifacts, X_bg)
    return model_artifacts, explainer


# ─── Sidebar controls ───
st.sidebar.markdown(
    '<div style="text-align:center; padding:1rem 0;">'
    '<span style="font-size:2rem">🔐</span><br>'
    '<span style="font-size:1.1rem; font-weight:bold; color:#00d4ff;">'
    'BBM92 QKD</span><br>'
    '<span style="font-size:0.75rem; color:#667788;">DRDO-SSPL Monitor</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown('---')

st.sidebar.markdown('#### ⚙️ Control Panel')
mode = st.sidebar.radio(
    'Execution Mode', ['Simulation', 'Live UDP'], index=0,
    help='Simulation replays telemetry file. Live UDP accepts real-time data.',
)
speed = st.sidebar.select_slider(
    'Playback Speed',
    options=[1, 5, 10, 30, 60], value=10,
    help='1 = real-time 1Hz, 60 = fastest replay',
)
show_shap = st.sidebar.checkbox('Show SHAP on Alerts', value=True)

st.sidebar.markdown('---')
st.sidebar.markdown('#### 🎯 Detection Settings')
threshold_override = st.sidebar.slider(
    'Detection Threshold', 0.20, 0.50, 0.30, 0.01,
    help='Lower = more sensitive (more alerts). Higher = more specific.',
)



# ─── Header ───
st.markdown(
    '<div class="main-header">'
    '<h1>🔐 BBM92 QKD Real-Time Anomaly Detection</h1>'
    '<p><strong>DRDO · Solid State Physics Laboratory (SSPL), Delhi</strong> · '
    'Entanglement-Based QKD Network Security Monitor</p>'
    '</div>',
    unsafe_allow_html=True,
)


# ─── Session state initialisation ───
if 'qber_history' not in st.session_state:
    st.session_state.qber_history = deque(maxlen=300)  # 5 minutes
    st.session_state.bell_history = deque(maxlen=300)
    st.session_state.prob_history = deque(maxlen=300)
    st.session_state.ts_history = deque(maxlen=300)
    st.session_state.alert_log = []
    st.session_state.window_buffer = deque(maxlen=WINDOW)
    st.session_state.tick = 0
    st.session_state.shap_cache = None
    st.session_state.total_alerts = 0
    st.session_state.max_prob = 0.0


# ─── Main layout ───
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    qber_placeholder = st.empty()
with col2:
    gauge_placeholder = st.empty()
with col3:
    bell_placeholder = st.empty()

# Metrics row
m1, m2, m3, m4, m5 = st.columns(5)
metric_placeholders = [m1.empty(), m2.empty(), m3.empty(),
                       m4.empty(), m5.empty()]

shap_col, log_col = st.columns([1, 2])
with shap_col:
    shap_placeholder = st.empty()
with log_col:
    log_placeholder = st.empty()

status_bar = st.empty()

# ─── Dataset statistics section ───
stats_expander = st.expander('📊 Dataset Statistics', expanded=False)


# ─── Load and display dataset stats ───
def display_dataset_stats():
    """Renders the dataset statistics panel."""
    telemetry_path = PROJECT_ROOT / 'data' / 'telemetry_86400.parquet'
    if not telemetry_path.exists():
        stats_expander.warning('No telemetry dataset found.')
        return

    df = pd.read_parquet(str(telemetry_path))
    with stats_expander:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric('Total Rows', f'{len(df):,}')
        sc2.metric('Normal', f'{(df["label"]==0).sum():,}')
        sc3.metric('Attack', f'{(df["label"]==1).sum():,}')
        sc4.metric('Attack %', f'{df["label"].mean()*100:.1f}%')

        st.markdown('##### Descriptive Statistics')
        num_cols = ['qber', 'bell_S', 'coincidence_rate', 'visibility',
                    'channel_loss_dB', 'detection_rate']
        stats_df = df[num_cols].describe().T
        stats_df.columns = ['Count', 'Mean', 'Std', 'Min', '25%',
                            '50%', '75%', 'Max']
        st.dataframe(stats_df.style.format('{:.4f}'), use_container_width=True)

        # Attack distribution by hour
        st.markdown('##### Attack Distribution by Hour')
        df['hour'] = df['timestamp'].dt.hour
        attack_by_hour = df.groupby('hour')['label'].sum()

        fig_hist = go.Figure(go.Bar(
            x=attack_by_hour.index,
            y=attack_by_hour.values,
            marker_color='#ff6b6b',
            marker_line_width=0,
        ))
        fig_hist.update_layout(
            xaxis_title='Hour of Day',
            yaxis_title='Attack Seconds',
            height=250,
            **PLOTLY_LAYOUT_DEFAULTS,
        )
        st.plotly_chart(fig_hist, use_container_width=True)


display_dataset_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

if mode == 'Simulation':
    telemetry_path = PROJECT_ROOT / 'data' / 'telemetry_86400.parquet'
    if not telemetry_path.exists():
        st.error(
            '⚠️ Telemetry file not found. Run telemetry generation first:\n\n'
            '```bash\n'
            'cd bbm92_drdo\n'
            'python -c "from core.telemetry import build_telemetry_dataset; '
            'build_telemetry_dataset()"\n'
            '```'
        )
        st.stop()

    telemetry_df = pd.read_parquet(str(telemetry_path))

    # Load models
    try:
        model_artifacts, explainer = load_models()
        model_artifacts['config']['threshold'] = threshold_override
        models_loaded = True
    except Exception as e:
        st.error(f'⚠️ Model loading failed: {e}. Train models first.')
        models_loaded = False

    is_running = st.toggle('▶️ Run Simulation', value=False)

    if is_running and models_loaded:
        for idx in range(st.session_state.tick, len(telemetry_df)):
            row = telemetry_df.iloc[idx]

            # Buffer for feature extraction
            st.session_state.window_buffer.append({
                'qber': row['qber'],
                'bell_S': row['bell_S'],
                'coincidence_rate': row['coincidence_rate'],
                'visibility': row['visibility'],
                'channel_loss_dB': row['channel_loss_dB'],
                'detection_rate': row['detection_rate'],
            })

            ts = row['timestamp']
            qber_val = float(row['qber'])
            bell_val = float(row['bell_S'])
            coinc_val = float(row['coincidence_rate'])

            st.session_state.qber_history.append(qber_val)
            st.session_state.bell_history.append(bell_val)
            st.session_state.ts_history.append(ts)

            # ── Inference (only when window is full) ──
            prob = 0.0
            result = {'severity': 'NORMAL', 'probability': 0.0, 'is_attack': False}
            shap_data = None

            if len(st.session_state.window_buffer) == WINDOW:
                window_df = pd.DataFrame(list(st.session_state.window_buffer))
                features = extract_features_single(window_df)
                result = predict_single(features, model_artifacts)
                prob = result['probability']
                st.session_state.max_prob = max(
                    st.session_state.max_prob, prob
                )

                if result['is_attack'] and show_shap:
                    shap_data = explain_single_alert(
                        features, explainer, model_artifacts
                    )
                    st.session_state.shap_cache = shap_data

                if result['is_attack']:
                    st.session_state.total_alerts += 1
                    skr = secure_key_rate(qber_val)
                    alert_entry = {
                        'Timestamp': str(ts),
                        'QBER (%)': f'{qber_val * 100:.2f}',
                        'Bell S': f'{bell_val:.3f}',
                        'Attack Prob': f'{prob:.3f}',
                        'Severity': result['severity'],
                        'SKR (bps)': f'{skr:.0f}',
                        'Top Feature': (
                            shap_data['feature_names'][0]
                            if shap_data else 'N/A'
                        ),
                    }
                    st.session_state.alert_log.append(alert_entry)

            st.session_state.prob_history.append(prob)

            # ── Update all panels ──
            qber_list = list(st.session_state.qber_history)
            bell_list = list(st.session_state.bell_history)
            prob_list = list(st.session_state.prob_history)
            ts_list = list(st.session_state.ts_history)

            with qber_placeholder.container():
                st.plotly_chart(
                    render_qber_panel(qber_list, prob_list, ts_list),
                    use_container_width=True, key=f'qber_{idx}',
                )

            with gauge_placeholder.container():
                st.plotly_chart(
                    render_gauge_panel(prob, threshold_override),
                    use_container_width=True, key=f'gauge_{idx}',
                )

            with bell_placeholder.container():
                st.plotly_chart(
                    render_bell_panel(bell_list, ts_list),
                    use_container_width=True, key=f'bell_{idx}',
                )

            # Metrics row
            skr_current = secure_key_rate(qber_val)
            metric_placeholders[0].metric('⏱️ Time', f't={idx}s')
            metric_placeholders[1].metric(
                '📊 QBER', f'{qber_val * 100:.2f}%'
            )
            metric_placeholders[2].metric('🔔 Bell S', f'{bell_val:.3f}')
            metric_placeholders[3].metric('🔑 SKR', f'{skr_current:.0f} bps')
            metric_placeholders[4].metric(
                '🚨 Alerts', str(st.session_state.total_alerts)
            )

            if st.session_state.shap_cache and show_shap:
                with shap_placeholder.container():
                    st.plotly_chart(
                        render_shap_panel(st.session_state.shap_cache),
                        use_container_width=True, key=f'shap_{idx}',
                    )

            if st.session_state.alert_log:
                with log_placeholder.container():
                    log_df = pd.DataFrame(
                        st.session_state.alert_log[-50:]
                    )
                    st.dataframe(
                        log_df.style.map(
                            lambda v: (
                                'background-color: rgba(255,51,68,0.2); color: #ff6b6b'
                                if v == 'CRITICAL'
                                else (
                                    'background-color: rgba(255,170,0,0.15); color: #ffaa00'
                                    if v == 'WARNING'
                                    else ''
                                )
                            ),
                            subset=['Severity'],
                        ),
                        use_container_width=True,
                        height=300,
                    )

            # Status bar
            if result['is_attack']:
                sev = result['severity']
                sev_class = 'status-critical' if sev == 'CRITICAL' else 'status-warning'
                status_bar.markdown(
                    f'<div style="padding:0.5rem; border-radius:6px; '
                    f'background:#1a1a2e; border:1px solid #2a2a4a;">'
                    f'⏱️ <strong>t = {idx}s</strong> │ '
                    f'QBER: <strong>{qber_val * 100:.2f}%</strong> │ '
                    f'Bell S: <strong>{bell_val:.3f}</strong> │ '
                    f'Coincidence: <strong>{coinc_val:.0f}/s</strong> │ '
                    f'Status: <span class="{sev_class}">🚨 {sev}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                status_bar.markdown(
                    f'<div style="padding:0.5rem; border-radius:6px; '
                    f'background:#1a1a2e; border:1px solid #2a2a4a;">'
                    f'⏱️ <strong>t = {idx}s</strong> │ '
                    f'QBER: <strong>{qber_val * 100:.2f}%</strong> │ '
                    f'Bell S: <strong>{bell_val:.3f}</strong> │ '
                    f'Coincidence: <strong>{coinc_val:.0f}/s</strong> │ '
                    f'Status: <span class="status-normal">✅ NORMAL</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.session_state.tick = idx + 1
            time.sleep(1.0 / speed)

    elif not is_running and st.session_state.tick > 0:
        # Render paused state
        idx = st.session_state.tick - 1
        qber_val = st.session_state.qber_history[-1]
        bell_val = st.session_state.bell_history[-1]
        prob = st.session_state.prob_history[-1]
        
        qber_list = list(st.session_state.qber_history)
        bell_list = list(st.session_state.bell_history)
        prob_list = list(st.session_state.prob_history)
        ts_list = list(st.session_state.ts_history)

        with qber_placeholder.container():
            st.plotly_chart(
                render_qber_panel(qber_list, prob_list, ts_list),
                use_container_width=True, key=f'qber_paused',
            )

        with gauge_placeholder.container():
            st.plotly_chart(
                render_gauge_panel(prob, threshold_override),
                use_container_width=True, key=f'gauge_paused',
            )

        with bell_placeholder.container():
            st.plotly_chart(
                render_bell_panel(bell_list, ts_list),
                use_container_width=True, key=f'bell_paused',
            )

        skr_current = secure_key_rate(qber_val)
        metric_placeholders[0].metric('⏱️ Time', f't={idx}s (PAUSED)')
        metric_placeholders[1].metric('📊 QBER', f'{qber_val * 100:.2f}%')
        metric_placeholders[2].metric('🔔 Bell S', f'{bell_val:.3f}')
        metric_placeholders[3].metric('🔑 SKR', f'{skr_current:.0f} bps')
        metric_placeholders[4].metric('🚨 Alerts', str(st.session_state.total_alerts))

        if st.session_state.shap_cache and show_shap:
            with shap_placeholder.container():
                st.plotly_chart(
                    render_shap_panel(st.session_state.shap_cache),
                    use_container_width=True, key=f'shap_paused',
                )

        if st.session_state.alert_log:
            with log_placeholder.container():
                log_df = pd.DataFrame(st.session_state.alert_log[-50:])
                st.dataframe(
                    log_df.style.map(
                        lambda v: (
                            'background-color: rgba(255,51,68,0.2); color: #ff6b6b'
                            if v == 'CRITICAL'
                            else (
                                'background-color: rgba(255,170,0,0.15); color: #ffaa00'
                                if v == 'WARNING'
                                else ''
                            )
                        ),
                        subset=['Severity'],
                    ),
                    use_container_width=True,
                    height=300,
                )

elif mode == 'Live UDP':
    st.info(
        '🔌 **Live UDP Mode**\n\n'
        'This mode accepts real-time telemetry from a physical BBM92 QKD testbed.\n\n'
        '**Protocol:** UDP datagrams on port `5555`\n\n'
        '**Payload format** (JSON):\n'
        '```json\n'
        '{\n'
        '  "qber": 0.023,\n'
        '  "bell_S": 2.74,\n'
        '  "coincidence_rate": 8500,\n'
        '  "visibility": 0.95,\n'
        '  "channel_loss_dB": 4.2,\n'
        '  "detection_rate": 17200\n'
        '}\n'
        '```\n\n'
        'Connect your telemetry source and press Start.'
    )
    st.warning(
        'Live UDP ingestion requires a physical QKD testbed connection. '
        'Use Simulation mode for development and testing.'
    )

    try:
        model_artifacts, explainer = load_models()
        model_artifacts['config']['threshold'] = threshold_override
        models_loaded = True
    except Exception as e:
        st.error(f'⚠️ Model loading failed: {e}. Train models first.')
        models_loaded = False

    is_running = st.toggle('▶️ Start Listening on Port 5555', value=False)
    
    if is_running and models_loaded:
        import socket
        import json
        import datetime
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('127.0.0.1', 5555))
        except OSError:
            pass
        sock.settimeout(1.0)
        
        idx = st.session_state.tick
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                payload = json.loads(data.decode('utf-8'))
                
                row = pd.Series({
                    'timestamp': datetime.datetime.now(),
                    'qber': payload.get('qber', 0.0),
                    'bell_S': payload.get('bell_S', 0.0),
                    'coincidence_rate': payload.get('coincidence_rate', 0.0),
                    'visibility': payload.get('visibility', 0.0),
                    'channel_loss_dB': payload.get('channel_loss_dB', 0.0),
                    'detection_rate': payload.get('detection_rate', 0.0),
                })
            except socket.timeout:
                continue
            except Exception as e:
                continue

            # Buffer for feature extraction
            st.session_state.window_buffer.append({
                'qber': row['qber'],
                'bell_S': row['bell_S'],
                'coincidence_rate': row['coincidence_rate'],
                'visibility': row['visibility'],
                'channel_loss_dB': row['channel_loss_dB'],
                'detection_rate': row['detection_rate'],
            })

            ts = row['timestamp']
            qber_val = float(row['qber'])
            bell_val = float(row['bell_S'])
            coinc_val = float(row['coincidence_rate'])

            st.session_state.qber_history.append(qber_val)
            st.session_state.bell_history.append(bell_val)
            st.session_state.ts_history.append(ts)

            # ── Inference (only when window is full) ──
            prob = 0.0
            result = {'severity': 'NORMAL', 'probability': 0.0, 'is_attack': False}
            shap_data = None

            if len(st.session_state.window_buffer) == WINDOW:
                window_df = pd.DataFrame(list(st.session_state.window_buffer))
                features = extract_features_single(window_df)
                result = predict_single(features, model_artifacts)
                prob = result['probability']
                st.session_state.max_prob = max(
                    st.session_state.max_prob, prob
                )

                if result['is_attack'] and show_shap:
                    shap_data = explain_single_alert(
                        features, explainer, model_artifacts
                    )
                    st.session_state.shap_cache = shap_data

                if result['is_attack']:
                    st.session_state.total_alerts += 1
                    skr = secure_key_rate(qber_val)
                    alert_entry = {
                        'Timestamp': str(ts),
                        'QBER (%)': f'{qber_val * 100:.2f}',
                        'Bell S': f'{bell_val:.3f}',
                        'Attack Prob': f'{prob:.3f}',
                        'Severity': result['severity'],
                        'SKR (bps)': f'{skr:.0f}',
                        'Top Feature': (
                            shap_data['feature_names'][0]
                            if shap_data else 'N/A'
                        ),
                    }
                    st.session_state.alert_log.append(alert_entry)

            st.session_state.prob_history.append(prob)

            # ── Update all panels ──
            qber_list = list(st.session_state.qber_history)
            bell_list = list(st.session_state.bell_history)
            prob_list = list(st.session_state.prob_history)
            ts_list = list(st.session_state.ts_history)

            with qber_placeholder.container():
                st.plotly_chart(
                    render_qber_panel(qber_list, prob_list, ts_list),
                    use_container_width=True, key=f'qber_{idx}',
                )

            with gauge_placeholder.container():
                st.plotly_chart(
                    render_gauge_panel(prob, threshold_override),
                    use_container_width=True, key=f'gauge_{idx}',
                )

            with bell_placeholder.container():
                st.plotly_chart(
                    render_bell_panel(bell_list, ts_list),
                    use_container_width=True, key=f'bell_{idx}',
                )

            # Metrics row
            skr_current = secure_key_rate(qber_val)
            metric_placeholders[0].metric('⏱️ Time', f't={idx}s')
            metric_placeholders[1].metric(
                '📊 QBER', f'{qber_val * 100:.2f}%'
            )
            metric_placeholders[2].metric('🔔 Bell S', f'{bell_val:.3f}')
            metric_placeholders[3].metric('🔑 SKR', f'{skr_current:.0f} bps')
            metric_placeholders[4].metric(
                '🚨 Alerts', str(st.session_state.total_alerts)
            )

            if st.session_state.shap_cache and show_shap:
                with shap_placeholder.container():
                    st.plotly_chart(
                        render_shap_panel(st.session_state.shap_cache),
                        use_container_width=True, key=f'shap_{idx}',
                    )

            if st.session_state.alert_log:
                with log_placeholder.container():
                    log_df = pd.DataFrame(
                        st.session_state.alert_log[-50:]
                    )
                    st.dataframe(
                        log_df.style.map(
                            lambda v: (
                                'background-color: rgba(255,51,68,0.2); color: #ff6b6b'
                                if v == 'CRITICAL'
                                else (
                                    'background-color: rgba(255,170,0,0.15); color: #ffaa00'
                                    if v == 'WARNING'
                                    else ''
                                )
                            ),
                            subset=['Severity'],
                        ),
                        use_container_width=True,
                        height=300,
                    )

            # Status bar
            if result['is_attack']:
                sev = result['severity']
                sev_class = 'status-critical' if sev == 'CRITICAL' else 'status-warning'
                status_bar.markdown(
                    f'<div style="padding:0.5rem; border-radius:6px; '
                    f'background:#1a1a2e; border:1px solid #2a2a4a;">'
                    f'⏱️ <strong>t = {idx}s</strong> │ '
                    f'QBER: <strong>{qber_val * 100:.2f}%</strong> │ '
                    f'Bell S: <strong>{bell_val:.3f}</strong> │ '
                    f'Coincidence: <strong>{coinc_val:.0f}/s</strong> │ '
                    f'Status: <span class="{sev_class}">🚨 {sev}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                status_bar.markdown(
                    f'<div style="padding:0.5rem; border-radius:6px; '
                    f'background:#1a1a2e; border:1px solid #2a2a4a;">'
                    f'⏱️ <strong>t = {idx}s</strong> │ '
                    f'QBER: <strong>{qber_val * 100:.2f}%</strong> │ '
                    f'Bell S: <strong>{bell_val:.3f}</strong> │ '
                    f'Coincidence: <strong>{coinc_val:.0f}/s</strong> │ '
                    f'Status: <span class="status-normal">✅ NORMAL</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.session_state.tick = idx + 1
            idx += 1

# ─── Export handler ───
st.sidebar.markdown('---')
st.sidebar.markdown('#### 📊 Data')
if 'alert_log' in st.session_state and st.session_state.alert_log:
    log_df = pd.DataFrame(st.session_state.alert_log)
    csv = log_df.to_csv(index=False)
    st.sidebar.download_button(
        label='📥 Download Alert Log (CSV)',
        data=csv,
        file_name='bbm92_alert_log.csv',
        mime='text/csv',
    )
else:
    st.sidebar.button('📥 Export Alert Log (CSV)', disabled=True, help="No alerts to export yet")
