"""
ml/features.py — 22-dimensional feature engineering pipeline for BBM92 QKD.

Computes rolling-window features from raw telemetry at 1 Hz. The 22 features
capture temporal structure invisible to static QBER thresholds, which is the
core advantage over naive threshold-based anomaly detection.

# Feature groups (22 total):
#     QBER features (6):       Statistical moments and autocorrelation
#     Bell S features (5):     Entanglement health and QBER correlation
#     Coincidence features (5): Rate analysis, PNS signatures
#     Temporal structure (2):  Drift vs step-change discrimination
#     Cross-channel (4):       Multi-observable anomaly decoupling
#
# Window size: 30 seconds (tunable hyperparameter).
#
"""

from __future__ import annotations

import logging


import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WINDOW: int = 30  # seconds — tunable hyperparameter

# Canonical feature names (must match training and inference)
FEATURE_NAMES: list[str] = [
    # -- QBER features (6) --
    'qber_mean',            # Window mean
    'qber_std',             # Window standard deviation
    'qber_delta',           # Current - 30s-ago (rate of change)
    'qber_skewness',        # Third moment: attack patterns are right-skewed
    'qber_autocorr_lag1',   # AR(1) coefficient: low for weather, high for attacks
    'qber_autocorr_lag5',   # AR(5) coefficient: longer memory

    # -- Bell S features (5) --
    'bell_S_mean',          # Window mean CHSH S
    'bell_S_std',           # S volatility
    'bell_S_delta',         # Rate of entanglement degradation
    'bell_S_below_2414',    # Fraction of window with S < 2√2 × 0.85
    'bell_S_pearson_qber',  # Pearson correlation(S, QBER) within window

    # -- Coincidence / Detection features (5) --
    'coincidence_mean',     # Mean coincidence rate
    'coincidence_drop_pct', # % drop from window maximum (PNS signature)
    'coincidence_cv',       # Coefficient of variation (burstiness)
    'coincidence_qber_corr',  # Cross-correlation with QBER
    'detection_to_coinc_ratio',  # Ratio: anomalous if Eve splits pairs

    # -- Temporal structure (2) --
    'bell_S_rolling_range',   # Max-min range of S in window (stability)
    'temporal_asymmetry',     # Rising vs falling edge asymmetry in QBER

    # -- Cross-channel (4) --
    'channel_loss_mean',    # Mean FSO path loss
    'visibility_mean',      # Mean Hong-Ou-Mandel visibility
    'loss_qber_decoupling', # High QBER but low channel loss = anomaly
    'S_coincidence_product', # S × coincidence_rate: joint health metric
]

assert len(FEATURE_NAMES) == 22, f"Expected 22 features, got {len(FEATURE_NAMES)}"


def _autocorr(x: np.ndarray, lag: int) -> float:
    """
    Vectorised autocorrelation at a given lag.

    Computes the normalised autocovariance:
        ρ(lag) = Cov(X_t, X_{t+lag}) / Var(X)

    Returns 0.0 if the series is too short or has zero variance
    (constant signal → undefined autocorrelation).

    Args:
        x: 1-D time series array.
        lag: Lag value (positive integer).

    Returns:
        Autocorrelation coefficient in [-1, 1], or 0.0 if underpowered.
    """
    if len(x) <= lag + 1:
        return 0.0
    n = len(x)
    mean = np.mean(x)
    var = np.var(x)
    if var < 1e-10:
        return 0.0
    x_centered = x - mean
    return float(np.dot(x_centered[:n - lag], x_centered[lag:]) / ((n - lag) * var))


def extract_features_single(window: pd.DataFrame) -> np.ndarray:
    """
    Extracts the 22-dimensional feature vector from a 30-row window.

    Called at 1 Hz during live inference. Each feature is documented
    in the FEATURE_NAMES list above.

    Args:
        window: DataFrame with ≤30 rows and columns:
            qber, bell_S, coincidence_rate, visibility,
            channel_loss_dB, detection_rate

    Returns:
        numpy array, shape (22,), dtype float32.

    Raises:
        AssertionError: If output shape is not (22,).
    """
    q = window['qber'].values.astype(np.float64)
    S = window['bell_S'].values.astype(np.float64)
    c = window['coincidence_rate'].values.astype(np.float64)
    v = window['visibility'].values.astype(np.float64)
    loss = window['channel_loss_dB'].values.astype(np.float64)
    det = window['detection_rate'].values.astype(np.float64)

    eps = 1e-9

    # ──── QBER features (6) ────
    qber_mean = float(np.mean(q))
    qber_std = float(np.std(q))
    qber_delta = float(q[-1] - q[0]) if len(q) >= 2 else 0.0
    # Skewness: E[(X-μ)³] / σ³
    qber_skewness = float(
        np.mean((q - qber_mean) ** 3) / (qber_std ** 3 + eps)
    )
    qber_autocorr_lag1 = _autocorr(q, 1)
    qber_autocorr_lag5 = _autocorr(q, 5)

    # ──── Bell S features (5) ────
    bell_S_mean = float(np.mean(S))
    bell_S_std = float(np.std(S))
    bell_S_delta = float(S[-1] - S[0]) if len(S) >= 2 else 0.0
    bell_S_below_2414 = float(np.mean(S < 2.414))  # 2√2 × 0.854 ≈ 2.414
    # Pearson(S, QBER): negative in healthy channel, may flip under attack
    if qber_std > eps and np.std(S) > eps:
        bell_S_pearson_qber = float(np.corrcoef(S, q)[0, 1])
    else:
        bell_S_pearson_qber = 0.0

    # ──── Coincidence / Detection features (5) ────
    coincidence_mean = float(np.mean(c))
    c_max = float(np.max(c)) if np.max(c) > 0 else 1.0
    coincidence_drop_pct = float((c_max - coincidence_mean) / (c_max + eps))
    coincidence_cv = float(np.std(c) / (coincidence_mean + eps))
    if np.std(c) > eps and qber_std > eps:
        coincidence_qber_corr = float(np.corrcoef(c, q)[0, 1])
    else:
        coincidence_qber_corr = 0.0
    det_mean = float(np.mean(det))
    detection_to_coinc_ratio = float(
        det_mean / (2.0 * coincidence_mean + eps)
    )

    # ──── Temporal structure (2) ────
    bell_S_rolling_range = float(np.max(S) - np.min(S))
    # Asymmetry: compare mean of second half vs first half
    mid = len(q) // 2
    temporal_asymmetry = float(np.mean(q[mid:]) - np.mean(q[:mid]))

    # ──── Cross-channel (4) ────
    channel_loss_mean = float(np.mean(loss))
    visibility_mean = float(np.mean(v))
    # Key anomaly: QBER high but channel loss normal → cannot explain by FSO → attack
    loss_linear = 10.0 ** (-channel_loss_mean / 10.0)
    expected_qber_from_loss = 0.02 + 0.05 * (1.0 - loss_linear)
    loss_qber_decoupling = float(qber_mean - expected_qber_from_loss)
    S_coincidence_product = float(
        bell_S_mean * coincidence_mean / 10000.0
    )  # normalised

    features = np.array([
        qber_mean, qber_std, qber_delta, qber_skewness,
        qber_autocorr_lag1, qber_autocorr_lag5,
        bell_S_mean, bell_S_std, bell_S_delta,
        bell_S_below_2414, bell_S_pearson_qber,
        coincidence_mean, coincidence_drop_pct, coincidence_cv,
        coincidence_qber_corr, detection_to_coinc_ratio,
        bell_S_rolling_range, temporal_asymmetry,
        channel_loss_mean, visibility_mean, loss_qber_decoupling,
        S_coincidence_product,
    ], dtype=np.float32)

    assert features.shape == (22,), f"Feature shape mismatch: {features.shape}"
    return features


def build_feature_matrix(
    df: pd.DataFrame,
    window: int = WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies rolling feature extraction to the full 86,400-row DataFrame.

    For each timestep t ≥ window, extracts the 22-dimensional feature vector
    from the preceding `window` seconds. The label is taken from the last
    timestep in the window (i.e., the current second).

    Complexity: O(n × window) — acceptable for 86,400 × 30.

    Args:
        df: Full telemetry DataFrame with columns:
            qber, bell_S, coincidence_rate, visibility,
            channel_loss_dB, detection_rate, label
        window: Rolling window size in seconds (default: 30).

    Returns:
        X: Feature matrix, shape (n - window, 22), dtype float32.
        y: Label vector, shape (n - window,), dtype int.
    """
    n = len(df)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    feature_cols = [
        'qber', 'bell_S', 'coincidence_rate',
        'visibility', 'channel_loss_dB', 'detection_rate',
    ]

    logger.info(
        "Building feature matrix: %d windows of %d seconds each",
        n - window, window,
    )

    for i in range(window, n):
        # E-6: Mask out windows containing sentinels
        if 'is_low_count' in df.columns and df.iloc[i - window:i]['is_low_count'].any():
            continue
            
        window_df = df.iloc[i - window:i][feature_cols]
        X_list.append(extract_features_single(window_df))
        y_list.append(int(df.iloc[i]['label']))

        # Progress logging every 10,000 steps
        if (i - window) % 10000 == 0 and (i - window) > 0:
            logger.info(
                "  Feature extraction: %d/%d (%.1f%%)",
                i - window, n - window,
                (i - window) / (n - window) * 100,
            )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=int)

    logger.info(
        "Feature matrix built: X=%s, y=%s, attack rate=%.3f",
        X.shape, y.shape, float(y.mean()),
    )

    return X, y

def build_raw_feature_matrix(
    df: pd.DataFrame,
    window: int = WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts raw instantaneous telemetry (QBER, Bell S, etc.) without rolling windows.
    Skips the first `window` seconds so the output aligns exactly with the temporal matrix.

    Args:
        df: Full telemetry DataFrame.
        window: Number of seconds to skip at the beginning to match `build_feature_matrix`.

    Returns:
        X: Feature matrix, shape (n - window, 6), dtype float32.
        y: Label vector, shape (n - window,), dtype int.
    """
    feature_cols = [
        'qber', 'bell_S', 'coincidence_rate',
        'visibility', 'channel_loss_dB', 'detection_rate',
    ]
    
    logger.info("Building RAW feature matrix (instantaneous telemetry only)")
    
    # Skip the first `window` rows to exactly match the temporal dataset size
    df_clipped = df.iloc[window:].copy()
    
    X = df_clipped[feature_cols].values.astype(np.float32)
    y = df_clipped['label'].values.astype(int)
    
    logger.info(
        "Raw Feature matrix built: X=%s, y=%s, attack rate=%.3f",
        X.shape, y.shape, float(y.mean()),
    )
    
    return X, y
