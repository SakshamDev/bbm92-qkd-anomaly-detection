"""
ml/explain.py — SHAP integration for BBM92 QKD anomaly detection.

Provides model explainability using SHAP (SHapley Additive exPlanations)
for the XGBoost model. Generates:
    - Per-alert top-k feature attribution (for dashboard display)
    - SHAP summary (beeswarm) plot for global feature importance
    - SHAP force plots for individual alert instances

SHAP computation is gated to alert-only ticks in the dashboard to
avoid saturating CPU at 1 Hz (§14, anti-pattern #7).

"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shap

from ml.features import FEATURE_NAMES

logger = logging.getLogger(__name__)


def build_shap_explainer(
    model_artifacts: dict[str, Any],
    X_background: np.ndarray,
) -> shap.TreeExplainer:
    """
    Builds a SHAP TreeExplainer using the XGBoost model.

    The background dataset provides the reference distribution for
    computing SHAP values (interventional or tree-path-based).

    Args:
        model_artifacts: Loaded model dict from load_model().
        X_background: Representative sample of 24-dim feature vectors
            (e.g., 500 rows from the training set). Will be scaled
            using the ensemble's scaler.

    Returns:
        shap.TreeExplainer instance ready for .shap_values() calls.
    """
    explainer = shap.TreeExplainer(model_artifacts['xgb'], data=X_background)

    logger.info(
        "SHAP TreeExplainer built with %d background samples",
        len(X_background),
    )
    return explainer


def explain_single_alert(
    feature_vector: np.ndarray,
    explainer: shap.TreeExplainer,
    model_artifacts: dict[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Computes SHAP values for a single alert instance.

    Returns the top-k features by absolute SHAP value, suitable for
    display in the dashboard's SHAP Attribution panel (§8, Panel 4).

    Performance target: < 200ms per call (§11).

    Args:
        feature_vector: Shape (24,) feature vector.
        explainer: SHAP TreeExplainer from build_shap_explainer().
        model_artifacts: Loaded model dict.
        top_k: Number of top features to return.

    Returns:
        Dict with keys:
            feature_names: List[str] — top-k feature names
            shap_values:   List[float] — corresponding SHAP values
            base_value:    float — expected model output (base rate)
    """
    t_start = time.perf_counter()

    X_sc = feature_vector.reshape(1, -1)
    sv = explainer.shap_values(X_sc)

    # Handle both 2D (binary) and 1D SHAP output formats
    if isinstance(sv, list):
        # Binary classification: sv[1] is the positive class
        shap_values = sv[1][0] if len(sv) > 1 else sv[0][0]
    elif sv.ndim == 2:
        shap_values = sv[0]
    else:
        shap_values = sv

    # Handle expected_value format
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])
    else:
        base_value = float(base_value)

    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[::-1][:top_k]

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.debug(
        "SHAP explanation computed in %.1f ms (top feature: %s = %.4f)",
        elapsed_ms,
        FEATURE_NAMES[top_indices[0]],
        float(shap_values[top_indices[0]]),
    )

    return {
        'feature_names': [FEATURE_NAMES[i] for i in top_indices],
        'shap_values': [float(shap_values[i]) for i in top_indices],
        'base_value': base_value,
        'all_shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else list(shap_values),
    }


def generate_shap_summary_plot(
    model_artifacts: dict[str, Any],
    X: np.ndarray,
    max_display: int = 24,
    output_path: str = 'data/figures/shap_summary.png',
) -> None:
    """
    Generates a SHAP beeswarm summary plot showing global feature importance.

    Each dot represents one sample. Position on x-axis shows SHAP value
    (impact on model output). Color indicates feature value (red=high, blue=low).

    Args:
        model_artifacts: Loaded model dict.
        X: Feature matrix (unscaled), shape (n, 24).
        max_display: Maximum number of features to display.
        output_path: File path for the saved figure.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Subsample for performance
    n_samples = min(2000, len(X))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X), size=n_samples, replace=False)
    X_sub = X[indices]

    explainer = shap.TreeExplainer(model_artifacts['xgb'])
    shap_values = explainer.shap_values(X_sub)

    # Handle binary classification output
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(
        shap_values, X_sub,
        feature_names=FEATURE_NAMES,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title('SHAP Feature Importance — BBM92 Attack Detection',
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close('all')

    logger.info("SHAP summary plot saved to %s (%d samples)", output_path, n_samples)



