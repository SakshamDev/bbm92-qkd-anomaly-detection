"""
ml/inference.py — Standalone inference module for BBM92 QKD anomaly detection.

Provides functions to load models and run predictions on incoming telemetry.
Designed for clean deployment on the Streamlit dashboard or edge hardware.
"""

import json
import logging
import numpy as np
import xgboost as xgb
from typing import Any

logger = logging.getLogger(__name__)

def load_model(model_dir: str = 'models/') -> dict[str, Any]:
    """
    Loads all saved artefacts for inference.

    Returns:
        Dict with keys: 'xgb', 'config'.
    """
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(f'{model_dir}/xgb_model.json')
    with open(f'{model_dir}/config.json') as f:
        config = json.load(f)

    logger.info(
        "Model loaded from %s (threshold=%.2f)",
        model_dir, config.get('threshold', 0.5),
    )
    return {
        'xgb': xgb_model,
        'config': config,
    }


def predict_single(
    feature_vector: np.ndarray,
    model_artifacts: dict[str, Any],
) -> dict[str, Any]:
    """
    Runs inference on a single 24-dimensional feature vector.

    Severity classification:
        - NORMAL:   probability < threshold
        - WARNING:  threshold ≤ probability < 0.65
        - CRITICAL: probability ≥ 0.65

    Args:
        feature_vector: Shape (24,) feature vector from extract_features_single().
        model_artifacts: Loaded model dict from load_model().

    Returns:
        Dict with keys: probability, is_attack, severity.
    """
    X = feature_vector.reshape(1, -1)

    prob = float(model_artifacts['xgb'].predict_proba(X)[0, 1])
    threshold = model_artifacts['config'].get('threshold', 0.5)
    is_attack = prob >= threshold

    if prob >= 0.65:
        severity = 'CRITICAL'
    elif is_attack:
        severity = 'WARNING'
    else:
        severity = 'NORMAL'

    return {
        'probability': prob,
        'is_attack': is_attack,
        'severity': severity,
    }
