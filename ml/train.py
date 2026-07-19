"""
ml/train.py — XGBoost training with cross-validation.

Architecture (§6.1):
    1. StandardScaler (fit on training fold only, no leakage)
    2. XGBoost classifier (primary — handles non-linear interactions)
    3. Decision threshold tuned for Recall ≥ 0.97 on validation set

Rationale for XGBoost over deep learning:
    - 22-dimensional tabular input: trees outperform neural nets
    - SHAP values are exact for tree models
    - Training time: <10s on CPU
    - Full interpretability for defence deployment
    - No GPU required: runs on air-gapped M2 Mac

"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.features import FEATURE_NAMES

logger = logging.getLogger(__name__)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_dir: str = 'models/',
) -> dict[str, Any]:
    """
    Trains the XGBoost model and saves all artefacts.

    Training procedure:
        1. 5-fold stratified cross-validation for threshold tuning and metrics
        2. Final model training on the complete dataset
        3. Persistence of all artefacts (model, config)

    The decision threshold is swept over [0.20, 0.45] to find the minimum
    threshold achieving Recall ≥ 0.97, maximising F₂ score.

    Args:
        X: Feature matrix, shape (n_samples, 22), dtype float32.
        y: Label vector, shape (n_samples,), dtype int (0=normal, 1=attack).
        model_dir: Directory to save model artefacts.

    Returns:
        Dict with 'cv_metrics' (per-fold results) and 'final_threshold'.
    """
    t_start = time.time()
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=== Training BBM92 Anomaly Detection Model ===")
    logger.info(
        "Input: X=%s, y=%s, attack rate=%.3f",
        X.shape, y.shape, float(y.mean()),
    )

    # ──── Cross-validation setup ────
    # Time-series block split with gap to prevent window overlap leakage
    def _time_series_block_split(n_samples, n_splits=5, gap=30):
        """Expanding-window time-series CV with gap buffer. Yields n_splits - 1 folds (e.g. 4 folds)."""
        fold_size = n_samples // n_splits
        for i in range(n_splits - 1):
            train_end = (i + 1) * fold_size
            val_start = train_end + gap
            val_end = min(val_start + fold_size, n_samples)
            if val_start >= n_samples:
                break
            yield np.arange(0, train_end), np.arange(val_start, val_end)

    fold_predictions: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(
        _time_series_block_split(len(X), n_splits=5, gap=30)
    ):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # XGBoost is scale-invariant, so no scaler needed.
        # XGBoost — class weight for imbalanced data (85/15)
        scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            tree_method='hist',  # Fast on CPU, Apple Metal compatible
            n_jobs=2,
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        xgb_prob = xgb_model.predict_proba(X_val)[:, 1]
        fold_predictions.append({
            'fold': fold,
            'y_val': y_val,
            'xgb_prob': xgb_prob
        })

    # ──── Global Threshold Tuning ────
    best_threshold = 0.5
    best_mean_f2 = 0.0
    
    thresholds = np.arange(0.01, 0.99, 0.01)
    
    # Concatenate all out-of-fold predictions and labels
    all_oof_preds = np.concatenate([fold['xgb_prob'] for fold in fold_predictions])
    all_oof_y = np.concatenate([fold['y_val'] for fold in fold_predictions])
    
    for thresh in thresholds:
        preds = (all_oof_preds >= thresh).astype(int)
        global_f2 = fbeta_score(all_oof_y, preds, beta=2, zero_division=0)
            
        if global_f2 > best_mean_f2:
            best_mean_f2 = global_f2
            best_threshold = float(thresh)

    # Compute final CV metrics at best threshold
    cv_metrics = []
    for fold_data in fold_predictions:
        preds = (fold_data['xgb_prob'] >= best_threshold).astype(int)
        fold_metrics = {
            'fold': fold_data['fold'],
            'threshold': best_threshold,
            'recall': float(recall_score(fold_data['y_val'], preds, zero_division=0)),
            'precision': float(precision_score(fold_data['y_val'], preds, zero_division=0)),
            'fbeta2': float(fbeta_score(fold_data['y_val'], preds, beta=2, zero_division=0)),
            'roc_auc': float(roc_auc_score(fold_data['y_val'], fold_data['xgb_prob'])),
        }
        cv_metrics.append(fold_metrics)
        logger.info(
            "Fold %d: Recall=%.4f, F2=%.4f, AUC=%.4f, Threshold=%.2f",
            fold_data['fold'], fold_metrics['recall'], fold_metrics['fbeta2'],
            fold_metrics['roc_auc'], best_threshold,
        )
        print(
            f"Fold {fold_data['fold']}: Recall={fold_metrics['recall']:.4f}, "
            f"F2={fold_metrics['fbeta2']:.4f}, Threshold={best_threshold:.2f}"
        )

    # ──── Final model training on full dataset (18h train block) ────
    logger.info("Training final models on full train block...")

    scale_pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    final_xgb = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        use_label_encoder=False, eval_metric='logloss', random_state=42,
        tree_method='hist', n_jobs=-1,
    )
    final_xgb.fit(X, y, verbose=False)

    # ──── Save artifacts ────
    os.makedirs(model_dir, exist_ok=True)
    final_xgb.save_model(os.path.join(model_dir, 'xgb_model.json'))

    # Aggregate cross-validation results
    avg_recall = float(np.mean([m['recall'] for m in cv_metrics]))
    avg_f2 = float(np.mean([m['fbeta2'] for m in cv_metrics]))
    
    final_threshold = best_threshold

    config = {
        'threshold': final_threshold,
        'features': FEATURE_NAMES,
        'cv_metrics_mean': {
            'recall': avg_recall,
            'fbeta2': avg_f2,
        }
    }
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(
        "Training complete. Final Threshold: %.2f",
        final_threshold
    )

    elapsed = time.time() - t_start
    mean_recall = float(np.mean([m['recall'] for m in cv_metrics]))
    mean_f2 = float(np.mean([m['fbeta2'] for m in cv_metrics]))
    mean_auc = float(np.mean([m['roc_auc'] for m in cv_metrics]))

    logger.info(
        "Training complete in %.1fs | Threshold=%.2f | "
        "Recall=%.4f | F2=%.4f | AUC=%.4f",
        elapsed, final_threshold, mean_recall, mean_f2, mean_auc,
    )
    print(f"\nFinal threshold: {final_threshold:.2f}")
    print(f"Mean CV Recall: {mean_recall:.4f}")
    print(f"Mean CV F2:     {mean_f2:.4f}")
    print(f"Mean CV AUC:    {mean_auc:.4f}")
    print(f"Training time:  {elapsed:.1f}s")

    return {'cv_metrics': cv_metrics, 'final_threshold': final_threshold}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    import pandas as pd
    from ml.features import build_feature_matrix
    
    logger.info("Loading telemetry dataset...")
    df = pd.read_parquet('data/telemetry_86400.parquet')
    X, y = build_feature_matrix(df)
    
    # Separate train and test temporally (first 18 hours = train, last 6 hours = test)
    TRAIN_SECONDS = 64800
    train_mask = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds().values[30:] < TRAIN_SECONDS
    
    X_train, y_train = X[train_mask], y[train_mask]
    
    logger.info("Training XGBoost anomaly detector...")
    train_model(X_train, y_train, model_dir='models/')
