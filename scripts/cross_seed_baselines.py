"""
Cross-seed baseline comparison for IEEE INDISCON 2026 Table I.

Runs 4 models (Threshold, LR, RF, XGBoost) across 5 seeds and reports
mean ± std for Recall, Precision, and F2.
"""
import os
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, fbeta_score

from core.telemetry import build_telemetry_dataset
from ml.features import build_feature_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 7, 2024, 314]
TRAIN_SECONDS = 64800
QBER_THRESHOLD = 0.11

def evaluate_seed(seed):
    """Run all 4 models on a single seed, return dict of metrics."""
    logger.info(f"{'='*40}\nProcessing Seed {seed}\n{'='*40}")

    data_path = f'data/telemetry_{seed}.parquet'
    if not os.path.exists(data_path):
        build_telemetry_dataset(n_seconds=86400, seed=seed, output_path=data_path)
    df = pd.read_parquet(data_path)

    X, y = build_feature_matrix(df, window=30)
    X_train, y_train = X[:TRAIN_SECONDS - 30], y[:TRAIN_SECONDS - 30]
    X_test, y_test = X[TRAIN_SECONDS - 30:], y[TRAIN_SECONDS - 30:]

    scale_pos = np.sum(y_train == 0) / np.sum(y_train == 1)
    results = {}

    # 1. QBER Threshold
    qber_test = df['qber'].values[TRAIN_SECONDS:]
    pred_thr = (qber_test > QBER_THRESHOLD).astype(int)
    # Align lengths
    min_len = min(len(pred_thr), len(y_test))
    pred_thr = pred_thr[:min_len]
    y_test_thr = y_test[:min_len]
    results['Threshold'] = {
        'recall': recall_score(y_test_thr, pred_thr),
        'precision': precision_score(y_test_thr, pred_thr, zero_division=0),
        'f2': fbeta_score(y_test_thr, pred_thr, beta=2, zero_division=0),
    }

    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Logistic Regression
    lr = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=seed)
    lr.fit(X_train_scaled, y_train)
    pred_lr = lr.predict(X_test_scaled)
    results['Logistic Regression'] = {
        'recall': recall_score(y_test, pred_lr),
        'precision': precision_score(y_test, pred_lr, zero_division=0),
        'f2': fbeta_score(y_test, pred_lr, beta=2, zero_division=0),
    }

    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=seed, n_jobs=2)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results['Random Forest'] = {
        'recall': recall_score(y_test, pred_rf),
        'precision': precision_score(y_test, pred_rf, zero_division=0),
        'f2': fbeta_score(y_test, pred_rf, beta=2, zero_division=0),
    }

    # 4. XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
        use_label_encoder=False, eval_metric='logloss',
        tree_method='exact', random_state=seed, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    config_dir = f'models_seed_{seed}' if seed != 42 else 'models'
    config_path = f'{config_dir}/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    threshold = config.get('threshold', 0.5)
    pred_xgb = (xgb_model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    results['XGBoost'] = {
        'recall': recall_score(y_test, pred_xgb),
        'precision': precision_score(y_test, pred_xgb, zero_division=0),
        'f2': fbeta_score(y_test, pred_xgb, beta=2, zero_division=0),
    }

    return results

def main():
    all_results = {model: {'recall': [], 'precision': [], 'f2': []}
                   for model in ['Threshold', 'Logistic Regression', 'Random Forest', 'XGBoost']}

    for seed in SEEDS:
        seed_results = evaluate_seed(seed)
        for model, metrics in seed_results.items():
            for metric, value in metrics.items():
                all_results[model][metric].append(value)

    print("\n" + "=" * 85)
    print("TABLE I: CROSS-SEED BASELINE COMPARISON (Mean ± Std over 5 seeds)")
    print("=" * 85 + "\n")
    print(f"{'Model':<25} | {'Recall':<18} | {'Precision':<18} | {'F2 Score':<18}")
    print("-" * 85)

    for model in ['Threshold', 'Logistic Regression', 'Random Forest', 'XGBoost']:
        m = all_results[model]
        rec = f"{np.mean(m['recall']):.3f} ± {np.std(m['recall']):.3f}"
        prec = f"{np.mean(m['precision']):.3f} ± {np.std(m['precision']):.3f}"
        f2 = f"{np.mean(m['f2']):.3f} ± {np.std(m['f2']):.3f}"
        print(f"{model:<25} | {rec:<18} | {prec:<18} | {f2:<18}")

    print()

if __name__ == "__main__":
    main()
