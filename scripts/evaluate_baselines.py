import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, IsolationForest, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar

from core.telemetry import build_telemetry_dataset
from ml.features import build_feature_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 7, 2024, 314]
N_SECONDS = 86400
TRAIN_SECONDS = 64800

def compute_mcnemar(y_true, y_pred1, y_pred2):
    correct1 = (y_true == y_pred1)
    correct2 = (y_true == y_pred2)
    
    both_correct = np.sum(correct1 & correct2)
    only_1_correct = np.sum(correct1 & ~correct2)
    only_2_correct = np.sum(~correct1 & correct2)
    both_incorrect = np.sum(~correct1 & ~correct2)
    
    table = [[both_correct, only_1_correct],
             [only_2_correct, both_incorrect]]
    
    result = mcnemar(table, exact=False, correction=True)
    return result.pvalue

def main():
    logger.info("Starting ML Baseline Evaluation...")
    
    models_to_evaluate = ['threshold', 'LogisticRegression', 'IsolationForest', 'RandomForest', 'LightGBM', 'XGBoost']
    
    results = {m: {'recall': [], 'precision': [], 'f2': [], 'auc': []} for m in models_to_evaluate}
    
    # Store pooled predictions for McNemar's test
    pooled_y_true = []
    pooled_preds = {m: [] for m in models_to_evaluate}
    
    for seed in SEEDS:
        logger.info(f"\n{'='*40}\nProcessing Seed {seed}\n{'='*40}")
        
        data_path = f'data/telemetry_{seed}.parquet'
        df = build_telemetry_dataset(n_seconds=N_SECONDS, seed=seed, output_path=data_path)
        
        X, y = build_feature_matrix(df, window=30)
        qber = df['qber'].values[30:]
        
        X_train, y_train = X[:TRAIN_SECONDS - 30], y[:TRAIN_SECONDS - 30]
        X_test, y_test = X[TRAIN_SECONDS - 30:], y[TRAIN_SECONDS - 30:]
        qber_test = qber[TRAIN_SECONDS - 30:]
        
        pooled_y_true.append(y_test)
        
        # 1. Threshold
        pred_thresh = (qber_test > 0.11).astype(int)
        results['threshold']['recall'].append(recall_score(y_test, pred_thresh, zero_division=0))
        results['threshold']['precision'].append(precision_score(y_test, pred_thresh, zero_division=0))
        results['threshold']['f2'].append(fbeta_score(y_test, pred_thresh, beta=2, zero_division=0))
        pooled_preds['threshold'].append(pred_thresh)
        
        # 2. Logistic Regression
        # Normalize data for LR
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr = LogisticRegression(class_weight='balanced', max_iter=2000)
        lr.fit(X_train_scaled, y_train)
        pred_lr = lr.predict(X_test_scaled)
        prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
        results['LogisticRegression']['recall'].append(recall_score(y_test, pred_lr, zero_division=0))
        results['LogisticRegression']['precision'].append(precision_score(y_test, pred_lr, zero_division=0))
        results['LogisticRegression']['f2'].append(fbeta_score(y_test, pred_lr, beta=2, zero_division=0))
        results['LogisticRegression']['auc'].append(roc_auc_score(y_test, prob_lr))
        pooled_preds['LogisticRegression'].append(pred_lr)
        
        # 3. Isolation Forest
        contam = np.sum(y_train) / len(y_train)
        if contam == 0: contam = 0.137
        iso = IsolationForest(contamination=contam, random_state=seed)
        iso.fit(X_train)
        pred_iso_raw = iso.predict(X_test)
        pred_iso = (pred_iso_raw == -1).astype(int)
        prob_iso = -iso.score_samples(X_test)
        results['IsolationForest']['recall'].append(recall_score(y_test, pred_iso, zero_division=0))
        results['IsolationForest']['precision'].append(precision_score(y_test, pred_iso, zero_division=0))
        results['IsolationForest']['f2'].append(fbeta_score(y_test, pred_iso, beta=2, zero_division=0))
        results['IsolationForest']['auc'].append(roc_auc_score(y_test, prob_iso))
        pooled_preds['IsolationForest'].append(pred_iso)
        
        # 4. Random Forest
        rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)
        prob_rf = rf.predict_proba(X_test)[:, 1]
        results['RandomForest']['recall'].append(recall_score(y_test, pred_rf, zero_division=0))
        results['RandomForest']['precision'].append(precision_score(y_test, pred_rf, zero_division=0))
        results['RandomForest']['f2'].append(fbeta_score(y_test, pred_rf, beta=2, zero_division=0))
        results['RandomForest']['auc'].append(roc_auc_score(y_test, prob_rf))
        pooled_preds['RandomForest'].append(pred_rf)
        
        # 5. HistGradientBoosting (LightGBM Surrogate)
        # HistGradientBoostingClassifier doesn't support scale_pos_weight directly, we can use class_weight='balanced' if we pass sample_weights to fit.
        # But we can just use it without if it's not supported, or compute sample_weights.
        scale_pos = np.sum(y_train == 0) / np.sum(y_train == 1)
        sample_weights = np.where(y_train == 1, scale_pos, 1.0)
        hgb = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.05, random_state=seed)
        hgb.fit(X_train, y_train, sample_weight=sample_weights)
        pred_hgb = hgb.predict(X_test)
        prob_hgb = hgb.predict_proba(X_test)[:, 1]
        results['LightGBM']['recall'].append(recall_score(y_test, pred_hgb, zero_division=0))
        results['LightGBM']['precision'].append(precision_score(y_test, pred_hgb, zero_division=0))
        results['LightGBM']['f2'].append(fbeta_score(y_test, pred_hgb, beta=2, zero_division=0))
        results['LightGBM']['auc'].append(roc_auc_score(y_test, prob_hgb))
        pooled_preds['LightGBM'].append(pred_hgb)
        
        # 6. XGBoost
        xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, scale_pos_weight=scale_pos, random_state=seed)
        xgb_model.fit(X_train, y_train)
        prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        pred_xgb = (prob_xgb >= 0.5).astype(int)
        results['XGBoost']['recall'].append(recall_score(y_test, pred_xgb, zero_division=0))
        results['XGBoost']['precision'].append(precision_score(y_test, pred_xgb, zero_division=0))
        results['XGBoost']['f2'].append(fbeta_score(y_test, pred_xgb, beta=2, zero_division=0))
        results['XGBoost']['auc'].append(roc_auc_score(y_test, prob_xgb))
        pooled_preds['XGBoost'].append(pred_xgb)

    # Combine pooled results
    y_true_all = np.concatenate(pooled_y_true)
    y_pred_all = {m: np.concatenate(pooled_preds[m]) for m in models_to_evaluate}
    
    # Compute McNemar's p-values relative to XGBoost
    p_values = {}
    for m in models_to_evaluate:
        if m == 'XGBoost':
            p_values[m] = 1.0
        else:
            p_values[m] = compute_mcnemar(y_true_all, y_pred_all['XGBoost'], y_pred_all[m])

    print("\n" + "="*95)
    print("FINAL ML BASELINE COMPARISON REPORT (Mean ± Std over 5 seeds on HOLDOUT set)")
    print("="*95 + "\n")
    
    print(f"{'Method':<22} | {'Recall':<14} | {'Precision':<14} | {'F2':<14} | {'AUC':<14} | {'McNemar (vs XGB)'}")
    print("-" * 105)
    
    def fmt(vals):
        if not vals: return "N/A"
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
        
    for model in models_to_evaluate:
        r = results[model]
        name = "QBER > 11%" if model == 'threshold' else model
        pval = f"{p_values[model]:.1e}" if p_values[model] < 0.05 else f"{p_values[model]:.3f}"
        if model == 'XGBoost': pval = "---"
        print(f"{name:<22} | {fmt(r['recall']):<14} | {fmt(r['precision']):<14} | {fmt(r['f2']):<14} | {fmt(r.get('auc', [])):<14} | {pval}")

if __name__ == "__main__":
    main()
