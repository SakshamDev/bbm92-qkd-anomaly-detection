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
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score

from core.telemetry import build_telemetry_dataset
from ml.features import build_feature_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 7, 2024, 314]
N_SECONDS = 86400
TRAIN_SECONDS = 64800

def main():
    logger.info("Starting Window Size Ablation...")
    
    window_sizes = [15, 30, 60, 120, 300]
    results = {ws: {'recall': [], 'precision': [], 'f2': [], 'auc': []} for ws in window_sizes}
    
    for seed in SEEDS:
        logger.info(f"\n{'='*40}\nProcessing Seed {seed}\n{'='*40}")
        data_path = f'data/telemetry_{seed}.parquet'
        df = build_telemetry_dataset(n_seconds=N_SECONDS, seed=seed, output_path=data_path)
        
        for ws in window_sizes:
            logger.info(f"Extracting features for window size = {ws}s")
            X, y = build_feature_matrix(df, window=ws)
            
            # strict temporal holdout
            X_train, y_train = X[:TRAIN_SECONDS - ws], y[:TRAIN_SECONDS - ws]
            X_test, y_test = X[TRAIN_SECONDS - ws:], y[TRAIN_SECONDS - ws:]
            
            scale_pos = np.sum(y_train == 0) / np.sum(y_train == 1)
            xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, scale_pos_weight=scale_pos, random_state=seed)
            xgb_model.fit(X_train, y_train)
            
            prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
            pred_xgb = (prob_xgb >= 0.5).astype(int)
            
            results[ws]['recall'].append(recall_score(y_test, pred_xgb, zero_division=0))
            results[ws]['precision'].append(precision_score(y_test, pred_xgb, zero_division=0))
            results[ws]['f2'].append(fbeta_score(y_test, pred_xgb, beta=2, zero_division=0))
            results[ws]['auc'].append(roc_auc_score(y_test, prob_xgb))

    print("\n" + "="*80)
    print("WINDOW SIZE ABLATION REPORT (Mean ± Std over 5 seeds on HOLDOUT set)")
    print("="*80 + "\n")
    
    print(f"{'Window Size':<15} | {'Recall':<14} | {'Precision':<14} | {'F2':<14} | {'AUC':<14}")
    print("-" * 80)
    
    def fmt(vals):
        if not vals: return "N/A"
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
        
    for ws in window_sizes:
        r = results[ws]
        print(f"{ws:<11} sec | {fmt(r['recall']):<14} | {fmt(r['precision']):<14} | {fmt(r['f2']):<14} | {fmt(r.get('auc', [])):<14}")

if __name__ == "__main__":
    main()
