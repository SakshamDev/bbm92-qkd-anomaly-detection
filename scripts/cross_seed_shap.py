"""
Cross-seed SHAP analysis to verify stable feature importance across seeds.
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
import shap

from ml.features import build_feature_matrix, FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 2023, 2024, 314]
TRAIN_SECONDS = 64800

def main():
    print("\nRunning Cross-Seed SHAP Analysis...")
    
    attack_names = {
        1: 'Intercept-Resend',
        2: 'Detector Blinding',
        3: 'MitM',
        4: 'Blended Sub-Threshold'
    }
    
    # Store sum of mean absolute SHAP values across seeds for each attack type
    # shape: (4, num_features)
    sum_mean_abs_shap = {k: np.zeros(len(FEATURE_NAMES)) for k in attack_names.keys()}
    counts = {k: 0 for k in attack_names.keys()}
    
    for seed in SEEDS:
        logger.info(f"Computing SHAP for seed {seed}...")
        data_path = f'data/telemetry_{seed}.parquet'
        df = pd.read_parquet(data_path)
        
        X, y = build_feature_matrix(df, window=30)
        attack_types = df['attack_type'].values[30:]
        
        X_train, y_train = X[:TRAIN_SECONDS - 30], y[:TRAIN_SECONDS - 30]
        X_test = X[TRAIN_SECONDS - 30:]
        test_attack_types = attack_types[TRAIN_SECONDS - 30:]
        
        scale_pos = np.sum(y_train == 0) / np.sum(y_train == 1)
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
            use_label_encoder=False, eval_metric='logloss',
            tree_method='hist', random_state=seed, n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        
        for a_type in attack_names.keys():
            mask = (test_attack_types == a_type)
            if mask.sum() > 0:
                sv_subset = shap_values[mask]
                mean_abs_shap = np.mean(np.abs(sv_subset), axis=0)
                sum_mean_abs_shap[a_type] += mean_abs_shap
                counts[a_type] += 1
                
    print("\n" + "="*80)
    print("TOP 5 SHAP FEATURES PER ATTACK TYPE (AVERAGED ACROSS 5 SEEDS)")
    print("="*80 + "\n")
    
    for a_type, a_name in attack_names.items():
        if counts[a_type] > 0:
            avg_shap = sum_mean_abs_shap[a_type] / counts[a_type]
            top_indices = np.argsort(avg_shap)[::-1][:5]
            
            print(f"--- {a_name} ---")
            for i, idx in enumerate(top_indices):
                print(f"  {i+1}. {FEATURE_NAMES[idx]:<25} (Avg SHAP: {avg_shap[idx]:.4f})")
            print()

if __name__ == "__main__":
    main()
