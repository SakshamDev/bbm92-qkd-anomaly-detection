import os
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import xgboost as xgb
import shap

from ml.features import build_feature_matrix, FEATURE_NAMES

def main():
    print("Running SHAP Analysis per Attack Type...")
    data_path = 'data/telemetry_42.parquet'
    df = pd.read_parquet(data_path)
    
    X, y = build_feature_matrix(df, window=30)
    attack_types = df['attack_type'].values[30:]
    
    TRAIN_SECONDS = 64800
    X_train, y_train = X[:TRAIN_SECONDS - 30], y[:TRAIN_SECONDS - 30]
    X_test, y_test = X[TRAIN_SECONDS - 30:], y[TRAIN_SECONDS - 30:]
    test_attack_types = attack_types[TRAIN_SECONDS - 30:]
    
    scale_pos = np.sum(y_train == 0) / np.sum(y_train == 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
        use_label_encoder=False, eval_metric='logloss',
        tree_method='hist', random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(xgb_model)
    # Use a background sample to speed up if needed, but tree SHAP is fast
    shap_values = explainer.shap_values(X_test)
    
    # Feature names
    cols = FEATURE_NAMES
            
    # Map attack types
    attack_names = {
        1: 'Intercept-Resend',
        2: 'Detector Blinding',
        3: 'MitM',
        4: 'Blended Sub-Threshold'
    }
    
    print("\n" + "="*80)
    print("TOP 5 SHAP FEATURES PER ATTACK TYPE")
    print("="*80 + "\n")
    
    for a_type, a_name in attack_names.items():
        mask = (test_attack_types == a_type)
        if mask.sum() == 0:
            continue
            
        sv_subset = shap_values[mask]
        mean_abs_shap = np.mean(np.abs(sv_subset), axis=0)
        
        top_indices = np.argsort(mean_abs_shap)[::-1][:5]
        
        print(f"--- {a_name} ---")
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. {cols[idx]:<25} (SHAP value: {mean_abs_shap[idx]:.4f})")
        print()

if __name__ == "__main__":
    main()
