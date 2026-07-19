import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import wilcoxon

from ml.features import build_feature_matrix

def main():
    print("Running Wilcoxon Signed-Rank Test (XGBoost vs LightGBM surrogate)...")
    data_path = 'data/telemetry_42.parquet'
    df = pd.read_parquet(data_path)
    
    X, y = build_feature_matrix(df, window=30)
    TRAIN_SECONDS = 64800
    X_train, y_train = X[:TRAIN_SECONDS - 30], y[:TRAIN_SECONDS - 30]
    X_test, y_test = X[TRAIN_SECONDS - 30:], y[TRAIN_SECONDS - 30:]
    
    scale_pos = np.sum(y_train == 0) / np.sum(y_train == 1)
    
    # Train XGB
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
        use_label_encoder=False, eval_metric='logloss',
        tree_method='hist', random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    # Train LightGBM surrogate
    lgb_model = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.05, random_state=42)
    # class weight for HGB
    sample_weight = np.where(y_train == 1, scale_pos, 1.0)
    lgb_model.fit(X_train, y_train, sample_weight=sample_weight)
    p_lgb = lgb_model.predict_proba(X_test)[:, 1]
    
    # Absolute errors
    err_xgb = np.abs(y_test - p_xgb)
    err_lgb = np.abs(y_test - p_lgb)
    
    stat, p_val = wilcoxon(err_xgb, err_lgb)
    
    print("="*60)
    print("WILCOXON SIGNED-RANK TEST (Absolute Errors)")
    print("="*60)
    print(f"XGBoost Mean Absolute Error:  {np.mean(err_xgb):.4f}")
    print(f"LightGBM Mean Absolute Error: {np.mean(err_lgb):.4f}")
    print(f"Wilcoxon Statistic:           {stat}")
    print(f"p-value:                      {p_val:.4e}")
    
    if p_val < 0.05:
        print("\nConclusion: The difference in absolute predicted probabilities is STATISTICALLY SIGNIFICANT (p < 0.05).")
    else:
        print("\nConclusion: The difference is NOT statistically significant (p >= 0.05).")

if __name__ == "__main__":
    main()
