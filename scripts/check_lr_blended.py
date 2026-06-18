import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler

from core.telemetry import build_telemetry_dataset
from ml.features import build_feature_matrix

def main():
    seed = 42
    N_SECONDS = 86400
    TRAIN_SECONDS = 64800
    
    data_path = f'data/telemetry_{seed}.parquet'
    df = pd.read_parquet(data_path)
    
    X, y = build_feature_matrix(df, window=30)
    attack_types = df['attack_type'].values[30:]
    
    X_train, y_train = X[:TRAIN_SECONDS - 30], y[:TRAIN_SECONDS - 30]
    X_test, y_test = X[TRAIN_SECONDS - 30:], y[TRAIN_SECONDS - 30:]
    attack_types_test = attack_types[TRAIN_SECONDS - 30:]
    
    # 1. XGBoost
    scale_pos = np.sum(y_train == 0) / np.sum(y_train == 1)
    xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, scale_pos_weight=scale_pos, random_state=seed)
    xgb_model.fit(X_train, y_train)
    pred_xgb = (xgb_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    
    # 2. Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = LogisticRegression(class_weight='balanced', max_iter=2000)
    lr.fit(X_train_scaled, y_train)
    pred_lr = lr.predict(X_test_scaled)
    
    print("\nPER-ATTACK RECALL COMPARISON (Seed 42)")
    print(f"{'Attack Type':<25} | {'LR Recall':<12} | {'XGB Recall':<12} | Count")
    print("-" * 65)
    
    attack_names = {
        1: 'Intercept-Resend',
        2: 'Detector Blinding',
        3: 'MitM',
        4: 'Blended Sub-Threshold'
    }
    
    for a_type, name in attack_names.items():
        mask = (attack_types_test == a_type)
        if mask.sum() > 0:
            a_y = y_test[mask]
            xgb_r = recall_score(a_y, pred_xgb[mask], zero_division=0)
            lr_r = recall_score(a_y, pred_lr[mask], zero_division=0)
            print(f"{name:<25} | {lr_r:<12.3f} | {xgb_r:<12.3f} | {mask.sum()}")

if __name__ == "__main__":
    main()
