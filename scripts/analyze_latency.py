import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import xgboost as xgb
import json

from core.telemetry import build_telemetry_dataset
from ml.features import build_feature_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Detection Latency Analysis...")
    
    seed = 42
    N_SECONDS = 86400
    TRAIN_SECONDS = 64800
    
    data_path = f'data/telemetry_{seed}.parquet'
    if not os.path.exists(data_path):
        build_telemetry_dataset(n_seconds=N_SECONDS, seed=seed, output_path=data_path)
    df = pd.read_parquet(data_path)
    
    X, y = build_feature_matrix(df, window=30)
    attack_types = df['attack_type'].values[30:]
    
    X_train, y_train = X[:TRAIN_SECONDS - 30], y[:TRAIN_SECONDS - 30]
    X_test, y_test = X[TRAIN_SECONDS - 30:], y[TRAIN_SECONDS - 30:]
    attack_types_test = attack_types[TRAIN_SECONDS - 30:]
    
    scale_pos = np.sum(y_train == 0) / np.sum(y_train == 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
        use_label_encoder=False, eval_metric='logloss',
        tree_method='hist', random_state=seed, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # Analyze latency vs detection rate
    config_path = f'models_seed_42/config.json'
    with open(config_path, 'r') as file:
        config = json.load(file)
    threshold = config.get('threshold', 0.5)
    pred_xgb = (xgb_model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    
    # Identify contiguous attack blocks in y_test
    # A block starts when y_test changes from 0 to 1
    # Note: we need to handle the very first element if it's 1
    diffs = np.diff(np.concatenate(([0], y_test, [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    latencies = {
        'Intercept-Resend': [],
        'Detector Blinding': [],
        'MitM': [],
        'Blended Sub-Threshold': []
    }
    
    attack_names = {
        1: 'Intercept-Resend',
        2: 'Detector Blinding',
        3: 'MitM',
        4: 'Blended Sub-Threshold'
    }
    
    for start_idx, end_idx in zip(starts, ends):
        block_preds = pred_xgb[start_idx:end_idx]
        block_attack_types = attack_types_test[start_idx:end_idx]
        
        # Get the primary attack type for this block (mode)
        a_type = pd.Series(block_attack_types).mode()[0]
        if a_type == 0:
            continue
            
        a_name = attack_names.get(a_type)
        
        # Find first alert
        alerts = np.where(block_preds == 1)[0]
        if len(alerts) > 0:
            latency = alerts[0] # index offset from start is exactly the latency in seconds
            latencies[a_name].append(latency)
        else:
            # Completely missed attack
            pass

    print("\n" + "="*80)
    print("DETECTION LATENCY REPORT (Holdout Set, Seed 42)")
    print("="*80 + "\n")
    
    print(f"{'Attack Type':<25} | {'Mean Latency':<15} | {'95th Pct':<15} | {'Detected / Total'}")
    print("-" * 80)
    
    for name in ['Intercept-Resend', 'Detector Blinding', 'MitM', 'Blended Sub-Threshold']:
        l_list = latencies[name]
        
        # Count total attacks of this type
        total = 0
        for start_idx, end_idx in zip(starts, ends):
            block_attack_types = attack_types_test[start_idx:end_idx]
            a_type = pd.Series(block_attack_types).mode()[0]
            if attack_names.get(a_type) == name:
                total += 1
                
        if len(l_list) > 0:
            mean_l = np.mean(l_list)
            pct95_l = np.percentile(l_list, 95)
            print(f"{name:<25} | {mean_l:<11.2f} sec | {pct95_l:<11.2f} sec | {len(l_list)} / {total}")
        else:
            print(f"{name:<25} | {'N/A':<15} | {'N/A':<15} | {0} / {total}")

if __name__ == "__main__":
    main()
