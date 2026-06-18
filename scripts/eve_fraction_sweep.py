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
from sklearn.metrics import recall_score

from core.channel import simulate_normal_channel
from core.attacks import (
    attack_intercept_resend,
    attack_detector_blinding,
    attack_blended_subthreshold,
)
from ml.features import build_feature_matrix
from core.telemetry import build_telemetry_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Eve Fraction Sensitivity Sweep...")
    
    seed = 42
    N_SECONDS = 86400
    TRAIN_SECONDS = 64800
    
    # 1. Train the baseline model on the standard dataset
    logger.info("Training baseline XGBoost model on standard dataset (Seed 42)...")
    data_path = f'data/telemetry_{seed}.parquet'
    if not os.path.exists(data_path):
        build_telemetry_dataset(n_seconds=N_SECONDS, seed=seed, output_path=data_path)
    df_train = pd.read_parquet(data_path)
    X, y = build_feature_matrix(df_train, window=30)
    X_train, y_train = X[:TRAIN_SECONDS - 30], y[:TRAIN_SECONDS - 30]
    
    scale_pos = np.sum(y_train == 0) / np.sum(y_train == 1)
    xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, scale_pos_weight=scale_pos, random_state=seed)
    xgb_model.fit(X_train, y_train)
    
    # 2. Sweep Eve Fractions
    fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65]
    
    results = {
        'Intercept-Resend': [],
        'Detector Blinding': [],
        'Blended Sub-Threshold': []
    }
    
    rng = np.random.default_rng(seed)
    
    for f in fractions:
        logger.info(f"Evaluating Eve Fraction / Intensity = {f:.2f}")
        
        # Generate a purely holdout-focused dataset (just 21,600 seconds to save time, or 86400 for consistency)
        # We'll do 86400 for consistency to ensure moving averages stabilize
        base = simulate_normal_channel(n_seconds=N_SECONDS, seed=123) # different seed for test
        base['attack_type'] = np.zeros(N_SECONDS, dtype=int)
        
        # Inject attacks strictly in the holdout window (64800 to 86400)
        # 1. IR Attack (Type 1)
        base = attack_intercept_resend(base, duration_sec=1000, start_sec=66000, eve_fraction=f, rng=rng)
        
        # 2. Detector Blinding (Type 2)
        base = attack_detector_blinding(base, duration_sec=1000, start_sec=70000, blinding_intensity=f, rng=rng)
        
        # 3. Blended Sub-Threshold (Type 4)
        # Normally injected across the whole array, but let's restrict it by overriding
        # wait, attack_blended_subthreshold injects randomly. We can just run it and filter for holdout later.
        base = attack_blended_subthreshold(base, n_bursts=100, burst_duration=30, eve_fraction=f, rng=rng)
        
        # Assemble DataFrame
        timestamps = pd.date_range('2026-01-01 00:00:00', periods=N_SECONDS, freq='1s')
        df_test = pd.DataFrame({
            'timestamp': timestamps,
            'qber': base['qber'],
            'bell_S': base['bell_S'],
            'coincidence_rate': base['coincidence_rate'].astype(float),
            'visibility': base['visibility'],
            'channel_loss_dB': base['channel_loss_dB'],
            'detection_rate': base['detection_rate'].astype(float),
            'label': base['label'],
            'attack_type': base['attack_type'],
        })
        
        X_test_all, y_test_all = build_feature_matrix(df_test, window=30)
        attack_types = df_test['attack_type'].values[30:]
        
        # Strictly evaluate on holdout window
        X_test = X_test_all[TRAIN_SECONDS - 30:]
        y_test = y_test_all[TRAIN_SECONDS - 30:]
        attack_types_test = attack_types[TRAIN_SECONDS - 30:]
        
        # Predict
        pred_xgb = (xgb_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        
        # Calculate recall
        # Intercept-Resend
        mask_ir = (attack_types_test == 1)
        if mask_ir.sum() > 0:
            results['Intercept-Resend'].append(recall_score(y_test[mask_ir], pred_xgb[mask_ir], zero_division=0))
        else:
            results['Intercept-Resend'].append(np.nan)
            
        # Detector Blinding
        mask_db = (attack_types_test == 2)
        if mask_db.sum() > 0:
            results['Detector Blinding'].append(recall_score(y_test[mask_db], pred_xgb[mask_db], zero_division=0))
        else:
            results['Detector Blinding'].append(np.nan)
            
        # Blended
        mask_bl = (attack_types_test == 4)
        if mask_bl.sum() > 0:
            results['Blended Sub-Threshold'].append(recall_score(y_test[mask_bl], pred_xgb[mask_bl], zero_division=0))
        else:
            results['Blended Sub-Threshold'].append(np.nan)

    print("\n" + "="*70)
    print("EVE FRACTION SENSITIVITY SWEEP (Recall)")
    print("="*70 + "\n")
    
    print(f"{'Eve Fraction':<15} | {'IR Recall':<15} | {'DB Recall':<15} | {'Blended Recall':<15}")
    print("-" * 70)
    
    for i, f in enumerate(fractions):
        ir = results['Intercept-Resend'][i]
        db = results['Detector Blinding'][i]
        bl = results['Blended Sub-Threshold'][i]
        
        print(f"{f:<15.2f} | {ir:<15.3f} | {db:<15.3f} | {bl:<15.3f}")

if __name__ == "__main__":
    main()
