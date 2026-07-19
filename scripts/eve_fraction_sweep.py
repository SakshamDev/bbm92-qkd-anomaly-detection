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
import json
from sklearn.metrics import recall_score

from core.channel import simulate_normal_channel
from core.attacks import (
    InterceptResendStrategy,
    FakedStateStrategy,
    BlendedSubthresholdStrategy,
)
from core.config import EveCapabilities
from ml.features import build_feature_matrix
from core.telemetry import build_telemetry_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Eve Fraction Sensitivity Sweep (Cross-Seed Averaged)...")
    
    seeds = [42, 123, 7, 2024, 314]
    N_SECONDS = 86400
    TRAIN_SECONDS = 64800
    fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65]
    
    # We will accumulate results across all seeds and average them at the end.
    results = {
        'Intercept-Resend': {f: [] for f in fractions},
        'Detector Blinding': {f: [] for f in fractions},
        'Blended Sub-Threshold': {f: [] for f in fractions}
    }
    
    for seed in seeds:
        logger.info(f"--- Processing Seed {seed} ---")
        
        # 1. Train the baseline model on this seed
        data_path = f'data/telemetry_{seed}.parquet'
        if not os.path.exists(data_path):
            build_telemetry_dataset(n_seconds=N_SECONDS, seed=seed, output_path=data_path)
        df_train = pd.read_parquet(data_path)
        X, y = build_feature_matrix(df_train, window=30)
        X_train, y_train = X[:TRAIN_SECONDS - 30], y[:TRAIN_SECONDS - 30]
        
        scale_pos = float(np.sum(y_train == 0)) / max(np.sum(y_train == 1), 1)
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
            use_label_encoder=False, eval_metric='logloss',
            tree_method='hist', random_state=seed, n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        
        rng = np.random.default_rng(seed)
        
        for f in fractions:
            logger.info(f"Evaluating Seed {seed} | Eve Fraction = {f:.2f}")
            
            caps = EveCapabilities()
            strategies = [
                # 1. IR Attack (Type 1)
                InterceptResendStrategy(caps, start_sec=66000, duration_sec=1000, fraction=f),
                # 2. Detector Blinding (Type 2)
                FakedStateStrategy(caps, start_sec=70000, duration_sec=1000, trigger_rate=f * 15000.0),
            ]
            
            # 3. Blended Sub-Threshold (Type 4)
            for _ in range(100):
                start = int(rng.integers(0, max(1, N_SECONDS - 30)))
                strategies.append(BlendedSubthresholdStrategy(caps, start_sec=start, duration_sec=30, fraction=f))

            base = simulate_normal_channel(n_seconds=N_SECONDS, seed=seed, attack_strategies=strategies)
            
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
            
            X_test = X_test_all[TRAIN_SECONDS - 30:]
            y_test = y_test_all[TRAIN_SECONDS - 30:]
            attack_types_test = attack_types[TRAIN_SECONDS - 30:]
            
            config_dir = f'models_seed_{seed}' if seed != 42 else 'models'
            config_path = f'{config_dir}/config.json'
            with open(config_path, 'r') as file:
                config = json.load(file)
            threshold = config.get('threshold', 0.5)

            prob = xgb_model.predict_proba(X_test)[:, 1]
            pred = (prob >= threshold).astype(int)
            
            # Calculate recall
            mask_ir = (attack_types_test == 1)
            if mask_ir.sum() > 0:
                results['Intercept-Resend'][f].append(recall_score(y_test[mask_ir], pred[mask_ir], zero_division=0))
                
            mask_db = (attack_types_test == 2)
            if mask_db.sum() > 0:
                results['Detector Blinding'][f].append(recall_score(y_test[mask_db], pred[mask_db], zero_division=0))
                
            mask_bl = (attack_types_test == 4)
            if mask_bl.sum() > 0:
                results['Blended Sub-Threshold'][f].append(recall_score(y_test[mask_bl], pred[mask_bl], zero_division=0))

    print("\n" + "="*70)
    print("EVE FRACTION SENSITIVITY SWEEP (Recall - 5 Seed Average)")
    print("="*70 + "\n")
    
    print(f"{'Eve Fraction':<15} | {'IR Recall':<15} | {'DB Recall':<15} | {'Blended Recall':<15}")
    print("-" * 70)
    
    for f in fractions:
        ir = np.nanmean(results['Intercept-Resend'][f])
        db = np.nanmean(results['Detector Blinding'][f])
        bl = np.nanmean(results['Blended Sub-Threshold'][f])
        
        print(f"{f:<15.2f} | {ir:<15.3f} | {db:<15.3f} | {bl:<15.3f}")

if __name__ == "__main__":
    main()
