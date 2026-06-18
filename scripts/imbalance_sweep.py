import os
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import recall_score, fbeta_score

from core.telemetry import build_telemetry_dataset
from core.channel import simulate_normal_channel
from core.attacks import attack_intercept_resend, attack_detector_blinding, attack_mitm, attack_blended_subthreshold
from ml.features import build_feature_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def build_custom_dataset(attack_ratio, seed=42):
    """Builds a custom dataset with approximately the specified attack ratio."""
    # This is a simplified proxy. To get an exact ratio, we vary the duration of blended bursts.
    rng = np.random.default_rng(seed)
    n_seconds = 86400
    base = simulate_normal_channel(n_seconds=n_seconds, seed=seed)
    base['attack_type'] = np.zeros(n_seconds, dtype=int)
    
    # We want attack_ratio * 86400 seconds of attacks.
    target_attack_sec = int(attack_ratio * n_seconds)
    
    # Fixed base attacks (IR, DB, MitM) take ~4000 seconds
    # Let's just scale the blended bursts to make up the difference
    base_attack_sec = 4000
    blended_sec_needed = max(0, target_attack_sec - base_attack_sec)
    n_bursts = int(blended_sec_needed / 30)
    
    # Inject baseline attacks
    base = attack_intercept_resend(base, duration_sec=1000, start_sec=20000, eve_fraction=0.3, rng=rng)
    base = attack_detector_blinding(base, duration_sec=1000, start_sec=40000, blinding_intensity=0.5, rng=rng)
    base = attack_mitm(base, duration_sec=1000, start_sec=60000, rng=rng)
    
    if n_bursts > 0:
        base = attack_blended_subthreshold(base, n_bursts=n_bursts, burst_duration=30, eve_fraction=0.15, rng=rng)
        
    timestamps = pd.date_range('2026-01-01 00:00:00', periods=n_seconds, freq='1s')
    df = pd.DataFrame({
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
    return df

def main():
    logger.info("Running Class Imbalance Sweep...")
    ratios = [0.05, 0.10, 0.15, 0.20, 0.30]
    
    results = []
    
    for r in ratios:
        logger.info(f"Evaluating Attack Ratio: {r*100:.1f}%")
        df = build_custom_dataset(r, seed=42)
        actual_ratio = df['label'].mean()
        
        X, y = build_feature_matrix(df, window=30)
        TRAIN_SECONDS = 64800
        X_train, y_train = X[:TRAIN_SECONDS - 30], y[:TRAIN_SECONDS - 30]
        X_test, y_test = X[TRAIN_SECONDS - 30:], y[TRAIN_SECONDS - 30:]
        
        scale_pos = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1.0
        
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, scale_pos_weight=scale_pos, random_state=42)
        xgb_model.fit(X_train, y_train)
        
        pred_xgb = (xgb_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        
        f2 = fbeta_score(y_test, pred_xgb, beta=2, zero_division=0)
        rec = recall_score(y_test, pred_xgb, zero_division=0)
        
        results.append((actual_ratio, f2, rec))
        
    print("\n" + "="*60)
    print("CLASS IMBALANCE SWEEP (Attack Ratio vs F2 / Recall)")
    print("="*60 + "\n")
    print(f"{'Target Ratio':<15} | {'Actual Ratio':<15} | {'F2 Score':<10} | {'Recall':<10}")
    print("-" * 60)
    for i, (actual_ratio, f2, rec) in enumerate(results):
        print(f"{ratios[i]:<15.2f} | {actual_ratio:<15.4f} | {f2:<10.3f} | {rec:<10.3f}")

if __name__ == "__main__":
    main()
