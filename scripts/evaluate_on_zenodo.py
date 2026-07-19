import os
import json
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.features import build_feature_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    # 1. Load real-world baseline
    data_path = os.path.join(PROJECT_ROOT, 'data', 'zenodo_ent_telemetry.parquet')
    df = pd.read_parquet(data_path)
    
    n = len(df)
    
    # 2. Synthesize missing base telemetry based on physics
    qber = df['qber'].values
    channel_loss_dB = -df['link_loss'].values  
    
    visibility = 1.0 - 2.0 * qber
    bell_S = 2.828 * visibility
    
    # Coincidence rate drops exponentially with channel loss
    coincidence_rate = (100000.0 * (10.0 ** (-channel_loss_dB / 10.0))).astype(int)
    detection_rate = coincidence_rate * 2.5 + 1000.0
    
    base_telemetry = {
        'qber': qber,
        'bell_S': bell_S,
        'coincidence_rate': coincidence_rate,
        'visibility': visibility,
        'channel_loss_dB': channel_loss_dB,
        'detection_rate': detection_rate,
        'label': np.zeros(n, dtype=int),
        'attack_type': np.zeros(n, dtype=int)
    }
    
    # 3. Use the synthesised telemetry directly (normal data only)
    df_eval = pd.DataFrame(base_telemetry)
    
    # 4. Extract features
    X, y = build_feature_matrix(df_eval, window=30)
    
    # 5. Load model and evaluate
    model_path = os.path.join(PROJECT_ROOT, 'models', 'xgb_model.json')
    config_path = os.path.join(PROJECT_ROOT, 'models', 'config.json')
    
    if not os.path.exists(model_path):
        logging.error("XGBoost model not found. Please train it first.")
        return
        
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_path)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    xgb_prob = xgb_model.predict_proba(X)[:, 1]
    y_pred = (xgb_prob >= config.get('threshold', 0.5)).astype(int)
    
    print("\n" + "="*60)
    print("EVALUATION ON REAL-WORLD SATELLITE DATASET (ZENODO)")
    print("="*60)
    print(f"Total evaluated windows (after 30s rolling): {len(y)}")
    print(f"Normal windows: {sum(y == 0)}")
    print(f"Attacked windows: {sum(y == 1)}\n")
    
    print("Classification Report:")
    print(classification_report(y, y_pred, target_names=['Normal', 'Attack'], zero_division=0))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

if __name__ == "__main__":
    main()
