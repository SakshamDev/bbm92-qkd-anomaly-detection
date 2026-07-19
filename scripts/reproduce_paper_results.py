"""
scripts/retrain_and_evaluate.py — Master script for multi-seed retraining.

Implements the 5-seed evaluation pipeline with a strict 6-hour holdout test set.
Computes baselines and reports per-attack-type metrics for the proposed XGBoost model.
"""

import json
import logging
import os
import shutil
import time
import sys
import gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score
from core.telemetry import build_telemetry_dataset
from ml.features import FEATURE_NAMES, build_feature_matrix
from ml.train import train_model
from ml.evaluate import run_full_evaluation

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 7, 2024, 314]
N_SECONDS = 86400
TRAIN_SECONDS = 64800  # First 18 hours

def evaluate_baselines(X_test, y_test, attack_types, qber_test, xgb_model, config):
    """Evaluate the baselines on the holdout test set."""
    
    metrics = {
        'threshold': {},
        'xgb': {},
    }

    # 1. QBER > 11% (Physics threshold)
    pred_thresh = (qber_test > 0.11).astype(int)
    metrics['threshold'] = {
        'recall': recall_score(y_test, pred_thresh, zero_division=0),
        'precision': precision_score(y_test, pred_thresh, zero_division=0),
        'f2': fbeta_score(y_test, pred_thresh, beta=2, zero_division=0)
    }

    # 2. Proposed XGBoost Model
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    pred_xgb = (xgb_prob >= config['threshold']).astype(int)
    metrics['xgb'] = {
        'recall': recall_score(y_test, pred_xgb, zero_division=0),
        'precision': precision_score(y_test, pred_xgb, zero_division=0),
        'f2': fbeta_score(y_test, pred_xgb, beta=2, zero_division=0),
        'auc': roc_auc_score(y_test, xgb_prob)
    }

    # Per-attack-type breakdown for proposed model
    attack_metrics = {}
    attack_names = {
        1: 'Intercept-Resend',
        2: 'Detector Blinding',
        3: 'MitM',
        4: 'Blended Sub-Threshold'
    }
    
    for a_type, name in attack_names.items():
        mask = (attack_types == a_type)
        if mask.sum() > 0:
            a_preds = pred_xgb[mask]
            a_y = y_test[mask]
            attack_metrics[name] = {
                'recall': recall_score(a_y, a_preds, zero_division=0),
                'count': int(mask.sum())
            }

    return metrics, attack_metrics

def main():
    logger.info("Starting multi-seed evaluation pipeline...")
    
    results = {
        'threshold': {'recall': [], 'precision': [], 'f2': []},
        'xgb': {'recall': [], 'precision': [], 'f2': [], 'auc': []},
    }
    
    attack_results = {
        'Intercept-Resend': {'recall': [], 'count': []},
        'Detector Blinding': {'recall': [], 'count': []},
        'MitM': {'recall': [], 'count': []},
        'Blended Sub-Threshold': {'recall': [], 'count': []}
    }

    for i, seed in enumerate(SEEDS):
        logger.info(f"\n{'='*40}\nProcessing Seed {seed}\n{'='*40}")
        
        # 1. Load or Generate data
        data_path = f'data/telemetry_{seed}.parquet'
        if os.path.exists(data_path):
            logger.info(f"Loading existing telemetry dataset from {data_path}")
            df = pd.read_parquet(data_path)
        else:
            df = build_telemetry_dataset(n_seconds=N_SECONDS, seed=seed, output_path=data_path)
        
        # 2. Extract features
        X, y = build_feature_matrix(df, window=30)
        attack_types = df['attack_type'].values[30:]  # Match window offset
        qber = df['qber'].values[30:]
        
        # 3. Train/Test split (strict temporal holdout)
        X_train, y_train = X[:TRAIN_SECONDS - 30], y[:TRAIN_SECONDS - 30]
        X_test, y_test = X[TRAIN_SECONDS - 30:], y[TRAIN_SECONDS - 30:]
        attack_types_test = attack_types[TRAIN_SECONDS - 30:]
        qber_test = qber[TRAIN_SECONDS - 30:]
        
        # 4. Train model on train block ONLY
        train_model(X_train, y_train, model_dir=f'models_seed_{seed}')
        
        # Load artifacts
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(f'models_seed_{seed}/xgb_model.json')
        with open(f'models_seed_{seed}/config.json', 'r') as f:
            config = json.load(f)
            
        # 5. Evaluate on holdout test set
        metrics, a_metrics = evaluate_baselines(
            X_test, y_test, attack_types_test, qber_test, 
            xgb_model, config
        )
        
        # Collect results
        for model in ['threshold', 'xgb']:
            for m in ['recall', 'precision', 'f2', 'auc']:
                if m in metrics[model]:
                    results[model][m].append(metrics[model][m])
                    
        for name, m in a_metrics.items():
            attack_results[name]['recall'].append(m['recall'])
            attack_results[name]['count'].append(m['count'])
            
        # Clean up models to save space
        if seed != 42:
            # Keep only the config.json for downstream analysis
            if os.path.exists(f'models_seed_{seed}/xgb_model.json'):
                os.remove(f'models_seed_{seed}/xgb_model.json')
        else:
            # Keep the first seed's model as the "primary" model for the dashboard
            if os.path.exists('models'):
                shutil.rmtree('models')
            shutil.copytree(f'models_seed_{seed}', 'models')
            shutil.rmtree(f'models_seed_{seed}')
            
            # Generate publication figures using the primary holdout set
            model_artifacts = {
                'xgb': xgb_model,
                'config': config
            }
            logger.info("Generating publication figures for primary holdout set...")
            run_full_evaluation(X_test, y_test, attack_types_test, model_artifacts, figures_dir='data/figures/')
            
        # Free memory of all large data objects explicitly
        del df, X, y, attack_types, qber, X_train, y_train, X_test, y_test, attack_types_test, qber_test
        del xgb_model, metrics, a_metrics
        if seed == 42:
            del model_artifacts
        gc.collect()
        
        # Cool down if not the last seed
        if seed != SEEDS[-1]:
            logger.info("Cooling down for 180 seconds to prevent thermal throttling... (skipped)")

    # --- Print Comparison Report ---
    print("\n" + "="*60)
    print("FINAL EVALUATION REPORT (Mean ± Std over 5 seeds on HOLDOUT set)")
    print("="*60 + "\n")
    
    print("Table 1: Baseline Comparison")
    print(f"{'Method':<22} | {'Recall':<14} | {'Precision':<14} | {'F2':<14} | {'AUC':<14}")
    print("-" * 75)
    
    def fmt(vals):
        if not vals: return "N/A"
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
        
    for model in ['threshold', 'xgb']:
        r = results[model]
        name = "QBER > 11%" if model == 'threshold' else "Proposed XGBoost"
        print(f"{name:<22} | {fmt(r['recall']):<14} | {fmt(r['precision']):<14} | {fmt(r['f2']):<14} | {fmt(r.get('auc', [])):<14}")
        
    print("\nTable 2: Per-Attack-Type Recall (Proposed XGBoost)")
    print(f"{'Attack Type':<22} | {'Recall':<14} | Avg Count per Holdout")
    print("-" * 65)
    for name, r in attack_results.items():
        if r['count']:
            print(f"{name:<22} | {fmt(r['recall']):<14} | {np.mean(r['count']):.0f}")
            
    print("\nTable 3: Old vs New Metrics (XGBoost)")
    print(f"{'Metric':<12} | Old (Inflated Ensemble) | New (Honest, 5-seed Holdout XGBoost)")
    print("-" * 60)
    print(f"{'Recall':<12} | 0.970                   | {fmt(results['xgb']['recall'])}")
    print(f"{'Precision':<12} | 0.970                   | {fmt(results['xgb']['precision'])}")
    print(f"{'F2':<12} | 0.970                   | {fmt(results['xgb']['f2'])}")
    print(f"{'AUC':<12} | 0.990                   | {fmt(results['xgb']['auc'])}")

if __name__ == "__main__":
    main()
