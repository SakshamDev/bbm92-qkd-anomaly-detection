"""
ml/evaluate.py — Evaluation metrics and figure generation for publication.

Generates all evaluation figures specified in the tech spec:
    - ROC curve with AUC
    - Precision-Recall curve
    - Confusion matrix heatmap
    - Per-attack recall chart
    - SHAP summary
    - Feature ablation chart

Outputs are saved to data/figures/.

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.explain import generate_shap_summary_plot


logger = logging.getLogger(__name__)

# Consistent styling for all figures
plt.rcParams.update({
    'figure.facecolor': '#0e1117',
    'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#444444',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#aaaaaa',
    'ytick.color': '#aaaaaa',
    'grid.color': '#333333',
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 11,
})

def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def generate_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str) -> float:
    _ensure_dir(output_path)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color='#00d4ff', linewidth=2.5, label=f'XGBoost (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], '--', color='#666666', linewidth=1, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#00d4ff')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=13)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return auc_score


def generate_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str) -> None:
    _ensure_dir(output_path)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, color='#ff6b6b', linewidth=2.5, label='XGBoost')
    ax.axhline(y=y_true.mean(), color='#666666', linestyle='--', label=f'Baseline ({y_true.mean():.3f})')
    ax.fill_between(recall, precision, alpha=0.06, color='#ff6b6b')
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower left')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str) -> None:
    _ensure_dir(output_path)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'], ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Oranges', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'], ax=ax2)
    ax2.set_title('Confusion Matrix (Normalised)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def generate_per_attack_chart(attack_metrics: dict, output_path: str) -> None:
    _ensure_dir(output_path)
    labels = list(attack_metrics.keys())
    recalls = [attack_metrics[k]['recall'] * 100 for k in labels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, recalls, color='#2ca02c')
    ax.set_ylabel('Recall (%)', fontsize=12)
    ax.set_title('Recall by Attack Type', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim([80, 105])
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', color='#e0e0e0', fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def generate_ablation_chart(baseline_recall: float, ablated_recalls: dict, output_path: str) -> None:
    _ensure_dir(output_path)
    labels = list(ablated_recalls.keys())
    drops = [baseline_recall - ablated_recalls[k] for k in labels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, drops, color='#ffaa00')
    ax.set_xlabel('Drop in Recall (Baseline - Ablated)', fontsize=12)
    ax.set_title('Feature Group Ablation Impact', fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def run_full_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    attack_types: np.ndarray,
    model_artifacts: dict[str, Any],
    figures_dir: str = 'data/figures/',
) -> dict[str, Any]:
    """Orchestrates generation of all paper figures."""
    logger.info("=== Running figure generation for paper ===")
    
    xgb_prob = model_artifacts['xgb'].predict_proba(X)[:, 1]
    threshold = model_artifacts['config'].get('threshold', 0.50)
    y_pred = (xgb_prob >= threshold).astype(int)
    
    from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix
    p = precision_score(y, y_pred, zero_division=0)
    r = recall_score(y, y_pred, zero_division=0)
    f2 = fbeta_score(y, y_pred, beta=2, zero_division=0)
    cm = confusion_matrix(y, y_pred)
    logger.info(f"SEED 42 METRICS (tau={threshold:.4f}):")
    logger.info(f"Precision: {p:.4f}")
    logger.info(f"Recall: {r:.4f}")
    logger.info(f"F2: {f2:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")


    # 1. ROC Curve
    generate_roc_curve(y, xgb_prob, f'{figures_dir}/roc_curve.png')
    
    # 2. Precision-Recall Curve
    generate_precision_recall_curve(y, xgb_prob, f'{figures_dir}/precision_recall_curve.png')
    
    # 3. Confusion Matrix
    generate_confusion_matrix(y, y_pred, f'{figures_dir}/confusion_matrix.png')
    
    # 4. SHAP Summary
    generate_shap_summary_plot(model_artifacts, X, output_path=f'{figures_dir}/shap_summary.png')
    
    # 5. Per-attack recall chart
    attack_names = {1: 'Intercept-Resend', 2: 'Detector Blinding', 3: 'MitM', 4: 'Blended Sub-Threshold'}
    attack_metrics = {}
    for a_type, name in attack_names.items():
        mask = (attack_types == a_type)
        if mask.sum() > 0:
            attack_metrics[name] = {'recall': float(recall_score(y[mask], y_pred[mask], zero_division=0))}
    if attack_metrics:
        generate_per_attack_chart(attack_metrics, f'{figures_dir}/per_attack_recall.png')
    
    # 6. Ablation Simulation (Mocked here since actual ablation requires retraining, 
    # but we are freezing models. We use the documented ablation results.)
    # M-3 Fix: Removed fabricated ablation values. Ablation chart generation requires
    # explicitly running `feature_group_ablation.py`.
    
    logger.info("All paper figures saved to %s", figures_dir)
    return {"status": "success"}

