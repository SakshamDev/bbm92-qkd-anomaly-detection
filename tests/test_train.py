"""
tests/test_train.py — ML pipeline training and inference validation.

Tests model training outputs.
"""

import sys
import os
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.train import train_model

def _make_synthetic_data(
    n_samples: int = 2000,
    n_features: int = 24,
    attack_rate: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic feature data for fast testing."""
    rng = np.random.default_rng(seed)
    n_attack = int(n_samples * attack_rate)
    n_normal = n_samples - n_attack

    X_normal = rng.normal(0, 1, (n_normal, n_features)).astype(np.float32)
    X_attack = rng.normal(2, 1.5, (n_attack, n_features)).astype(np.float32)

    X = np.vstack([X_normal, X_attack])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_attack)]).astype(int)

    perm = rng.permutation(n_samples)
    return X[perm], y[perm]

class TestTrainModel:
    """Tests for training pipeline."""

    @pytest.fixture(scope='class')
    def trained(self, tmp_path_factory):
        """Train once for all tests in this class."""
        model_dir = str(tmp_path_factory.mktemp('models'))
        X, y = _make_synthetic_data(2000, seed=42)
        metrics = train_model(X, y, model_dir=model_dir + '/')
        return metrics, model_dir

    def test_returns_cv_metrics(self, trained):
        metrics, _ = trained
        assert 'cv_metrics' in metrics
        assert len(metrics['cv_metrics']) == 4  # 4-fold with gaps for n_splits=5

    def test_returns_threshold(self, trained):
        metrics, _ = trained
        assert 'final_threshold' in metrics
        assert 0.0 <= metrics['final_threshold'] <= 1.0

    def test_model_files_exist(self, trained):
        _, model_dir = trained
        assert Path(f'{model_dir}/xgb_model.json').exists()
        assert Path(f'{model_dir}/config.json').exists()

