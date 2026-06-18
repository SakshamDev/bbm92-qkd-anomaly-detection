"""
tests/test_features.py — Feature engineering pipeline validation.

Tests the 24-dimensional feature extraction and matrix building.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bbm92_drdo'))

from ml.features import (
    FEATURE_NAMES,
    WINDOW,
    _autocorr,
    build_feature_matrix,
    extract_features_single,
)


def _make_test_window(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic window DataFrame for testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'qber': rng.uniform(0.01, 0.05, n),
        'bell_S': rng.uniform(2.5, 2.8, n),
        'coincidence_rate': rng.uniform(5000, 9000, n),
        'visibility': rng.uniform(0.90, 0.98, n),
        'channel_loss_dB': rng.uniform(3, 8, n),
        'detection_rate': rng.uniform(10000, 18000, n),
    })


class TestFeatureNames:
    """Tests for the canonical feature names list."""

    def test_feature_count(self):
        assert len(FEATURE_NAMES) == 24

    def test_unique_names(self):
        assert len(set(FEATURE_NAMES)) == 24

    def test_window_size(self):
        assert WINDOW == 30


class TestAutocorrelation:
    """Tests for the _autocorr helper function."""

    def test_constant_signal(self):
        """Constant signal should have zero autocorrelation."""
        x = np.ones(100)
        assert _autocorr(x, 1) == 0.0

    def test_short_signal(self):
        """Signal shorter than lag should return 0."""
        x = np.array([1.0, 2.0])
        assert _autocorr(x, 5) == 0.0

    def test_known_autocorrelation(self):
        """Sinusoidal signal should have positive lag-1 autocorrelation."""
        t = np.linspace(0, 4 * np.pi, 200)
        x = np.sin(t)
        ac = _autocorr(x, 1)
        assert ac > 0.9  # sin is strongly autocorrelated at lag 1

    def test_bounds(self):
        """Autocorrelation should be in [-1, 1]."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        for lag in [1, 5, 10]:
            ac = _autocorr(x, lag)
            assert -1.0 <= ac <= 1.0


class TestExtractFeaturesSingle:
    """Tests for single-window feature extraction."""

    def test_output_shape(self):
        window = _make_test_window(30)
        features = extract_features_single(window)
        assert features.shape == (24,)

    def test_output_dtype(self):
        window = _make_test_window(30)
        features = extract_features_single(window)
        assert features.dtype == np.float32

    def test_no_nan(self):
        window = _make_test_window(30)
        features = extract_features_single(window)
        assert not np.any(np.isnan(features))

    def test_no_inf(self):
        window = _make_test_window(30)
        features = extract_features_single(window)
        assert not np.any(np.isinf(features))

    def test_qber_mean_reasonable(self):
        """QBER mean feature should match actual window mean."""
        window = _make_test_window(30)
        features = extract_features_single(window)
        expected_mean = float(np.mean(window['qber'].values))
        # Feature 0 = qber_mean
        np.testing.assert_allclose(features[0], expected_mean, rtol=1e-4)

    def test_bell_s_mean_reasonable(self):
        """Bell S mean feature should match actual window mean."""
        window = _make_test_window(30)
        features = extract_features_single(window)
        expected_mean = float(np.mean(window['bell_S'].values))
        # Feature 6 = bell_S_mean
        np.testing.assert_allclose(features[6], expected_mean, rtol=1e-4)

    def test_reproducibility(self):
        """Same input should produce same output."""
        window = _make_test_window(30, seed=42)
        f1 = extract_features_single(window)
        f2 = extract_features_single(window)
        np.testing.assert_array_equal(f1, f2)

    def test_different_windows_different_features(self):
        """Different inputs should produce different outputs."""
        w1 = _make_test_window(30, seed=42)
        w2 = _make_test_window(30, seed=99)
        f1 = extract_features_single(w1)
        f2 = extract_features_single(w2)
        assert not np.array_equal(f1, f2)


class TestBuildFeatureMatrix:
    """Tests for rolling feature matrix construction."""

    def test_output_shapes(self):
        """Feature matrix should have correct shape."""
        n = 200
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'qber': rng.uniform(0.01, 0.05, n),
            'bell_S': rng.uniform(2.5, 2.8, n),
            'coincidence_rate': rng.uniform(5000, 9000, n),
            'visibility': rng.uniform(0.90, 0.98, n),
            'channel_loss_dB': rng.uniform(3, 8, n),
            'detection_rate': rng.uniform(10000, 18000, n),
            'label': rng.choice([0, 1], n, p=[0.85, 0.15]),
        })
        X, y = build_feature_matrix(df, window=30)
        assert X.shape == (n - 30, 24)
        assert y.shape == (n - 30,)

    def test_no_nan_in_matrix(self):
        """No NaN values in the feature matrix."""
        n = 100
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'qber': rng.uniform(0.01, 0.05, n),
            'bell_S': rng.uniform(2.5, 2.8, n),
            'coincidence_rate': rng.uniform(5000, 9000, n),
            'visibility': rng.uniform(0.90, 0.98, n),
            'channel_loss_dB': rng.uniform(3, 8, n),
            'detection_rate': rng.uniform(10000, 18000, n),
            'label': np.zeros(n, dtype=int),
        })
        X, y = build_feature_matrix(df, window=30)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))
