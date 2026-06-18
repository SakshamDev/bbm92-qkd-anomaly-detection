"""
tests/test_channel.py — Physics validation tests for the BBM92 channel simulator.

Implements Test A from §15.1 (Normal Channel) and additional validation
of individual atmospheric model components.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bbm92_drdo'))

from core.channel import (
    precipitation_model,
    scintillation_model,
    simulate_normal_channel,
    thermal_gradient_model,
)


class TestScintillationModel:
    """Tests for log-normal irradiance fluctuation model."""

    def test_output_shape(self):
        result = scintillation_model(1000)
        assert result.shape == (1000,)

    def test_output_range(self):
        result = scintillation_model(10000)
        assert np.all(result >= 0.0)
        assert np.all(result <= 0.95)

    def test_reproducibility(self):
        r1 = scintillation_model(100, seed=42)
        r2 = scintillation_model(100, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds(self):
        r1 = scintillation_model(100, seed=42)
        r2 = scintillation_model(100, seed=99)
        assert not np.array_equal(r1, r2)

    def test_no_nan(self):
        result = scintillation_model(10000)
        assert not np.any(np.isnan(result))

    def test_weak_turbulence_range(self):
        """Weak turbulence (sigma_I=0.1) should produce modest losses."""
        result = scintillation_model(10000, sigma_I=0.1)
        assert np.mean(result) < 0.3


class TestThermalGradientModel:
    """Tests for diurnal thermal cycle model."""

    def test_output_shape(self):
        result = thermal_gradient_model(86400)
        assert result.shape == (86400,)

    def test_non_negative(self):
        result = thermal_gradient_model(86400)
        assert np.all(result >= 0.0)

    def test_max_amplitude(self):
        """Maximum thermal QBER contribution should be ~0.016."""
        result = thermal_gradient_model(86400)
        assert np.max(result) <= 0.02  # small margin

    def test_periodicity(self):
        """Model should be approximately periodic over 24 hours."""
        result = thermal_gradient_model(172800)  # 2 days
        day1 = result[:86400]
        day2 = result[86400:]
        np.testing.assert_allclose(day1, day2, atol=1e-10)

    def test_no_nan(self):
        result = thermal_gradient_model(86400)
        assert not np.any(np.isnan(result))


class TestPrecipitationModel:
    """Tests for Poisson-arrival precipitation burst model."""

    def test_output_shape(self):
        result = precipitation_model(86400)
        assert result.shape == (86400,)

    def test_output_range(self):
        result = precipitation_model(86400)
        assert np.all(result >= 0.0)
        assert np.all(result <= 0.15)

    def test_mostly_zero(self):
        """Most of the time there is no precipitation event."""
        result = precipitation_model(86400, p_event=0.0003)
        zero_frac = np.mean(result == 0.0)
        assert zero_frac > 0.5  # at least half the time is clear

    def test_no_nan(self):
        result = precipitation_model(86400)
        assert not np.any(np.isnan(result))


class TestSimulateNormalChannel:
    """
    Test A from §15.1: Normal channel physics validation.

    Expected:
        - mean(QBER) between 0.01 and 0.05
        - mean(Bell_S) > 2.50
        - mean(coincidence_rate) > 3000
        - no NaN values
    """

    @pytest.fixture(scope='class')
    def normal_channel(self):
        """Generate once for all tests in this class."""
        return simulate_normal_channel(86400, seed=42)

    def test_acceptance_a_qber_range(self, normal_channel):
        """§15.1 Test A: mean QBER between 0.01 and 0.05."""
        mean_qber = float(np.mean(normal_channel['qber']))
        assert 0.01 <= mean_qber <= 0.05, (
            f"mean QBER = {mean_qber:.4f}, expected [0.01, 0.05]"
        )

    def test_acceptance_a_bell_s(self, normal_channel):
        """§15.1 Test A: mean Bell S > 2.50."""
        mean_s = float(np.mean(normal_channel['bell_S']))
        assert mean_s > 2.50, f"mean Bell S = {mean_s:.3f}, expected > 2.50"

    def test_acceptance_a_coincidence(self, normal_channel):
        """§15.1 Test A: mean coincidence_rate > 3000."""
        mean_c = float(np.mean(normal_channel['coincidence_rate']))
        assert mean_c > 3000, (
            f"mean coincidence = {mean_c:.0f}, expected > 3000"
        )

    def test_acceptance_a_no_nan(self, normal_channel):
        """§15.1 Test A: no NaN values in any column."""
        for key, arr in normal_channel.items():
            assert not np.any(np.isnan(arr.astype(float))), (
                f"NaN found in column '{key}'"
            )

    def test_output_keys(self, normal_channel):
        """All expected keys are present."""
        expected = {
            'qber', 'bell_S', 'coincidence_rate', 'visibility',
            'channel_loss_dB', 'detection_rate', 'label', 'attack_type'
        }
        assert set(normal_channel.keys()) == expected

    def test_output_shapes(self, normal_channel):
        """All arrays have correct length."""
        for key, arr in normal_channel.items():
            assert arr.shape == (86400,), (
                f"'{key}' shape = {arr.shape}, expected (86400,)"
            )

    def test_all_labels_zero(self, normal_channel):
        """Normal channel should have all labels = 0."""
        assert np.all(normal_channel['label'] == 0)

    def test_qber_bounds(self, normal_channel):
        """QBER should be clipped to [0.005, 0.12]."""
        assert np.all(normal_channel['qber'] >= 0.005)
        assert np.all(normal_channel['qber'] <= 0.12)

    def test_bell_s_bounds(self, normal_channel):
        """Bell S should be in [2.0, 2.828]."""
        assert np.all(normal_channel['bell_S'] >= 2.0)
        assert np.all(normal_channel['bell_S'] <= 2.828)

    def test_coincidence_bounds(self, normal_channel):
        """Coincidence rate should be non-negative and ≤ 12000."""
        assert np.all(normal_channel['coincidence_rate'] >= 0)
        assert np.all(normal_channel['coincidence_rate'] <= 12000)

    def test_visibility_bounds(self, normal_channel):
        """Visibility should be in [0.7, 1.0]."""
        assert np.all(normal_channel['visibility'] >= 0.7)
        assert np.all(normal_channel['visibility'] <= 1.0)

    def test_bell_s_qber_anticorrelation(self, normal_channel):
        """Bell S should be negatively correlated with QBER."""
        corr = float(np.corrcoef(
            normal_channel['qber'],
            normal_channel['bell_S'],
        )[0, 1])
        assert corr < 0, f"QBER-S correlation = {corr:.3f}, expected negative"
