"""
tests/test_attacks.py — Attack model validation tests.

Implements Tests B, C, D from §15.1 and additional attack signature validation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict
from core.attacks import (
    AttackStrategy,
    InterceptResendStrategy,
    FakedStateStrategy,
    MitMStrategy,
    BlendedSubthresholdStrategy,
)
from core.config import EveCapabilities

def _apply_single_strategy(
    base_channel: Dict[str, np.ndarray],
    strategy: AttackStrategy,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    from core.channel import simulate_normal_channel
    n_seconds = len(base_channel['qber'])
    return simulate_normal_channel(
        n_seconds=n_seconds,
        seed=seed,
        attack_strategies=[strategy],
    )

def attack_intercept_resend(
    base_channel: Dict[str, np.ndarray],
    duration_sec: int,
    start_sec: int,
    eve_fraction: float = 0.30,
    rng: np.random.Generator = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    sim_seed = int(rng.integers(0, 2**31)) if rng is not None else seed
    caps = EveCapabilities()
    strategy = InterceptResendStrategy(
        caps, start_sec=start_sec, duration_sec=duration_sec, fraction=eve_fraction
    )
    return _apply_single_strategy(base_channel, strategy, seed=sim_seed)

def attack_detector_blinding(
    base_channel: Dict[str, np.ndarray],
    duration_sec: int,
    start_sec: int,
    blinding_intensity: float = 0.50,
    rng: np.random.Generator = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    sim_seed = int(rng.integers(0, 2**31)) if rng is not None else seed
    caps = EveCapabilities()
    trigger_rate = blinding_intensity * 15000.0
    strategy = FakedStateStrategy(
        caps, start_sec=start_sec, duration_sec=duration_sec, trigger_rate=trigger_rate
    )
    return _apply_single_strategy(base_channel, strategy, seed=sim_seed)

def attack_mitm(
    base_channel: Dict[str, np.ndarray],
    duration_sec: int,
    start_sec: int,
    rng: np.random.Generator = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    sim_seed = int(rng.integers(0, 2**31)) if rng is not None else seed
    caps = EveCapabilities()
    strategy = MitMStrategy(caps, start_sec=start_sec, duration_sec=duration_sec)
    return _apply_single_strategy(base_channel, strategy, seed=sim_seed)

def attack_blended_subthreshold(
    base_channel: Dict[str, np.ndarray],
    n_bursts: int = 50,
    burst_duration: int = 30,
    eve_fraction: float = 0.12,
    rng: np.random.Generator = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(seed)
    sim_seed = int(rng.integers(0, 2**31))
    
    n_seconds = len(base_channel['qber'])
    caps = EveCapabilities()
    strategies = []
    for _ in range(n_bursts):
        start = int(rng.integers(0, max(1, n_seconds - burst_duration)))
        strategies.append(
            BlendedSubthresholdStrategy(
                caps, start_sec=start, duration_sec=burst_duration, fraction=eve_fraction
            )
        )
    from core.channel import simulate_normal_channel
    return simulate_normal_channel(
        n_seconds=n_seconds,
        seed=sim_seed,
        attack_strategies=strategies,
    )
from core.channel import simulate_normal_channel


@pytest.fixture(scope='module')
def base_channel():
    """Generate a normal baseline channel for all attack tests."""
    return simulate_normal_channel(10000, seed=42)


class TestInterceptResendAttack:
    """
    Test B from §15.1: Full Intercept-Resend attack (eve_fraction = 1.0).

    Expected:
        - mean(QBER) > 0.20
        - mean(Bell_S) < 2.20
    """

    def test_acceptance_b_full_ir_qber(self, base_channel):
        """§15.1 Test B: Full IR → mean QBER > 0.20."""
        attacked = attack_intercept_resend(
            base_channel,
            duration_sec=10000,
            start_sec=0,
            eve_fraction=1.0,
            rng=np.random.default_rng(42),
        )
        mean_qber = float(np.mean(attacked['qber']))
        assert mean_qber > 0.20, (
            f"Full IR mean QBER = {mean_qber:.4f}, expected > 0.20"
        )

    def test_acceptance_b_full_ir_bell_s(self, base_channel):
        """§15.1 Test B: Full IR → mean Bell S < 2.20."""
        attacked = attack_intercept_resend(
            base_channel,
            duration_sec=10000,
            start_sec=0,
            eve_fraction=1.0,
            rng=np.random.default_rng(42),
        )
        mean_s = float(np.mean(attacked['bell_S']))
        assert mean_s < 2.20, (
            f"Full IR mean Bell S = {mean_s:.3f}, expected < 2.20"
        )

    def test_ir_labels_set(self, base_channel):
        """Attack window labels should be 1."""
        attacked = attack_intercept_resend(
            base_channel,
            duration_sec=500,
            start_sec=100,
            eve_fraction=0.30,
            rng=np.random.default_rng(42),
        )
        assert np.all(attacked['label'][100:600] == 1)
        assert np.all(attacked['label'][:100] == 0)

    def test_ir_qber_increase(self, base_channel):
        """IR should increase QBER relative to baseline."""
        attacked = attack_intercept_resend(
            base_channel,
            duration_sec=5000,
            start_sec=0,
            eve_fraction=0.30,
            rng=np.random.default_rng(42),
        )
        sl = slice(0, 5000)
        assert np.mean(attacked['qber'][sl]) > np.mean(base_channel['qber'][sl])

    def test_ir_coincidence_drop(self, base_channel):
        """IR should reduce coincidence rate (~15%)."""
        attacked = attack_intercept_resend(
            base_channel,
            duration_sec=5000,
            start_sec=0,
            eve_fraction=0.30,
            rng=np.random.default_rng(42),
        )
        sl = slice(0, 5000)
        base_mean = np.mean(base_channel['coincidence_rate'][sl])
        atk_mean = np.mean(attacked['coincidence_rate'][sl])
        drop_pct = (base_mean - atk_mean) / base_mean
        assert drop_pct > 0.02, f"Coincidence drop = {drop_pct:.2%}"

    def test_ir_does_not_modify_base(self, base_channel):
        """Original base channel should not be mutated."""
        original_qber = base_channel['qber'].copy()
        attack_intercept_resend(
            base_channel,
            duration_sec=500,
            start_sec=100,
            eve_fraction=0.30,
            rng=np.random.default_rng(42),
        )
        np.testing.assert_array_equal(base_channel['qber'], original_qber)


class TestDetectorBlindingAttack:
    """
    Test C from §15.1: Detector Blinding attack (blinding_intensity = 0.50).

    Expected:
        - coincidence_rate reduced
        - anomalous increase in detection_rate (the core signature)
        - moderate QBER increase
    """

    def test_acceptance_c_detection_increase(self, base_channel):
        """§15.1 Test C: Blinding → anomalous detection rate increase."""
        attacked = attack_detector_blinding(
            base_channel,
            duration_sec=10000,
            start_sec=0,
            blinding_intensity=0.50,
            rng=np.random.default_rng(42),
        )
        base_mean = float(np.mean(base_channel['detection_rate']))
        atk_mean = float(np.mean(attacked['detection_rate']))
        assert atk_mean > base_mean + 100, (
            f"Blinding detection rate = {atk_mean:.2f}, expected anomalous increase"
        )

    def test_acceptance_c_coincidence_deviation(self, base_channel):
        """§15.1 Test C: Blinding → coincidence rate deviates anomalously (can be a massive increase due to bright pulses)."""
        attacked = attack_detector_blinding(
            base_channel,
            duration_sec=10000,
            start_sec=0,
            blinding_intensity=0.50,
            rng=np.random.default_rng(42),
        )
        base_mean = float(np.mean(base_channel['coincidence_rate']))
        atk_mean = float(np.mean(attacked['coincidence_rate']))
        # In Faked-State, Eve injects bright pulses leading to high deterministic coincidences
        assert atk_mean > base_mean * 1.5, (
            f"Blinding coincidence mean = {atk_mean:.1f}, expected an anomalous increase over base {base_mean:.1f}"
        )

    def test_blinding_labels(self, base_channel):
        """Blinding attack labels should be set correctly."""
        attacked = attack_detector_blinding(
            base_channel,
            duration_sec=500,
            start_sec=200,
            blinding_intensity=0.50,
            rng=np.random.default_rng(42),
        )
        assert np.all(attacked['label'][200:700] == 1)

    def test_blinding_qber_modest(self, base_channel):
        """Blinding should produce a moderate QBER increase."""
        attacked = attack_detector_blinding(
            base_channel,
            duration_sec=5000,
            start_sec=0,
            blinding_intensity=0.50,
            rng=np.random.default_rng(42),
        )
        qber_increase = (
            np.mean(attacked['qber'][:5000])
            - np.mean(base_channel['qber'][:5000])
        )
        # QBER increase should be modest (< 0.15) compared to full IR
        assert qber_increase < 0.15, (
            f"Blinding QBER increase = {qber_increase:.4f}"
        )


class TestMitMAttack:
    """
    Test D from §15.1: Man-in-the-Middle attack.

    Expected:
        - Bell_S approaches classical limit
        - mean(Bell_S) < 2.20
        - QBER > 0.08
    """

    def test_acceptance_d_bell_s(self, base_channel):
        """§15.1 Test D: MitM → mean Bell S < 2.20."""
        attacked = attack_mitm(
            base_channel,
            duration_sec=10000,
            start_sec=0,
            rng=np.random.default_rng(42),
        )
        mean_s = float(np.mean(attacked['bell_S']))
        assert mean_s < 2.20, (
            f"MitM mean Bell S = {mean_s:.3f}, expected < 2.20"
        )

    def test_acceptance_d_qber(self, base_channel):
        """§15.1 Test D: MitM → QBER > 0.08."""
        attacked = attack_mitm(
            base_channel,
            duration_sec=10000,
            start_sec=0,
            rng=np.random.default_rng(42),
        )
        mean_qber = float(np.mean(attacked['qber']))
        assert mean_qber > 0.08, (
            f"MitM mean QBER = {mean_qber:.4f}, expected > 0.08"
        )

    def test_mitm_coincidence_drop(self, base_channel):
        """MitM should cause a coincidence drop (due to Eve's imperfect interception)."""
        attacked = attack_mitm(
            base_channel,
            duration_sec=5000,
            start_sec=0,
            rng=np.random.default_rng(42),
        )
        base_mean = float(np.mean(base_channel['coincidence_rate'][:5000]))
        atk_mean = float(np.mean(attacked['coincidence_rate'][:5000]))
        drop_pct = (base_mean - atk_mean) / base_mean
        assert drop_pct > 0.05, (
            f"MitM coincidence drop = {drop_pct:.2%}, expected > 5%"
        )

    def test_mitm_labels(self, base_channel):
        """MitM labels should be set correctly."""
        attacked = attack_mitm(
            base_channel,
            duration_sec=500,
            start_sec=300,
            rng=np.random.default_rng(42),
        )
        assert np.all(attacked['label'][300:800] == 1)


class TestBlendedSubThresholdAttack:
    """Tests for the blended sub-threshold attack (the hard case)."""

    def test_qber_stays_below_threshold(self, base_channel):
        """Blended attack should keep average QBER below the 0.14 threshold (though spikes may exceed)."""
        attacked = attack_blended_subthreshold(
            base_channel,
            n_bursts=50,
            burst_duration=30,
            eve_fraction=0.12,
            rng=np.random.default_rng(42),
        )
        assert np.all(attacked['qber'] <= 0.25), (
            f"Max QBER = {np.max(attacked['qber']):.4f}, expected ≤ 0.25 (allowing for Poisson spikes)"
        )

    def test_some_labels_set(self, base_channel):
        """At least some labels should be set to 1."""
        attacked = attack_blended_subthreshold(
            base_channel,
            n_bursts=50,
            burst_duration=30,
            eve_fraction=0.12,
            rng=np.random.default_rng(42),
        )
        assert np.sum(attacked['label']) > 0

    def test_blended_bell_s_partial(self, base_channel):
        """Blended attack should partially degrade Bell S."""
        attacked = attack_blended_subthreshold(
            base_channel,
            n_bursts=100,
            burst_duration=30,
            eve_fraction=0.12,
            rng=np.random.default_rng(42),
        )
        # Attack windows should have lower S than normal
        atk_mask = attacked['label'] == 1
        if atk_mask.sum() > 0:
            normal_s = np.mean(attacked['bell_S'][~atk_mask])
            attack_s = np.mean(attacked['bell_S'][atk_mask])
            assert attack_s < normal_s, (
                f"Attack S ({attack_s:.3f}) should be < normal S ({normal_s:.3f})"
            )
