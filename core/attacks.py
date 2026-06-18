"""
core/attacks.py — Attack scenario generators for BBM92 QKD channels.

Implements four physically motivated attack models that modify base telemetry:
  1. Intercept-Resend (IR): Eve intercepts photons, measures, and resends.
  2. Detector Blinding: Eve blinds Bob's SPADs with bright light, controls clicks.
  3. Man-in-the-Middle (MitM): Eve replaces the entanglement source.
  4. Blended Sub-Threshold: Short clustered bursts hidden in atmospheric noise.

Each attack function takes a base telemetry dict (from channel.simulate_normal_channel)
and returns a modified copy with physically correct perturbations applied.

Attack taxonomy (from §3.1):
  ┌──────────────────┬────────────┬───────────────┬──────────────────┬──────────────┐
  │ Attack           │ QBER       │ Bell S        │ Coincidence      │ Detectability│
  ├──────────────────┼────────────┼───────────────┼──────────────────┼──────────────┤
  │ IR               │ +6–18%     │ → 2.0–2.2     │ ~15% drop        │ Medium       │
  │ Detector Blind   │ +3–8%      │ → 2.0–2.2     │ ~40–60% drop     │ Hard         │
  │ MitM             │ +8–15%     │ → ≈2.0        │ ~60% drop        │ Easy         │
  │ Blended          │ +3–8%      │ Partial       │ ~10% drop        │ Very Hard    │
  └──────────────────┴────────────┴───────────────┴──────────────────┴──────────────┘

"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def attack_intercept_resend(
    base: Dict[str, np.ndarray],
    duration_sec: int,
    start_sec: int,
    eve_fraction: float = 0.30,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Intercept-Resend attack: Eve intercepts a fraction of photon pairs,
    measures in a random basis, and resends.

    Physical derivation:
        QBER_attack = QBER_base + eve_fraction × 0.25
        S_attack = 2√2 × (1 - eve_fraction) + 2.0 × eve_fraction
                 = 2√2 - eve_fraction × (2√2 - 2.0)

    Eve's random-basis measurement collapses entanglement on intercepted photons,
    causing 50% basis mismatch → 50% errors on those → 25% net QBER increase
    per unit interception fraction.

    Args:
        base: Base telemetry dict from simulate_normal_channel().
        duration_sec: Duration of the attack in seconds.
        start_sec: Start time (second index) of the attack.
        eve_fraction: Fraction of photon pairs intercepted (0–1).
        rng: NumPy random generator.

    Returns:
        Modified telemetry dict with attack applied. Label set to 1
        in the attack window.
    """
    if rng is None:
        rng = np.random.default_rng()

    attacked = {k: v.copy() for k, v in base.items()}
    end_sec = min(start_sec + duration_sec, len(base['qber']))
    sl = slice(start_sec, end_sec)
    n = end_sec - start_sec

    logger.info(
        "IR attack: t=%d–%d (%ds), eve_fraction=%.2f",
        start_sec, end_sec, n, eve_fraction,
    )

    # QBER: step increase from the IR formula with noise
    delta_qber = eve_fraction * 0.25 + rng.normal(0, 0.005, n)
    attacked['qber'][sl] = np.clip(attacked['qber'][sl] + delta_qber, 0, 0.35)

    # Bell S: collapse proportional to intercept fraction
    s_collapse = eve_fraction * (2.828 - 2.0)
    attacked['bell_S'][sl] -= (s_collapse + rng.normal(0, 0.03, n))
    attacked['bell_S'][sl] = np.clip(attacked['bell_S'][sl], 2.0, 2.828)

    # Coincidence: slight drop (photons lost during Eve's measurement)
    coincidence_drop = rng.uniform(0.10, 0.20, n)
    attacked['coincidence_rate'][sl] = np.round(
        attacked['coincidence_rate'][sl] * (1.0 - coincidence_drop)
    ).astype(int)

    attacked['label'][sl] = 1
    attacked['attack_type'][sl] = 1  # IR
    return attacked


def attack_detector_blinding(
    base: Dict[str, np.ndarray],
    duration_sec: int,
    start_sec: int,
    blinding_intensity: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Detector Blinding attack: Eve blinds Bob's SPADs with bright classical light.

    Eve sends a continuous bright laser at Bob's Single-Photon Avalanche Diodes
    (SPADs), driving them from Geiger mode into linear mode. In linear mode,
    the detectors are no longer single-photon sensitive. Eve then sends
    tailored trigger pulses to control exactly when Bob's detectors click.

    This attack is physically valid against BBM92 (unlike PNS, which only
    applies to attenuated-laser BB84).

    Signatures:
        - QBER: moderate increase (+3–8%). Eve cannot perfectly replicate
          entanglement statistics with classical trigger pulses.
        - Bell S: significant drop to near-classical (~2.0–2.2). Eve sends
          separable states, breaking entanglement correlations.
        - Coincidence: dramatic drop (~40–60%). Eve's trigger pulses don't
          perfectly replicate SPDC timing correlations.
        - Detection rate: ANOMALOUS INCREASE. The bright blinding light
          itself triggers many non-coincident single-detector clicks.
          This is the unique signature distinguishing detector blinding
          from other attacks.

    Args:
        base: Base telemetry dict from simulate_normal_channel().
        duration_sec: Duration of the attack in seconds.
        start_sec: Start time (second index) of the attack.
        blinding_intensity: Intensity of blinding (0–1). Higher = more
            aggressive blinding, easier to detect but more control.
        rng: NumPy random generator.

    Returns:
        Modified telemetry dict with detector blinding attack applied.
    """
    if rng is None:
        rng = np.random.default_rng()

    attacked = {k: v.copy() for k, v in base.items()}
    end_sec = min(start_sec + duration_sec, len(base['qber']))
    sl = slice(start_sec, end_sec)
    n = end_sec - start_sec

    logger.info(
        "Detector blinding attack: t=%d–%d (%ds), intensity=%.2f",
        start_sec, end_sec, n, blinding_intensity,
    )

    # QBER: moderate increase (imperfect click control)
    delta_qber = blinding_intensity * 0.12 + rng.normal(0, 0.008, n)
    attacked['qber'][sl] = np.clip(attacked['qber'][sl] + delta_qber, 0, 0.25)

    # Bell S: significant drop (Eve sends separable states)
    s_drop = blinding_intensity * (2.828 - 2.0) * 0.8
    attacked['bell_S'][sl] -= (s_drop + rng.normal(0, 0.04, n))
    attacked['bell_S'][sl] = np.clip(attacked['bell_S'][sl], 2.0, 2.828)

    # Coincidence: dramatic drop (broken timing correlations)
    coincidence_drop = rng.uniform(
        0.35 * blinding_intensity, 0.65 * blinding_intensity, n
    )
    attacked['coincidence_rate'][sl] = np.round(
        attacked['coincidence_rate'][sl] * (1.0 - coincidence_drop)
        + rng.normal(0, 15, n)
    ).astype(int)
    attacked['coincidence_rate'][sl] = np.clip(
        attacked['coincidence_rate'][sl], 0, None
    )

    # Detection rate: ANOMALOUS INCREASE (unique blinding signature)
    # Bright blinding laser causes excess single-detector clicks
    blinding_excess = blinding_intensity * rng.uniform(300, 800, n)
    attacked['detection_rate'][sl] += blinding_excess

    attacked['label'][sl] = 1
    attacked['attack_type'][sl] = 2  # Detector Blinding
    return attacked


def attack_mitm(
    base: Dict[str, np.ndarray],
    duration_sec: int,
    start_sec: int,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Man-in-the-Middle attack: Eve positions herself as the entanglement source.

    Eve intercepts the quantum channel completely and sends separable
    (non-entangled) states to both Alice and Bob while building her own
    correlated key from the measurement results.

    Signatures:
        - Full Bell-inequality violation (S → classical limit ≈ 2.0)
        - Severe coincidence drop (~60%, no genuine SPDC source)
        - Large QBER (separable states produce high error rates)

    This is the most detectable attack type.

    Args:
        base: Base telemetry dict from simulate_normal_channel().
        duration_sec: Duration of the attack in seconds.
        start_sec: Start time (second index) of the attack.
        rng: NumPy random generator.

    Returns:
        Modified telemetry dict with MitM attack applied.
    """
    if rng is None:
        rng = np.random.default_rng()

    attacked = {k: v.copy() for k, v in base.items()}
    end_sec = min(start_sec + duration_sec, len(base['qber']))
    sl = slice(start_sec, end_sec)
    n = end_sec - start_sec

    logger.info(
        "MitM attack: t=%d–%d (%ds)",
        start_sec, end_sec, n,
    )

    # Full S collapse to classical bound
    attacked['bell_S'][sl] = np.clip(
        2.0 + rng.normal(0, 0.05, n), 2.0, 2.3
    )

    # Large QBER (separable states → high error rate)
    attacked['qber'][sl] = np.clip(
        0.10 + rng.normal(0, 0.02, n), 0.06, 0.25
    )

    # Severe coincidence drop (no genuine SPDC source)
    attacked['coincidence_rate'][sl] = np.round(
        attacked['coincidence_rate'][sl] * 0.40 + rng.normal(0, 30, n)
    ).astype(int)
    attacked['coincidence_rate'][sl] = np.clip(
        attacked['coincidence_rate'][sl], 0, None
    )

    attacked['label'][sl] = 1
    attacked['attack_type'][sl] = 3  # MitM
    return attacked


def attack_blended_subthreshold(
    base: Dict[str, np.ndarray],
    n_bursts: int,
    burst_duration: int = 30,
    eve_fraction: float = 0.12,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Blended Sub-Threshold attack: the operationally critical scenario.

    Eve operates in short, clustered bursts designed to keep total QBER below
    the classical 11% threshold while still extracting key material. She varies
    timing to exploit scintillation events (natural QBER spikes) as cover.

    Strategy:
        - Attack for 20–40 seconds at moderate intensity
        - Wait 200–600 seconds between bursts
        - 60% of bursts timed to coincide with high-QBER windows
          (scintillation/precipitation) for camouflage

    This is the primary target of the ML classifier.

    Args:
        base: Base telemetry dict from simulate_normal_channel().
        n_bursts: Number of attack bursts to inject.
        burst_duration: Duration of each burst in seconds.
        eve_fraction: Interception intensity per burst (lower than IR).
        rng: NumPy random generator.

    Returns:
        Modified telemetry dict with blended attack bursts applied.
    """
    if rng is None:
        rng = np.random.default_rng()

    attacked = {k: v.copy() for k, v in base.items()}
    n = len(base['qber'])

    # Identify high-QBER windows (scintillation events) as attack camouflage
    high_noise_windows = np.where(
        base['qber'] > np.percentile(base['qber'], 70)
    )[0]

    burst_count_applied = 0
    for burst_idx in range(n_bursts):
        if burst_idx == 0:
            # Guarantee at least 1 burst in the holdout period
            if n > 64800 + burst_duration:
                start = int(rng.integers(64800, n - burst_duration))
            else:
                start = int(rng.integers(0, max(1, n - burst_duration)))
        elif len(high_noise_windows) > 0 and rng.random() < 0.6:
            start = int(rng.choice(high_noise_windows))
        else:
            start = int(rng.integers(0, max(1, n - burst_duration)))

        end = min(start + burst_duration, n)
        sl = slice(start, end)
        bn = end - start

        # QBER increase: eve_fraction × 0.25 with ±20% variation
        delta = eve_fraction * 0.25 * rng.uniform(0.8, 1.2, bn)
        attacked['qber'][sl] = np.clip(
            attacked['qber'][sl] + delta, 0, 0.14
        )

        # Bell S: partial degradation (70% of full IR collapse)
        s_drop = eve_fraction * (2.828 - 2.0) * 0.7
        attacked['bell_S'][sl] = np.clip(
            attacked['bell_S'][sl] - s_drop + rng.normal(0, 0.02, bn),
            2.0, 2.828,
        )

        # Coincidence: small drop (~eve_fraction × 15%)
        attacked['coincidence_rate'][sl] = np.round(
            attacked['coincidence_rate'][sl] * (1.0 - eve_fraction * 0.15)
        ).astype(int)

        attacked['label'][sl] = 1
        attacked['attack_type'][sl] = 4  # Blended sub-threshold
        burst_count_applied += 1

    logger.info(
        "Blended sub-threshold attack: %d bursts of %ds at eve_fraction=%.2f",
        burst_count_applied, burst_duration, eve_fraction,
    )

    return attacked
