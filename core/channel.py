"""
core/channel.py — FSO+fibre BBM92 physical channel simulator.

Implements the complete atmospheric channel model for BBM92 entanglement-based QKD:
  - Log-normal scintillation (Rytov variance model)
  - Diurnal thermal gradient cycle
  - Poisson-arrival precipitation burst noise
  - Master normal-conditions simulator producing 24-hour telemetry

All operations are fully vectorised NumPy. No Python-level per-second loops
(except precipitation model which uses Poisson inter-arrival sampling).

Physics reference:
  BBM92 uses SPDC to produce |Φ⁺⟩ = (1/√2)(|HH⟩ + |VV⟩).
  QBER baseline is 0% for ideal entanglement; real channels: 1–5%.
  Bell S parameter: S = 2√2 × cos(2 × arcsin(√QBER)) for the |Φ⁺⟩ state.

"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def scintillation_model(
    n_seconds: int,
    sigma_I: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """
    Models log-normal irradiance fluctuations due to atmospheric turbulence.

    The irradiance I follows a log-normal distribution:
        I ~ LogNormal(mu, sigma_I^2)
    where mu = -0.5 * sigma_I^2 ensures E[I] = 1.

    The scintillation index SI = sigma_I^2 characterises turbulence strength:
        - Weak turbulence:   SI ∈ [0.1, 0.4]
        - Moderate:          SI ∈ [0.4, 1.0]
        - Strong:            SI > 1.0

    Args:
        n_seconds: Number of 1-second time steps to simulate.
        sigma_I: Scintillation parameter (√SI). Default 0.3 (weak turbulence).
        seed: RNG seed for reproducibility.

    Returns:
        Normalised photon loss fraction per second, shape (n_seconds,),
        values clipped to [0.0, 0.95].
    """
    rng = np.random.default_rng(seed)
    mu = -0.5 * sigma_I**2  # ensures E[I] = 1
    log_I = rng.normal(mu, sigma_I, size=n_seconds)
    I = np.exp(log_I)
    # Convert irradiance to photon loss
    loss_fraction = np.clip(1.0 - I, 0.0, 0.95)
    return loss_fraction


def thermal_gradient_model(
    n_seconds: int,
    day_start_sec: int = 0,
) -> np.ndarray:
    """
    Models QBER contribution from ground-level thermal gradients.

    Uses a sinusoidal model with:
        - Peak at 14h (solar heating lag after solar noon at 12h)
        - Trough at 4h (pre-dawn minimum)
        - Amplitude: 0–0.016 QBER contribution

    Args:
        n_seconds: Number of 1-second time steps.
        day_start_sec: Starting second offset within the day.

    Returns:
        QBER additive component, shape (n_seconds,), range [0, ~0.016].
    """
    t = np.arange(n_seconds) + day_start_sec
    hour_of_day = (t / 3600.0) % 24.0
    # Sinusoidal model: peak at 14h (solar heating lag), trough at 4h
    thermal_qber = 0.008 * (0.5 + 0.5 * np.sin(
        2 * np.pi * (hour_of_day - 4.0) / 24.0
    ))
    return thermal_qber


def precipitation_model(
    n_seconds: int,
    p_event: float = 0.0003,
    duration_range: Tuple[int, int] = (60, 600),
    intensity: float = 0.04,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Poisson-arrival precipitation bursts with Gaussian-envelope profiles.

    Each burst event:
        1. Inter-arrival time ~ Exponential(1/p_event)
        2. Duration ~ Uniform(duration_range)
        3. Envelope: sinusoidal ramp-up/ramp-down

    Args:
        n_seconds: Number of 1-second time steps.
        p_event: Per-second probability of a precipitation event starting.
        duration_range: (min_duration, max_duration) in seconds.
        intensity: Peak QBER contribution during a burst.
        rng: NumPy random generator. Created if None.

    Returns:
        QBER additive component (burst), shape (n_seconds,), clipped to [0, 0.15].
    """
    if rng is None:
        rng = np.random.default_rng()

    qber_contribution = np.zeros(n_seconds)
    t = 0
    while t < n_seconds:
        inter_arrival = rng.exponential(1.0 / (p_event + 1e-10))
        t += int(inter_arrival)
        if t >= n_seconds:
            break
        dur = int(rng.uniform(*duration_range))
        end = min(t + dur, n_seconds)
        # Sinusoidal envelope: ramp up and down
        burst_t = np.linspace(0, 1, end - t)
        envelope = intensity * np.sin(np.pi * burst_t)
        qber_contribution[t:end] += envelope
        t = end

    return np.clip(qber_contribution, 0, 0.15)


def simulate_normal_channel(
    n_seconds: int = 86400,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generates a full 24-hour BBM92 telemetry stream under normal atmospheric conditions.

    All operations are vectorised. Combines:
        1. Baseline QBER from detector dark counts + optical misalignment
        2. Scintillation-induced photon loss
        3. Diurnal thermal gradient contribution
        4. Precipitation burst noise

    The Bell S parameter is computed from the exact BBM92 relationship:
        S = 2√2 × cos(2 × arcsin(√QBER))

    Coincidence rate is derived from channel loss (Beer-Lambert model).
    Visibility tracks entanglement quality: V ≈ 1 - 2×QBER.

    Args:
        n_seconds: Duration in seconds (default: 86,400 = 24 hours).
        seed: RNG seed for full reproducibility.

    Returns:
        Dictionary of NumPy arrays, each shape (n_seconds,):
            qber:             Quantum Bit Error Rate (float, 0–1)
            bell_S:           CHSH S parameter (float, 2.0–2.828)
            coincidence_rate: Detected photon pairs per second (int)
            visibility:       Hong-Ou-Mandel visibility (float, 0–1)
            channel_loss_dB:  Total optical path loss (float, dB)
            detection_rate:   Single-detector click rate (float)
            label:            0 (normal) for all timesteps
    """
    logger.info(
        "Simulating normal BBM92 channel: %d seconds, seed=%d",
        n_seconds, seed,
    )
    rng = np.random.default_rng(seed)

    # Base QBER = detector dark counts + optical misalignment + fibre birefringence
    qber_base = 0.02 + rng.normal(0, 0.003, n_seconds)

    # Add atmospheric contributions (vectorised)
    scint_loss = scintillation_model(n_seconds, sigma_I=0.25, seed=seed)
    thermal_qber = thermal_gradient_model(n_seconds)
    precip_qber = precipitation_model(n_seconds, rng=rng)

    qber = np.clip(
        qber_base + 0.1 * scint_loss + thermal_qber + precip_qber,
        0.005, 0.12,
    )

    # Bell S degrades with QBER: S ≈ 2√2 × cos(2 × arcsin(√QBER))
    # This is the exact theoretical BBM92 relationship for the |Φ⁺⟩ state
    bell_S = 2.0 * np.sqrt(2) * np.cos(
        2.0 * np.arcsin(np.sqrt(np.clip(qber, 0, 0.499)))
    )
    bell_S = np.clip(bell_S + rng.normal(0, 0.02, n_seconds), 2.0, 2.828)

    # Coincidence rate: baseline 10,000/s, degraded by channel loss
    channel_loss_dB = 3.0 + 20.0 * scint_loss + rng.normal(0, 0.5, n_seconds)
    channel_loss_linear = 10.0 ** (-channel_loss_dB / 10.0)
    coincidence_rate = np.round(
        10000 * channel_loss_linear + rng.normal(0, 50, n_seconds)
    ).astype(int)
    coincidence_rate = np.clip(coincidence_rate, 0, 12000)

    # Visibility: related to entanglement quality
    visibility = np.clip(
        1.0 - 2.0 * qber + rng.normal(0, 0.01, n_seconds),
        0.7, 1.0,
    )

    # Single detector rate (dark counts + real events)
    detection_rate = 2.0 * coincidence_rate + rng.poisson(200, n_seconds)

    logger.info(
        "Normal channel generated: mean QBER=%.4f, mean S=%.3f, mean coincidence=%d",
        float(np.mean(qber)),
        float(np.mean(bell_S)),
        int(np.mean(coincidence_rate)),
    )

    return {
        'qber': qber,
        'bell_S': bell_S,
        'coincidence_rate': coincidence_rate,
        'visibility': visibility,
        'channel_loss_dB': channel_loss_dB,
        'detection_rate': detection_rate.astype(float),
        'label': np.zeros(n_seconds, dtype=int),
        'attack_type': np.zeros(n_seconds, dtype=int),
    }
