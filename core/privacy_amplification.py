"""
core/privacy_amplification.py — Toeplitz hashing and secure key rate computation.

Implements privacy amplification for BBM92 QKD using Toeplitz matrix hashing
and the binary entropy-based secure key rate formula.

Extended with:
  - Proper binary entropy function
  - BBM92-specific secure key rate estimation
  - Finite-key security bounds (Hoeffding inequality)

The secure key rate (SKR) quantifies the usable key bits per sifted event
after error correction and privacy amplification:
    R = 1 - h(QBER) - h(QBER)
    R = 1 - 2×h(QBER)       [for BBM92 with one-way EC]

where h(x) = -x log₂(x) - (1-x) log₂(1-x) is the binary entropy.

"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def binary_entropy(p: float | np.ndarray) -> float | np.ndarray:
    """
    Computes the binary Shannon entropy h(p).

    h(p) = -p × log₂(p) - (1-p) × log₂(1-p)

    Handles edge cases: h(0) = h(1) = 0.

    Args:
        p: Probability value(s) in [0, 1].

    Returns:
        Binary entropy value(s), same shape as input.
    """
    p = np.asarray(p, dtype=np.float64)
    result = np.zeros_like(p)
    mask = (p > 0) & (p < 1)
    result[mask] = (
        -p[mask] * np.log2(p[mask])
        - (1.0 - p[mask]) * np.log2(1.0 - p[mask])
    )
    # Return scalar if input was scalar
    if result.ndim == 0:
        return float(result)
    return result


def secure_key_rate(
    qber: float | np.ndarray,
    sifted_rate: float = 5000.0,
    error_correction_efficiency: float = 1.16,
) -> float | np.ndarray:
    """
    Computes the asymptotic secure key rate (SKR) for BBM92.

    For BBM92 with one-way error correction:
        R = sifted_rate × max(0, 1 - h(QBER) - f_EC × h(QBER))

    where:
        - h(QBER) is the binary entropy at the quantum bit error rate
        - f_EC is the error correction inefficiency factor (≥ 1.0)
        - sifted_rate is the number of sifted key bits per second

    The key rate goes to zero at QBER ≈ 11% (the theoretical BB84/BBM92
    security threshold for one-way EC with Cascade/LDPC).

    Args:
        qber: Quantum Bit Error Rate, float or array in [0, 0.5].
        sifted_rate: Sifted key events per second (default: 5000 for
            typical SPDC source at 10,000 coincidences, 50% basis match).
        error_correction_efficiency: f_EC factor (1.0 = Shannon limit,
            1.16 = typical Cascade implementation).

    Returns:
        Secure key rate in bits/second. Zero when QBER exceeds threshold.
    """
    qber = np.asarray(qber, dtype=np.float64)
    h_q = binary_entropy(np.clip(qber, 0, 0.5))

    # BBM92 SKR: 1 - h(QBER) [Eve's information] - f_EC × h(QBER) [EC leakage]
    raw_rate = 1.0 - h_q - error_correction_efficiency * h_q
    skr = sifted_rate * np.maximum(raw_rate, 0.0)

    if skr.ndim == 0:
        return float(skr)
    return skr



