"""
core/channel.py — First-Principles BBM92 Physical Simulator.

Implements the complete process-driven simulator:
  1. Entangled pair generation
  2. Channel propagation (Log-normal fading, precipitation)
  3. Detection probabilities
  4. Singles rates (Signal + Dark Counts)
  5. True and Accidental coincidences
  6. Sampled empirical measurement counters (Poisson distributed)
  7. Derived observables (QBER, Bell S, SKR)
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from core.config import PHYSICS_CONFIG

logger = logging.getLogger(__name__)


def generate_lognormal_fading(n_seconds: int, sigma_I2: float, seed: int) -> np.ndarray:
    """Generates log-normal atmospheric transmittance fading."""
    rng = np.random.default_rng(seed)
    mu = -0.5 * sigma_I2
    # X ~ Normal(mu, sigma_I)
    X = rng.normal(mu, np.sqrt(sigma_I2), size=n_seconds)
    return np.exp(X)


def precipitation_model(n_seconds: int, p_event: float, rng: np.random.Generator) -> np.ndarray:
    """Poisson-arrival precipitation bursts returning attenuation in dB."""
    loss_dB = np.zeros(n_seconds)
    t = 0
    while t < n_seconds:
        inter_arrival = rng.exponential(1.0 / (p_event + 1e-10))
        t += int(inter_arrival)
        if t >= n_seconds:
            break
        dur = int(rng.uniform(60, 600))
        end = min(t + dur, n_seconds)
        
        burst_t = np.linspace(0, 1, end - t)
        # Intensity between 1.0 and 5.0 dB of loss
        intensity_dB = rng.uniform(1.0, 5.0)
        envelope = intensity_dB * np.sin(np.pi * burst_t)
        loss_dB[t:end] += envelope
        t = end
        
    return loss_dB


def simulate_normal_channel(n_seconds: int = 86400, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Generates a full 24-hour BBM92 telemetry stream from first principles.
    
    Pipeline:
    Entangled pair generation → channel propagation → detection probabilities → 
    singles → true coincidences → accidental coincidences → sampled counters.
    """
    logger.info("Simulating first-principles benign BBM92 channel: %d seconds, seed=%d", n_seconds, seed)
    rng = np.random.default_rng(seed)
    cfg = PHYSICS_CONFIG

    # 1. Entangled pair generation (pairs/sec)
    R_pair = np.full(n_seconds, cfg.source_pair_rate)
    V = cfg.source_visibility

    # 2. Channel propagation
    # Alice is colocated with the source
    eta_atm_A = np.full(n_seconds, cfg.eta_atm_alice_base)
    
    # Bob experiences log-normal fading and precipitation
    eta_scint_B = generate_lognormal_fading(n_seconds, cfg.scintillation_variance, seed)
    precip_loss_dB = precipitation_model(n_seconds, cfg.precipitation_prob, rng)
    eta_precip_B = 10.0 ** (-precip_loss_dB / 10.0)
    
    eta_atm_B = cfg.eta_atm_bob_base * eta_scint_B * eta_precip_B

    # Total end-to-end transmittances
    eta_total_A = eta_atm_A * cfg.eta_sys_alice
    eta_total_B = eta_atm_B * cfg.eta_sys_bob

    # 3. Detection probabilities
    P_det_A = eta_total_A * cfg.detector_efficiency
    P_det_B = eta_total_B * cfg.detector_efficiency

    # 4. Singles
    # Alice is local, typically dark counts only, but may have local stray light
    R_A = R_pair * P_det_A + cfg.detector_dark_counts + getattr(cfg, 'detector_background_counts', 0.0)
    # Bob is remote, experiences dark counts + background stray light (e.g. 50kcps)
    R_B = R_pair * P_det_B + cfg.detector_dark_counts + getattr(cfg, 'detector_background_counts', 0.0)

    # 5. True and Accidental Coincidences (Expected Rates)
    R_true_c = R_pair * P_det_A * P_det_B
    R_acc_c = R_A * R_B * cfg.coincidence_window
    
    # Total coincidence rate expected
    R_c_expected = R_true_c + R_acc_c
    
    channel_loss_dB = -10.0 * np.log10(np.clip(eta_atm_B * cfg.eta_sys_bob, 1e-10, 1.0))

    # 6. Sampled Counters (Empirical Measurements)
    # We model a system that randomly alternates between key generation bases (Z, X) 
    # and Bell test bases (CHSH angles A1, A2, B1, B2).
    # Assume 16 total basis combinations, chosen uniformly.
    # Each combination gets 1/16 of the total rates.
    
    # Key Generation Counters (2 matching bases: ZZ, XX)
    # True coincidence probabilities: P(match) = (1+V)/2, P(mismatch) = (1-V)/2
    E_true_key = (R_true_c / 16.0)
    E_acc = (R_acc_c / 16.0)
    
    # We have 2 matching bases. 
    # Each basis has 2 matched outcomes (++, --) and 2 mismatched outcomes (+-, -+).
    # So 4 match counters and 4 mismatch counters for key generation.
    lambda_key_match = (E_true_key * (1 + V) / 4.0) + (E_acc / 4.0)
    lambda_key_mismatch = (E_true_key * (1 - V) / 4.0) + (E_acc / 4.0)
    
    # Sample the empirical counters
    C_key_match = rng.poisson(lambda_key_match, size=(4, n_seconds))
    C_key_mismatch = rng.poisson(lambda_key_mismatch, size=(4, n_seconds))
    
    total_key_matches = np.sum(C_key_match, axis=0)
    total_key_mismatches = np.sum(C_key_mismatch, axis=0)
    total_key_events = total_key_matches + total_key_mismatches

    # QBER is purely empirical
    qber = total_key_mismatches / np.maximum(1, total_key_events)

    # Bell S Counters (4 CHSH bases: A1B1, A1B2, A2B1, A2B2)
    # For CHSH, 3 correlators have E = V / sqrt(2), 1 has E = -V / sqrt(2)
    V_CHSH = V / np.sqrt(2)
    
    lambda_chsh_pos_match = (E_true_key * (1 + V_CHSH) / 4.0) + (E_acc / 4.0)
    lambda_chsh_pos_mismatch = (E_true_key * (1 - V_CHSH) / 4.0) + (E_acc / 4.0)
    
    lambda_chsh_neg_match = (E_true_key * (1 - V_CHSH) / 4.0) + (E_acc / 4.0)
    lambda_chsh_neg_mismatch = (E_true_key * (1 + V_CHSH) / 4.0) + (E_acc / 4.0)

    # Sample the 4 CHSH bases (each has 4 counters: ++, --, +-, -+)
    # Base 1, 2, 3 (Positive correlators)
    C_chsh_pos_match = rng.poisson(lambda_chsh_pos_match, size=(3, 2, n_seconds)) # 3 bases, 2 match outcomes
    C_chsh_pos_mismatch = rng.poisson(lambda_chsh_pos_mismatch, size=(3, 2, n_seconds)) # 3 bases, 2 mismatch outcomes
    
    # Base 4 (Negative correlator)
    C_chsh_neg_match = rng.poisson(lambda_chsh_neg_match, size=(1, 2, n_seconds))
    C_chsh_neg_mismatch = rng.poisson(lambda_chsh_neg_mismatch, size=(1, 2, n_seconds))
    
    # Compute the empirical correlators E(x,y)
    def compute_E(match_arr, mismatch_arr):
        # match_arr shape: (n_bases, 2, n_seconds)
        match_sum = np.sum(match_arr, axis=1)
        mismatch_sum = np.sum(mismatch_arr, axis=1)
        total = match_sum + mismatch_sum
        return (match_sum - mismatch_sum) / np.maximum(1, total)

    E_pos = compute_E(C_chsh_pos_match, C_chsh_pos_mismatch) # shape (3, n_seconds)
    E_neg = compute_E(C_chsh_neg_match, C_chsh_neg_mismatch) # shape (1, n_seconds)

    # S = |E1 + E2 + E3 - E4| (standard formulation grouping)
    bell_S = np.abs(E_pos[0] + E_pos[1] + E_pos[2] - E_neg[0])
    
    # 7. Additional Observables
    # Total coincidences (scaled up from the 16 bases to represent 100% operation for standard metrics)
    coincidence_rate = rng.poisson(R_c_expected)
    detection_rate = rng.poisson(R_A + R_B)
    
    # Visibility (empirical from key bases)
    visibility = (total_key_matches - total_key_mismatches) / np.maximum(1, total_key_events)
    visibility = np.clip(visibility, 0, 1.0)
    
    # Smooth out anomalies in extreme low-count periods to avoid math warnings
    mask_low_counts = total_key_events < 5
    qber[mask_low_counts] = 0.5
    bell_S[mask_low_counts] = 0.0
    visibility[mask_low_counts] = 0.0

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

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    res = simulate_normal_channel(n_seconds=10)
    for k, v in res.items():
        print(f"{k}: {v[:5]}")
