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
from typing import Dict, Optional, List
import numpy as np
import pandas as pd

from core.config import PHYSICS_CONFIG, DetectorMode
from core.attacks import AttackStrategy

logger = logging.getLogger(__name__)


def generate_lognormal_fading(n_seconds: int, sigma_I2: float, seed: int) -> np.ndarray:
    """Generates log-normal atmospheric transmittance fading."""
    rng = np.random.default_rng(seed)
    mu = -0.5 * sigma_I2
    # X ~ Normal(mu, sigma_I)
    X = rng.normal(mu, np.sqrt(sigma_I2), size=n_seconds)
    return np.exp(X)


def scintillation_model(n_seconds: int, sigma_I: float = 0.3, seed: int = 42) -> np.ndarray:
    """
    Models log-normal irradiance fluctuations due to atmospheric turbulence.

    I ~ LogNormal(mu, sigma_I^2), scintillation index SI = sigma_I^2.
    Weak turbulence regime: sigma_I in [0.1, 0.4].

    Returns: normalised photon loss fraction per second, shape (n_seconds,),
             values in [0, 0.95].
    """
    sigma_I2 = sigma_I ** 2
    transmittance = generate_lognormal_fading(n_seconds, sigma_I2, seed)
    # Convert transmittance to loss fraction, clipped to [0, 0.95]
    loss_fraction = np.clip(1.0 - transmittance / (transmittance.max() + 1e-9), 0.0, 0.95)
    return loss_fraction


def thermal_gradient_model(n_seconds: int, day_start_sec: int = 0) -> np.ndarray:
    """
    Models QBER contribution from ground-level thermal gradients.
    Peak at solar noon (~14h due to heating lag), minimum at pre-dawn (~4h).

    Returns: QBER additive component, shape (n_seconds,), values in [0, ~0.016].
    """
    t = np.arange(n_seconds) + day_start_sec
    hour_of_day = (t / 3600.0) % 24.0
    # Sinusoidal model: peak at 14h (solar heating lag), trough at 2h
    thermal_qber = 0.008 * (0.5 + 0.5 * np.sin(2 * np.pi * (hour_of_day - 8.0) / 24.0))
    return thermal_qber


def precipitation_model(
    n_seconds: int,
    p_event: float = 0.0003,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Poisson-arrival precipitation bursts returning attenuation in dB."""
    if rng is None:
        rng = np.random.default_rng()
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


def simulate_normal_channel(
    n_seconds: int = 86400, 
    seed: int = 42,
    attack_strategies: Optional[List[AttackStrategy]] = None
) -> Dict[str, np.ndarray]:
    """
    Generates a full 24-hour BBM92 telemetry stream from first principles.
    
    Pipeline:
    Entangled pair generation → channel propagation → detection probabilities → 
    singles → true coincidences → accidental coincidences → sampled counters.
    """
    logger.info("Simulating first-principles BBM92 channel: %d seconds, seed=%d", n_seconds, seed)
    rng = np.random.default_rng(seed)
    cfg = PHYSICS_CONFIG

    # 1. Base Physical State Variables (Arrays for n_seconds)
    R_pair = np.full(n_seconds, cfg.source_pair_rate)
    V_source = np.full(n_seconds, cfg.source_visibility)
    eta_atm_A = np.full(n_seconds, cfg.eta_atm_alice_base)
    
    eta_scint_B = generate_lognormal_fading(n_seconds, cfg.scintillation_variance, seed)
    precip_loss_dB = precipitation_model(n_seconds, cfg.precipitation_prob, rng)
    eta_precip_B = 10.0 ** (-precip_loss_dB / 10.0)
    eta_atm_B = cfg.eta_atm_bob_base * eta_scint_B * eta_precip_B
    
    detector_mode_B = np.full(n_seconds, DetectorMode.GEIGER.value)
    background_B = np.full(n_seconds, getattr(cfg, 'detector_background_counts', 0.0))

    deterministic_match_rate = np.zeros(n_seconds)
    deterministic_mismatch_rate = np.zeros(n_seconds)
    deterministic_chsh_pos = np.zeros(n_seconds)
    deterministic_chsh_neg = np.zeros(n_seconds)
    
    attack_label = np.zeros(n_seconds, dtype=int)
    attack_type = np.zeros(n_seconds, dtype=int)
    final_channel_loss_dB: np.ndarray | None = None

    # 2. Apply Attack Strategies (Eve manipulates physics)
    if attack_strategies:
        for strategy in attack_strategies:
            attack_mods = strategy.apply(
                n_seconds, V_source, eta_atm_B, detector_mode_B, background_B, rng
            )
            if attack_mods:
                V_source = attack_mods['V']
                eta_atm_B = attack_mods['eta_atm_B']
                detector_mode_B = attack_mods['detector_mode_B']
                background_B = attack_mods['background_B']
                # Add deterministic rates from all strategies
                deterministic_match_rate += attack_mods['deterministic_match_rate']
                deterministic_mismatch_rate += attack_mods['deterministic_mismatch_rate']
                deterministic_chsh_pos += attack_mods['deterministic_chsh_pos']
                deterministic_chsh_neg += attack_mods['deterministic_chsh_neg']
                
                # Apply classical spoofing override if present
                spoofed_loss = attack_mods.get('spoofed_channel_loss_dB')
                sl = strategy._get_slice(n_seconds)
                if spoofed_loss is not None:
                    # Initialize the array if this is the first spoof
                    if final_channel_loss_dB is None:
                        final_channel_loss_dB = -10.0 * np.log10(np.clip(eta_atm_B * cfg.eta_sys_bob, 1e-10, 1.0))
                    try:
                        final_channel_loss_dB[sl] = spoofed_loss
                    except ValueError as e:
                        print(f"CRASH! sl={sl}, spoofed_loss.shape={np.shape(spoofed_loss)}, final_channel_loss_dB.shape={np.shape(final_channel_loss_dB)}")
                        raise e
                
                attack_label[sl] = 1
                attack_type[sl] = strategy.attack_type_id

    # 3. Detection probabilities
    eta_total_A = eta_atm_A * cfg.eta_sys_alice
    eta_total_B = eta_atm_B * cfg.eta_sys_bob
    
    P_det_A = eta_total_A * cfg.detector_efficiency
    P_det_B = eta_total_B * cfg.detector_efficiency
    
    # If Bob is in linear mode, his single-photon detection drops to 0
    is_linear = (detector_mode_B == DetectorMode.LINEAR.value)
    P_det_B[is_linear] = 0.0

    # 4. Singles
    R_A = R_pair * P_det_A + cfg.detector_dark_counts + getattr(cfg, 'detector_background_counts', 0.0)
    
    # In linear mode, Bob's dark counts vanish (they don't cross discriminator)
    DCR_B = np.where(is_linear, 0.0, cfg.detector_dark_counts)
    R_B = R_pair * P_det_B + DCR_B + background_B

    # 5. True and Accidental Coincidences (Expected Rates)
    R_true_c = R_pair * P_det_A * P_det_B
    R_acc_c = 1.0 * R_A * R_B * cfg.coincidence_window
    
    # Total expected quantum coincidence rate
    R_c_expected = R_true_c + R_acc_c
    
    # Base classical telemetry (if not spoofed)
    if final_channel_loss_dB is None:
        final_channel_loss_dB = -10.0 * np.log10(np.clip(eta_atm_B * cfg.eta_sys_bob, 1e-10, 1.0))
    channel_loss_dB = final_channel_loss_dB

    # 6. Sampled Counters (Empirical Measurements)
    # 16 total basis combinations, chosen uniformly.
    
    # Key Generation Counters
    E_true_key = (R_true_c / 16.0)
    E_acc = (R_acc_c / 16.0)
    
    lambda_key_match = (E_true_key * (1 + V_source) / 4.0) + (E_acc / 4.0) + (deterministic_match_rate / 4.0)
    lambda_key_mismatch = (E_true_key * (1 - V_source) / 4.0) + (E_acc / 4.0) + (deterministic_mismatch_rate / 4.0)
    
    C_key_match = rng.poisson(lambda_key_match, size=(4, n_seconds))
    C_key_mismatch = rng.poisson(lambda_key_mismatch, size=(4, n_seconds))
    
    total_key_matches = np.sum(C_key_match, axis=0)
    total_key_mismatches = np.sum(C_key_mismatch, axis=0)
    total_key_events = total_key_matches + total_key_mismatches

    qber = total_key_mismatches / np.maximum(1, total_key_events)

    # Bell S Counters
    V_CHSH = V_source / np.sqrt(2)
    
    lambda_chsh_pos_match = (E_true_key * (1 + V_CHSH) / 4.0) + (E_acc / 4.0) + (deterministic_chsh_pos / 6.0)
    lambda_chsh_pos_mismatch = (E_true_key * (1 - V_CHSH) / 4.0) + (E_acc / 4.0)
    
    lambda_chsh_neg_match = (E_true_key * (1 - V_CHSH) / 4.0) + (E_acc / 4.0) + (deterministic_chsh_neg / 2.0)
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
    # The deterministic rates are already total sums for their respective basis groups.
    det_total = deterministic_match_rate + deterministic_chsh_pos + deterministic_chsh_neg
    coincidence_rate = rng.poisson(R_c_expected + det_total)
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
        'label': attack_label,
        'attack_type': attack_type,
    }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    res = simulate_normal_channel(n_seconds=10)
    for k, v in res.items():
        print(f"{k}: {v[:5]}")
