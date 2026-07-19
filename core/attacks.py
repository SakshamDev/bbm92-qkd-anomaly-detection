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

import logging
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from core.config import EveCapabilities, DetectorMode

logger = logging.getLogger(__name__)

class AttackStrategy(ABC):
    """Abstract base class for all causal physics attacks."""
    def __init__(self, caps: EveCapabilities, start_sec: int, duration_sec: int):
        self.caps = caps
        self.start_sec = start_sec
        self.end_sec = start_sec + duration_sec
        
    def _get_slice(self, n_seconds: int) -> slice:
        end = min(self.end_sec, n_seconds)
        return slice(self.start_sec, end)

    @abstractmethod
    def apply(
        self, 
        n_seconds: int,
        V: np.ndarray, 
        eta_atm_B: np.ndarray, 
        detector_mode_B: np.ndarray, 
        background_B: np.ndarray,
        rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        """
        Applies Eve's physical manipulations.
        
        Returns a dictionary of modified physical arrays and deterministic rates:
        - 'V': modified visibility
        - 'eta_atm_B': modified base transmittance
        - 'detector_mode_B': array of DetectorMode
        - 'background_B': background stray light
        - 'deterministic_match_rate': classical trigger matches (cps)
        - 'deterministic_mismatch_rate': classical trigger mismatches (cps)
        """
        pass
    
    @property
    @abstractmethod
    def attack_type_id(self) -> int:
        pass


class InterceptResendStrategy(AttackStrategy):
    """
    Modulated Intercept-Resend Attack (IRA).
    Eve intercepts a fraction of photons, measures them, and resends.
    """
    def __init__(self, caps: EveCapabilities, start_sec: int, duration_sec: int, fraction: float = 0.5):
        super().__init__(caps, start_sec, duration_sec)
        self.fraction = fraction

    @property
    def attack_type_id(self) -> int:
        return 1

    def apply(
        self, 
        n_seconds: int,
        V: np.ndarray, 
        eta_atm_B: np.ndarray, 
        detector_mode_B: np.ndarray, 
        background_B: np.ndarray,
        rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        
        sl = self._get_slice(n_seconds)
        if sl.stop <= sl.start:
            return {} # Out of bounds
            
        n_attack = sl.stop - sl.start
        
        V_mod = V.copy()
        eta_mod = eta_atm_B.copy()
        
        # 1. State Collapse: Visibility drops for the intercepted fraction
        V_mod[sl] = V[sl] * (1.0 - self.fraction)
        
        # 2. Channel Substitution / Attenuation Policy
        if self.caps.attenuation_policy:
            eta_mod[sl] = self.caps.attenuation_policy.get_quantum_transmittance(eta_atm_B, sl, self.fraction, rng)
        else:
            # Fallback natural channel + efficiency drops
            coupling_efficiency = self.caps.interception_efficiency * self.caps.detection_efficiency
            eta_mod[sl] = eta_atm_B[sl] * (1.0 - self.fraction + self.fraction * coupling_efficiency)

        # 3. Beacon Spoofing Policy
        spoofed_channel_loss = None
        if self.caps.beacon_spoofing_policy:
            from core.config import PHYSICS_CONFIG
            spoofed_transmittance = self.caps.beacon_spoofing_policy.get_classical_transmittance(eta_atm_B, sl, rng)
            spoofed_channel_loss = -10.0 * np.log10(np.clip(spoofed_transmittance * PHYSICS_CONFIG.eta_sys_bob, 1e-10, 1.0))

        return {
            'V': V_mod,
            'eta_atm_B': eta_mod,
            'detector_mode_B': detector_mode_B,
            'background_B': background_B,
            'spoofed_channel_loss_dB': spoofed_channel_loss,
            'deterministic_match_rate': np.zeros(n_seconds),
            'deterministic_mismatch_rate': np.zeros(n_seconds),
            'deterministic_chsh_pos': np.zeros(n_seconds),
            'deterministic_chsh_neg': np.zeros(n_seconds),
        }


class FakedStateStrategy(AttackStrategy):
    """
    Detector Blinding and Faked-State Attack.
    Eve blinds Bob's APD into linear mode, then sends bright classical pulses.
    """
    def __init__(self, caps: EveCapabilities, start_sec: int, duration_sec: int, trigger_rate: float = 10000.0):
        super().__init__(caps, start_sec, duration_sec)
        self.trigger_rate = trigger_rate # Rate at which Eve sends tailored classical pulses

    @property
    def attack_type_id(self) -> int:
        return 2

    def apply(
        self, 
        n_seconds: int,
        V: np.ndarray, 
        eta_atm_B: np.ndarray, 
        detector_mode_B: np.ndarray, 
        background_B: np.ndarray,
        rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        
        sl = self._get_slice(n_seconds)
        if sl.stop <= sl.start:
            return {}
            
        n_attack = sl.stop - sl.start
        
        mode_mod = detector_mode_B.copy()
        bg_mod = background_B.copy()
        det_match = np.zeros(n_seconds)
        det_mismatch = np.zeros(n_seconds)
        det_chsh_pos = np.zeros(n_seconds)
        det_chsh_neg = np.zeros(n_seconds)
        
        # 1. Blinding Phase: APD transitions to LINEAR mode.
        for i in range(sl.start, sl.stop):
            mode_mod[i] = DetectorMode.LINEAR.value
            
        # The CW laser contributes to background, causing anomalies in singles rate
        bg_mod[sl] += 1e7  
        
        # 2. Faked-State Phase: Deterministic Clicks
        if self.caps.detector_control_policy:
            expected_rate = 10000.0 # simplified expectation
            trigger_rate = self.caps.detector_control_policy.get_deterministic_click_rates(expected_rate, sl, rng)
        else:
            trigger_rate = self.trigger_rate
            
        det_match[sl] = trigger_rate
        det_mismatch[sl] = 0.0
        
        # For CHSH bases, Faked-State approximates classical correlator limit: S <= 2.
        classical_chsh_rate = trigger_rate 
        det_chsh_pos[sl] = classical_chsh_rate * 0.75 # bias towards positive
        det_chsh_neg[sl] = classical_chsh_rate * 0.25 # bias towards negative

        # 3. Beacon Spoofing Policy
        spoofed_channel_loss = None
        if self.caps.beacon_spoofing_policy:
            from core.config import PHYSICS_CONFIG
            spoofed_transmittance = self.caps.beacon_spoofing_policy.get_classical_transmittance(eta_atm_B, sl, rng)
            spoofed_channel_loss = -10.0 * np.log10(np.clip(spoofed_transmittance * PHYSICS_CONFIG.eta_sys_bob, 1e-10, 1.0))

        return {
            'V': V,  # Quantum state is irrelevant in linear mode
            'eta_atm_B': eta_atm_B,
            'detector_mode_B': mode_mod,
            'background_B': bg_mod,
            'deterministic_match_rate': det_match,
            'deterministic_mismatch_rate': det_mismatch,
            'deterministic_chsh_pos': det_chsh_pos,
            'deterministic_chsh_neg': det_chsh_neg,
        }

class MitMStrategy(AttackStrategy):
    """
    Man-in-the-Middle Attack.
    Eve completely intercepts the quantum channel and sends separable states to both Alice and Bob.
    """
    def __init__(self, caps: EveCapabilities, start_sec: int, duration_sec: int):
        super().__init__(caps, start_sec, duration_sec)

    @property
    def attack_type_id(self) -> int:
        return 3

    def apply(
        self, 
        n_seconds: int,
        V: np.ndarray, 
        eta_atm_B: np.ndarray, 
        detector_mode_B: np.ndarray, 
        background_B: np.ndarray,
        rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        
        sl = self._get_slice(n_seconds)
        if sl.stop <= sl.start:
            return {}
            
        V_mod = V.copy()
        eta_mod = eta_atm_B.copy()
        
        # 1. Complete Entanglement Collapse
        # Eve sends separable states, meaning visibility is effectively 0 for Bell inequalities.
        # But for QBER, separable states in mismatched bases produce high errors (~50%).
        # We model this by dropping visibility to 0.5, yielding 25% QBER
        V_mod[sl] = 0.5
        
        # 2. Channel Substitution / Attenuation Policy
        if self.caps.attenuation_policy:
            eta_mod[sl] = self.caps.attenuation_policy.get_quantum_transmittance(eta_atm_B, sl, 1.0, rng)
        else:
            eta_mod[sl] = eta_atm_B[sl] * self.caps.interception_efficiency * self.caps.detection_efficiency

        # 3. Beacon Spoofing Policy
        spoofed_channel_loss = None
        if self.caps.beacon_spoofing_policy:
            from core.config import PHYSICS_CONFIG
            spoofed_transmittance = self.caps.beacon_spoofing_policy.get_classical_transmittance(eta_atm_B, sl, rng)
            spoofed_channel_loss = -10.0 * np.log10(np.clip(spoofed_transmittance * PHYSICS_CONFIG.eta_sys_bob, 1e-10, 1.0))

        return {
            'V': V_mod,
            'eta_atm_B': eta_mod,
            'detector_mode_B': detector_mode_B,
            'background_B': background_B,
            'spoofed_channel_loss_dB': spoofed_channel_loss,
            'deterministic_match_rate': np.zeros(n_seconds),
            'deterministic_mismatch_rate': np.zeros(n_seconds),
            'deterministic_chsh_pos': np.zeros(n_seconds),
            'deterministic_chsh_neg': np.zeros(n_seconds),
        }


class BlendedSubthresholdStrategy(AttackStrategy):
    """
    Blended Sub-Threshold attack.
    Eve operates in short bursts (timed to coincide with scintillation spikes) 
    to extract key without tripping the 11% QBER alarm.
    """
    def __init__(self, caps: EveCapabilities, start_sec: int, duration_sec: int, fraction: float = 0.12):
        super().__init__(caps, start_sec, duration_sec)
        self.fraction = fraction

    @property
    def attack_type_id(self) -> int:
        return 4

    def apply(
        self, 
        n_seconds: int,
        V: np.ndarray, 
        eta_atm_B: np.ndarray, 
        detector_mode_B: np.ndarray, 
        background_B: np.ndarray,
        rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        
        sl = self._get_slice(n_seconds)
        if sl.stop <= sl.start:
            return {}
            
        V_mod = V.copy()
        eta_mod = eta_atm_B.copy()
        
        # 1. State Collapse: Visibility drops for the intercepted fraction
        V_mod[sl] = V[sl] * (1.0 - self.fraction)
        
        # 2. Channel Substitution / Attenuation Policy
        if self.caps.attenuation_policy:
            eta_mod[sl] = self.caps.attenuation_policy.get_quantum_transmittance(eta_atm_B, sl, self.fraction, rng)
        else:
            eta_mod[sl] = eta_atm_B[sl] * (1.0 - self.fraction + self.fraction * self.caps.interception_efficiency * self.caps.detection_efficiency)

        # 3. Beacon Spoofing Policy
        spoofed_channel_loss = None
        if self.caps.beacon_spoofing_policy:
            from core.config import PHYSICS_CONFIG
            spoofed_transmittance = self.caps.beacon_spoofing_policy.get_classical_transmittance(eta_atm_B, sl, rng)
            spoofed_channel_loss = -10.0 * np.log10(np.clip(spoofed_transmittance * PHYSICS_CONFIG.eta_sys_bob, 1e-10, 1.0))

        return {
            'V': V_mod,
            'eta_atm_B': eta_mod,
            'detector_mode_B': detector_mode_B,
            'background_B': background_B,
            'spoofed_channel_loss_dB': spoofed_channel_loss,
            'deterministic_match_rate': np.zeros(n_seconds),
            'deterministic_mismatch_rate': np.zeros(n_seconds),
            'deterministic_chsh_pos': np.zeros(n_seconds),
            'deterministic_chsh_neg': np.zeros(n_seconds),
        }




