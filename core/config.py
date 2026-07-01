"""
core/config.py — Physical and engineering parameters for the BBM92 FSO simulator.

All parameters are strictly categorized as:
- [PHYSICS]: Derived from physical principles.
- [LITERATURE]: Backed by experimental published literature.
- [HARDWARE]: Taken from commercial component specifications.
- [ENGINEERING]: Reasonable choices for system deployment.
- [SIMULATOR]: Structural choices for generating the dataset.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class PhysicsConfig:
    # --- Source Parameters ---
    # [LITERATURE/ENGINEERING] Yin et al. 2012. Typical bright SPDC pair generation rate.
    source_pair_rate: float = 2e6  # pairs per second
    
    # [ENGINEERING] Realistic source intrinsic visibility including field optical misalignment (Erven 2012).
    # V=0.95 corresponds to a base QBER of 2.5%.
    source_visibility: float = 0.95
    
    # --- Detectors ---
    # [HARDWARE] Typical Si-APD quantum efficiency at 800nm (e.g., Excelitas SPCM).
    detector_efficiency: float = 0.50
    
    # [HARDWARE] Typical Si-APD dark count rate (internal thermal noise).
    detector_dark_counts: float = 200.0  # counts per second

    # [ENGINEERING] Background stray light entering telescopes (e.g. urban FSO link).
    detector_background_counts: float = 50000.0  # counts per second
    
    # [HARDWARE] Coincidence resolving time window (standard timestamping resolution).
    coincidence_window: float = 2e-9  # 2 nanoseconds
    
    # --- Optics & Path ---
    # [ENGINEERING] Alice is colocated with the source. Transmittance = 1.0 (0 dB).
    eta_atm_alice_base: float = 1.0
    
    # [ENGINEERING] Base transmittance for Bob's free-space path (e.g. geometric beam spread over 1km).
    # 10 dB geometric loss = 0.1 transmittance
    eta_atm_bob_base: float = 0.10
    
    # [ENGINEERING] Fixed system losses (coupling, filtering, geometric).
    # -5 dB loss = 10^(-5/10) = 0.316. Applied symmetrically here for lenses/fiber coupling.
    eta_sys_alice: float = 0.316
    eta_sys_bob: float = 0.316
    
    # --- Atmosphere ---
    # [LITERATURE] Andrews & Phillips. Weak turbulence Rytov variance.
    scintillation_variance: float = 0.0625  # sigma_I^2, yielding sigma_I = 0.25
    
    # [ENGINEERING] Probability of a precipitation event starting per second.
    precipitation_prob: float = 0.0003
    
    # --- Data Processing ---
    # [LITERATURE] Brassard 1993. Cascade error correction inefficiency.
    f_EC: float = 1.16
    
    # [SIMULATOR] Macro-scale temporal aggregation window.
    window_size: int = 30  # seconds

PHYSICS_CONFIG = PhysicsConfig()
