"""
core/telemetry.py — Dataset assembler for BBM92 QKD telemetry.

Generates the full 86,400-second (24-hour) telemetry dataset by:
  1. Simulating a normal BBM92 atmospheric channel
  2. Injecting attack events with a controlled budget (~15%)
  3. Assembling into a Parquet-persisted DataFrame

Attack injection budget (§4.2):
    IR attacks:              4 events × ~400s each  ≈  1,600s
    Detector Blinding:       3 events × ~600s each  ≈  1,800s
    MitM attacks:            2 events × ~300s each  ≈    600s
    Blended bursts:        300 bursts ×  ~30s each  ≈  9,000s
                                            Total ≈ 12,960s (≈15%)

Output schema (9 columns):
    timestamp, qber, bell_S, coincidence_rate, visibility,
    channel_loss_dB, detection_rate, label, attack_type

Attack type encoding:
    0 = normal, 1 = intercept_resend, 2 = detector_blinding,
    3 = mitm, 4 = blended_subthreshold

"""

import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from core.channel import simulate_normal_channel
from core.attacks import (
    InterceptResendStrategy,
    FakedStateStrategy,
    MitMStrategy,
    BlendedSubthresholdStrategy,
    AttackStrategy
)
from core.config import EveCapabilities

logger = logging.getLogger(__name__)


def _assert_no_overlap(strategies: List[AttackStrategy], n_seconds: int) -> None:
    occupied_dict: dict[int, str] = {}
    for s in strategies:
        sl = s._get_slice(n_seconds)
        for t in range(sl.start, sl.stop):
            if t in occupied_dict:
                raise ValueError(
                    f"Temporal overlap at t={t}s: "
                    f"{occupied_dict[t]} and {s.__class__.__name__} both claim this second."
                )
            occupied_dict[t] = s.__class__.__name__


def _randomized_starts(rng, n_events, window_start, window_end, min_duration, existing_starts=None, min_gap=1800, occupied_seconds=None):
    starts = list(existing_starts) if existing_starts is not None else []
    occupied = set(occupied_seconds) if occupied_seconds else set()
    attempts = 0
    added = 0
    
    valid_start = max(window_start, min_duration)
    valid_end = max(valid_start + 1, window_end - min_duration)
    
    while added < n_events and attempts < 20000:
        candidate = int(rng.integers(valid_start, valid_end))
        candidate_seconds = set(range(candidate, min(candidate + min_duration, window_end)))
        same_type_clear = all(abs(candidate - s) >= min_gap for s in starts)
        cross_type_clear = candidate_seconds.isdisjoint(occupied)
        if same_type_clear and cross_type_clear:
            starts.append(candidate)
            occupied |= candidate_seconds
            added += 1
        attempts += 1
        
    return sorted(starts), occupied


def build_level3_caps(seed: int, n_seconds: int) -> EveCapabilities:
    """
    Build Level 3 EveCapabilities with true historical beacon replay.
    The historical atmospheric recording uses an independent RNG seed
    (seed XOR 0xBEEF5A5A) so it has the same lognormal marginal
    distribution (including precipitation) as the real channel but is 
    statistically independent.
    """
    from core.channel import generate_lognormal_fading, precipitation_model
    from core.policies import (
        MeanMatchingAttenuationPolicy,
        HistoricalReplayPolicy,
        AdaptiveCountMatching,
    )
    from core.config import PHYSICS_CONFIG

    alt_seed = seed ^ 0xBEEF5A5A
    alt_rng = np.random.default_rng(alt_seed)
    
    historical_eta_scint = generate_lognormal_fading(
        n_seconds, PHYSICS_CONFIG.scintillation_variance, alt_seed
    )
    historical_precip_loss_dB = precipitation_model(
        n_seconds, PHYSICS_CONFIG.precipitation_prob, alt_rng
    )
    historical_eta_precip = 10.0 ** (-historical_precip_loss_dB / 10.0)
    
    # Scale by the same base transmittance used in the real channel
    historical_eta = PHYSICS_CONFIG.eta_atm_bob_base * historical_eta_scint * historical_eta_precip

    return EveCapabilities(
        attenuation_policy=MeanMatchingAttenuationPolicy(),
        beacon_spoofing_policy=HistoricalReplayPolicy(historical_eta),
        detector_control_policy=AdaptiveCountMatching(),
    )


def build_telemetry_dataset(
    n_seconds: int = 86400,
    seed: int = 42,
    output_path: str = 'data/telemetry_86400.parquet',
    caps: EveCapabilities = None,
) -> pd.DataFrame:
    if output_path is not None and Path(output_path).exists():
        logger.info("Found existing telemetry dataset at %s. Loading from disk instead of regenerating.", output_path)
        return pd.read_parquet(output_path)
        
    t_start = time.time()
    rng = np.random.default_rng(seed)

    logger.info("=== Building BBM92 telemetry dataset ===")
    logger.info("Duration: %d seconds, Seed: %d", n_seconds, seed)

    # ──────────────────────────────────────────────────────────────────
    # Step 1: Schedule Attacks
    # ──────────────────────────────────────────────────────────────────
    logger.info("Step 1/3: Scheduling attack strategies (randomized timings)...")
    strategies: List[AttackStrategy] = []
    
    holdout_start = int(n_seconds * 0.75)  # 75% point for holdout

    # We assume Eve has access to the target capabilities for this dataset
    if caps is None:
        caps = EveCapabilities()

    occupied = set()
    # IR attacks (4 events)
    ir_holdout, occupied = _randomized_starts(rng, 1, holdout_start, n_seconds, min_duration=600, occupied_seconds=occupied)
    ir_starts, occupied = _randomized_starts(rng, 3, 0, n_seconds, min_duration=600, existing_starts=ir_holdout, occupied_seconds=occupied)
    for start in ir_starts:
        duration = int(rng.integers(300, 600))
        eve_frac = float(rng.uniform(0.20, 0.35))
        strategies.append(InterceptResendStrategy(caps, start_sec=start, duration_sec=duration, fraction=eve_frac))
        logger.info("  IR scheduled: t=%d, dur=%ds, fraction=%.2f", start, duration, eve_frac)

    # Detector blinding attacks (3 events)
    db_holdout, occupied = _randomized_starts(rng, 1, holdout_start, n_seconds, min_duration=800, occupied_seconds=occupied)
    db_starts, occupied = _randomized_starts(rng, 2, 0, n_seconds, min_duration=800, existing_starts=db_holdout, occupied_seconds=occupied)
    for start in db_starts:
        duration = int(rng.integers(400, 800))
        intensity = float(rng.uniform(0.35, 0.65))
        trigger_rate = intensity * 15000.0
        strategies.append(FakedStateStrategy(caps, start_sec=start, duration_sec=duration, trigger_rate=trigger_rate))
        logger.info("  Detector blinding scheduled: t=%d, dur=%ds, trigger_rate=%.1f", start, duration, trigger_rate)

    # MitM attacks (2 events)
    mitm_holdout, occupied = _randomized_starts(rng, 1, holdout_start, n_seconds, min_duration=400, occupied_seconds=occupied)
    mitm_starts, occupied = _randomized_starts(rng, 1, 0, n_seconds, min_duration=400, existing_starts=mitm_holdout, occupied_seconds=occupied)
    for start in mitm_starts:
        duration = int(rng.integers(200, 400))
        strategies.append(MitMStrategy(caps, start_sec=start, duration_sec=duration))
        logger.info("  MitM scheduled: t=%d, dur=%ds", start, duration)

    # Blended sub-threshold (300 bursts)
    burst_count = 300
    burst_duration = 30
    blended_starts, occupied = _randomized_starts(
        rng, burst_count, 0, n_seconds, min_duration=burst_duration, 
        min_gap=burst_duration + 1, occupied_seconds=occupied
    )
    for start in blended_starts:
        eve_frac = float(rng.uniform(0.10, 0.14))
        strategies.append(BlendedSubthresholdStrategy(caps, start_sec=start, duration_sec=burst_duration, fraction=eve_frac))

    logger.info("  Blended sub-threshold: %d bursts scheduled", burst_count)

    # ──────────────────────────────────────────────────────────────────
    # Step 2: Execute First-Principles Simulation
    # ──────────────────────────────────────────────────────────────────
    _assert_no_overlap(strategies, n_seconds)
    logger.info("Step 2/3: Simulating physics core with attacks...")
    base = simulate_normal_channel(n_seconds=n_seconds, seed=seed, attack_strategies=strategies)

    # ──────────────────────────────────────────────────────────────────
    # Step 3: Build DataFrame
    # ──────────────────────────────────────────────────────────────────
    logger.info("Step 3/3: Assembling DataFrame and persisting...")

    timestamps = pd.date_range('2026-01-01 00:00:00', periods=n_seconds, freq='1s')
    df = pd.DataFrame({
        'timestamp': timestamps,
        'qber': base['qber'],
        'bell_S': base['bell_S'],
        'coincidence_rate': base['coincidence_rate'].astype(float),
        'visibility': base['visibility'],
        'channel_loss_dB': base['channel_loss_dB'],
        'detection_rate': base['detection_rate'].astype(float),
        'label': base['label'],
        'attack_type': base['attack_type'],
    })
    
    # E-6: QBER sentinel explicit masking for feature extraction
    df['is_low_count'] = (df['qber'] == 0.5) & (df['bell_S'] == 0.0)

    # Verify class balance
    attack_frac = df['label'].mean()
    n_attack = int(df['label'].sum())
    n_normal = len(df) - n_attack

    logger.info("Dataset: %d rows | Normal: %d (%.1f%%) | Attack: %d (%.1f%%)", len(df), n_normal, (1 - attack_frac) * 100, n_attack, attack_frac * 100)

    # Persist
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info("Saved to %s", output_path)

    elapsed = time.time() - t_start
    logger.info("Finished dataset generation in %.2f seconds", elapsed)
    print(f"Dataset: {len(df):,} rows | Attack fraction: {attack_frac:.3f} "
          f"({attack_frac * 100:.1f}%) | Time: {elapsed:.2f}s")
    print(f"Attack type distribution:\n{df['attack_type'].value_counts().sort_index().to_string()}")

    return df

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    build_telemetry_dataset()
