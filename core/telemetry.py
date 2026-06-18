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

import numpy as np
import pandas as pd

from core.channel import simulate_normal_channel
from core.attacks import (
    attack_intercept_resend,
    attack_detector_blinding,
    attack_mitm,
    attack_blended_subthreshold,
)

logger = logging.getLogger(__name__)


def _randomized_starts(rng, n_events, window_start, window_end, min_duration, existing_starts=None, min_gap=1800):
    """
    Generate non-overlapping randomized attack start times within a specific window.

    Args:
        rng: NumPy random generator.
        n_events: Number of attack events to place.
        window_start: Minimum start time.
        window_end: Maximum end time limit.
        min_duration: Minimum duration of each attack (for gap calculation).
        existing_starts: List of already scheduled starts to avoid overlapping with.
        min_gap: Minimum gap between any two attack start times.

    Returns:
        Sorted list of start times combining existing and newly scheduled starts.
    """
    starts = list(existing_starts) if existing_starts is not None else []
    attempts = 0
    added = 0
    
    # Ensure window is valid
    valid_start = max(window_start, min_duration)
    valid_end = max(valid_start + 1, window_end - min_duration)
    
    while added < n_events and attempts < 1000:
        candidate = int(rng.integers(valid_start, valid_end))
        # Check minimum gap from all existing starts
        if all(abs(candidate - s) >= min_gap for s in starts):
            starts.append(candidate)
            added += 1
        attempts += 1
        
    return sorted(starts)


def build_telemetry_dataset(
    n_seconds: int = 86400,
    seed: int = 42,
    output_path: str = 'data/telemetry_86400.parquet',
) -> pd.DataFrame:
    """
    Generates the full BBM92 telemetry dataset with attack injection.

    The dataset represents one complete 24-hour operational day of a
    BBM92 entanglement-based QKD link between two DRDO nodes.

    Attack events are injected at RANDOMIZED times (seeded for
    reproducibility) to prevent the ML model from learning temporal
    position rather than physics signatures.

    Args:
        n_seconds: Total duration in seconds (default: 86,400 = 24h).
        seed: Master RNG seed for full reproducibility.
        output_path: Filesystem path for the Parquet output.

    Returns:
        pd.DataFrame with 86,400 rows and 9 columns.

    Raises:
        AssertionError: If the attack fraction falls outside [0.10, 0.20].
    """
    t_start = time.time()
    rng = np.random.default_rng(seed)

    logger.info("=== Building BBM92 telemetry dataset ===")
    logger.info("Duration: %d seconds, Seed: %d", n_seconds, seed)

    # ──────────────────────────────────────────────────────────────────
    # Step 1: Generate full normal baseline
    # ──────────────────────────────────────────────────────────────────
    logger.info("Step 1/3: Generating normal baseline channel...")
    base = simulate_normal_channel(n_seconds=n_seconds, seed=seed)
    # Add attack_type column (0 = normal)
    base['attack_type'] = np.zeros(n_seconds, dtype=int)

    # ──────────────────────────────────────────────────────────────────
    # Step 2: Inject attack events with RANDOMIZED timings
    # ──────────────────────────────────────────────────────────────────
    logger.info("Step 2/3: Injecting attack events (randomized timings)...")

    holdout_start = 64800  # 18 hours

    # IR attacks (4 events: guarantee 1 in holdout)
    ir_holdout = _randomized_starts(rng, 1, holdout_start, n_seconds, min_duration=600)
    ir_starts = _randomized_starts(rng, 3, 0, n_seconds, min_duration=600, existing_starts=ir_holdout)
    for start in ir_starts:
        duration = int(rng.integers(300, 600))
        eve_frac = float(rng.uniform(0.20, 0.35))
        base = attack_intercept_resend(
            base,
            duration_sec=duration,
            start_sec=start,
            eve_fraction=eve_frac,
            rng=rng,
        )
        logger.info(
            "  IR attack injected: t=%d, dur=%ds, eve=%.2f",
            start, duration, eve_frac,
        )

    # Detector blinding attacks (3 events: guarantee 1 in holdout)
    db_holdout = _randomized_starts(rng, 1, holdout_start, n_seconds, min_duration=800)
    db_starts = _randomized_starts(rng, 2, 0, n_seconds, min_duration=800, existing_starts=db_holdout)
    for start in db_starts:
        duration = int(rng.integers(400, 800))
        intensity = float(rng.uniform(0.35, 0.65))
        base = attack_detector_blinding(
            base,
            duration_sec=duration,
            start_sec=start,
            blinding_intensity=intensity,
            rng=rng,
        )
        logger.info(
            "  Detector blinding injected: t=%d, dur=%ds, intensity=%.2f",
            start, duration, intensity,
        )

    # MitM attacks (2 events: guarantee 1 in holdout)
    mitm_holdout = _randomized_starts(rng, 1, holdout_start, n_seconds, min_duration=400)
    mitm_starts = _randomized_starts(rng, 1, 0, n_seconds, min_duration=400, existing_starts=mitm_holdout)
    for start in mitm_starts:
        duration = int(rng.integers(200, 400))
        base = attack_mitm(
            base,
            duration_sec=duration,
            start_sec=start,
            rng=rng,
        )
        logger.info(
            "  MitM attack injected: t=%d, dur=%ds",
            start, duration,
        )

    # Blended sub-threshold (300 short bursts — the hard case)
    base = attack_blended_subthreshold(
        base,
        n_bursts=300,
        burst_duration=30,
        eve_fraction=0.12,
        rng=rng,
    )
    logger.info("  Blended sub-threshold: 300 bursts × 30s injected")

    # ──────────────────────────────────────────────────────────────────
    # Step 3: Build DataFrame
    # ──────────────────────────────────────────────────────────────────
    logger.info("Step 3/3: Assembling DataFrame and persisting...")

    timestamps = pd.date_range(
        '2026-01-01 00:00:00', periods=n_seconds, freq='1s'
    )
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

    # Verify class balance
    attack_frac = df['label'].mean()
    n_attack = int(df['label'].sum())
    n_normal = len(df) - n_attack

    logger.info(
        "Dataset: %d rows | Normal: %d (%.1f%%) | Attack: %d (%.1f%%)",
        len(df), n_normal, (1 - attack_frac) * 100,
        n_attack, attack_frac * 100,
    )

    assert 0.10 <= attack_frac <= 0.20, (
        f"Class balance out of target range: {attack_frac:.3f} "
        f"(expected 0.10–0.20). Got {n_attack} attack seconds."
    )

    # Verify no missing values
    assert df.notna().all().all(), (
        f"Missing values detected: {df.isna().sum().to_dict()}"
    )

    # Persist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    elapsed = time.time() - t_start
    logger.info("Saved to %s (%.2f seconds)", output_path, elapsed)
    print(f"Dataset: {len(df):,} rows | Attack fraction: {attack_frac:.3f} "
          f"({attack_frac * 100:.1f}%) | Time: {elapsed:.2f}s")
    print(f"Attack type distribution:\n{df['attack_type'].value_counts().sort_index().to_string()}")

    return df


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    )
    build_telemetry_dataset()
