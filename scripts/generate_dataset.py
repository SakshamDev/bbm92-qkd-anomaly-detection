"""
scripts/generate_dataset.py — Standalone dataset generation script for BBM92 QKD telemetry.

Generates the canonical 24-hour telemetry dataset used for dashboard simulation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from core.telemetry import build_telemetry_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Generating canonical 24-hour telemetry dataset...")
    output_path = 'data/telemetry_86400.parquet'
    build_telemetry_dataset(n_seconds=86400, seed=42, output_path=output_path)
    logger.info(f"Dataset successfully generated at {output_path}")

if __name__ == "__main__":
    main()
