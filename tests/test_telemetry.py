"""
tests/test_telemetry.py — Dataset generation validation.

Implements Test E from §15.1: Dataset Generation.

Expected:
    - exactly 86,400 rows
    - no missing values
    - attack fraction between 12% and 18%
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bbm92_drdo'))

from core.telemetry import build_telemetry_dataset


class TestBuildTelemetryDataset:
    """
    Test E from §15.1: Dataset generation validation.
    """

    @pytest.fixture(scope='class')
    def dataset(self, tmp_path_factory):
        """Generate dataset once for all tests."""
        tmp_dir = tmp_path_factory.mktemp('data')
        output_path = str(tmp_dir / 'test_telemetry.parquet')
        df = build_telemetry_dataset(
            n_seconds=86400,
            seed=42,
            output_path=output_path,
        )
        return df, output_path

    def test_acceptance_e_row_count(self, dataset):
        """§15.1 Test E: exactly 86,400 rows."""
        df, _ = dataset
        assert len(df) == 86400, (
            f"Row count = {len(df)}, expected 86400"
        )

    def test_acceptance_e_no_missing(self, dataset):
        """§15.1 Test E: no missing values."""
        df, _ = dataset
        assert df.notna().all().all(), (
            f"Missing values found: {df.isna().sum().to_dict()}"
        )

    def test_acceptance_e_attack_fraction(self, dataset):
        """§15.1 Test E: attack fraction between 12% and 18%."""
        df, _ = dataset
        attack_frac = df['label'].mean()
        assert 0.12 <= attack_frac <= 0.18, (
            f"Attack fraction = {attack_frac:.3f}, expected [0.12, 0.18]"
        )

    def test_expected_columns(self, dataset):
        """All 9 columns should be present."""
        df, _ = dataset
        expected = {
            'timestamp', 'qber', 'bell_S', 'coincidence_rate',
            'visibility', 'channel_loss_dB', 'detection_rate', 'label', 'attack_type',
            'is_low_count',
        }
        assert set(df.columns) == expected

    def test_parquet_file_exists(self, dataset):
        """Parquet file should be created on disk."""
        _, path = dataset
        assert Path(path).exists()

    def test_parquet_roundtrip(self, dataset):
        """Read back the Parquet file and verify shape."""
        df_orig, path = dataset
        df_read = pd.read_parquet(path)
        assert len(df_read) == len(df_orig)
        assert set(df_read.columns) == set(df_orig.columns)

    def test_label_values(self, dataset):
        """Labels should only be 0 or 1."""
        df, _ = dataset
        assert set(df['label'].unique()).issubset({0, 1})

    def test_timestamp_monotonic(self, dataset):
        """Timestamps should be monotonically increasing."""
        df, _ = dataset
        assert df['timestamp'].is_monotonic_increasing

    def test_qber_physical_range(self, dataset):
        """QBER should be in physically plausible range (0 to 1.0)."""
        df, _ = dataset
        assert df['qber'].min() >= 0.0
        assert df['qber'].max() <= 1.0

    def test_bell_s_physical_range(self, dataset):
        """Bell S should be in [0.0, 3.5] (0.0 during low-count masking)."""
        df, _ = dataset
        assert df['bell_S'].min() >= 0.0
        assert df['bell_S'].max() <= 3.5
