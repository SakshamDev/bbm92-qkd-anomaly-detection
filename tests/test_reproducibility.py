import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.telemetry import build_telemetry_dataset

def test_dataset_reproducibility(tmp_path):
    """Verify that dataset generation is bit-exact deterministic given a seed."""
    path1 = tmp_path / "ds1.parquet"
    path2 = tmp_path / "ds2.parquet"
    
    # Generate dataset 1
    df1 = build_telemetry_dataset(n_seconds=1000, seed=42, output_path=str(path1))
    
    # Generate dataset 2
    df2 = build_telemetry_dataset(n_seconds=1000, seed=42, output_path=str(path2))
    
    # Check shape
    assert df1.shape == df2.shape
    
    # Check bit-exact equality of all numeric columns
    for col in df1.columns:
        if pd.api.types.is_numeric_dtype(df1[col]):
            np.testing.assert_array_equal(df1[col].values, df2[col].values)
            
    # Also verify that a different seed produces different results
    path3 = tmp_path / "ds3.parquet"
    df3 = build_telemetry_dataset(n_seconds=1000, seed=43, output_path=str(path3))
    assert not np.array_equal(df1['qber'].values, df3['qber'].values)
