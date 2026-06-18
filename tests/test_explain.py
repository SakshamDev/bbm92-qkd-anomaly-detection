"""
tests/test_explain.py — SHAP explainability module validation.

Tests SHAP explainer construction and alert explanation output.
"""

import sys
import os
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.explain import build_shap_explainer, explain_single_alert
from ml.features import FEATURE_NAMES

def _train_test_model(tmp_path):
    """Train a small model for SHAP testing."""
    rng = np.random.default_rng(42)
    n = 500
    n_attack = 75
    X_normal = rng.normal(0, 1, (n - n_attack, 24)).astype(np.float32)
    X_attack = rng.normal(2, 1.5, (n_attack, 24)).astype(np.float32)
    X = np.vstack([X_normal, X_attack])
    y = np.concatenate([np.zeros(n - n_attack), np.ones(n_attack)]).astype(int)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]

    xgb_model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    xgb_model.fit(X, y)
    
    model_artifacts = {'xgb': xgb_model}
    return model_artifacts, X

class TestBuildShapExplainer:
    """Tests for SHAP explainer construction."""

    def test_builds_without_error(self, tmp_path):
        artifacts, X = _train_test_model(tmp_path)
        explainer = build_shap_explainer(artifacts, X[:100])
        assert explainer is not None

    def test_expected_value_exists(self, tmp_path):
        artifacts, X = _train_test_model(tmp_path)
        explainer = build_shap_explainer(artifacts, X[:100])
        assert hasattr(explainer, 'expected_value')

class TestExplainSingleAlert:
    """Tests for single-alert SHAP explanation."""

    @pytest.fixture(scope='class')
    def shap_setup(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp('shap')
        artifacts, X = _train_test_model(tmp_path)
        explainer = build_shap_explainer(artifacts, X[:100])
        return artifacts, explainer, X

    def test_output_keys(self, shap_setup):
        artifacts, explainer, X = shap_setup
        result = explain_single_alert(X[0], explainer, artifacts, top_k=5)
        assert 'feature_names' in result
        assert 'shap_values' in result
        assert 'base_value' in result

    def test_top_k_count(self, shap_setup):
        artifacts, explainer, X = shap_setup
        result = explain_single_alert(X[0], explainer, artifacts, top_k=5)
        assert len(result['feature_names']) == 5
        assert len(result['shap_values']) == 5

    def test_feature_names_valid(self, shap_setup):
        artifacts, explainer, X = shap_setup
        result = explain_single_alert(X[0], explainer, artifacts, top_k=5)
        for name in result['feature_names']:
            assert name in FEATURE_NAMES, f"'{name}' not in FEATURE_NAMES"

    def test_base_value_float(self, shap_setup):
        artifacts, explainer, X = shap_setup
        result = explain_single_alert(X[0], explainer, artifacts, top_k=5)
        assert isinstance(result['base_value'], float)

    def test_shap_values_are_float(self, shap_setup):
        artifacts, explainer, X = shap_setup
        result = explain_single_alert(X[0], explainer, artifacts, top_k=5)
        for v in result['shap_values']:
            assert isinstance(v, float)
