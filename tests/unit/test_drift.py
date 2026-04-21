"""Unit tests for DriftDetector."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoguard.drift.detector import DriftDetector, _compute_psi
from autoguard.core.config import DriftConfig
from autoguard.core.exceptions import DriftError


@pytest.fixture
def ref_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age":    rng.normal(40, 10, 500),
        "income": rng.normal(50000, 15000, 500),
        "city":   rng.choice(["NYC", "LA", "Chicago"], 500),
    })


@pytest.fixture
def stable_df() -> pd.DataFrame:
    """Same distribution as ref."""
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "age":    rng.normal(40, 10, 300),
        "income": rng.normal(50000, 15000, 300),
        "city":   rng.choice(["NYC", "LA", "Chicago"], 300),
    })


@pytest.fixture
def drifted_df() -> pd.DataFrame:
    """Clearly different distribution."""
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "age":    rng.normal(70, 5, 300),       # mean shifted massively
        "income": rng.normal(200000, 5000, 300), # different scale
        "city":   rng.choice(["Boston", "Miami"], 300),  # new categories
    })


class TestDriftDetectorNoShift:
    def test_stable_data_low_severity(self, ref_df, stable_df):
        det = DriftDetector()
        report = det.detect(ref_df, stable_df)
        assert report["overall_drift_severity"] < 30

    def test_report_has_required_keys(self, ref_df, stable_df):
        det = DriftDetector()
        report = det.detect(ref_df, stable_df)
        for key in ["timestamp", "reference_size", "current_size",
                    "n_features_analyzed", "n_features_drifted",
                    "drifted_features", "overall_drift_severity",
                    "drift_level", "features"]:
            assert key in report

    def test_no_common_cols_raises(self):
        det = DriftDetector()
        ref = pd.DataFrame({"a": [1, 2, 3]})
        cur = pd.DataFrame({"b": [4, 5, 6]})
        with pytest.raises(DriftError):
            det.detect(ref, cur)


class TestDriftDetectorWithShift:
    def test_drifted_data_high_severity(self, ref_df, drifted_df):
        det = DriftDetector()
        report = det.detect(ref_df, drifted_df)
        assert report["overall_drift_severity"] > 30

    def test_features_flagged(self, ref_df, drifted_df):
        det = DriftDetector()
        report = det.detect(ref_df, drifted_df)
        assert len(report["drifted_features"]) > 0

    def test_numeric_result_has_ks_and_psi(self, ref_df, drifted_df):
        det = DriftDetector()
        report = det.detect(ref_df, drifted_df)
        for feat in ["age", "income"]:
            r = report["features"][feat]
            assert "ks_pvalue" in r
            assert "psi" in r

    def test_categorical_result_has_chi2(self, ref_df, drifted_df):
        det = DriftDetector()
        report = det.detect(ref_df, drifted_df)
        r = report["features"]["city"]
        assert "chi2_pvalue" in r


class TestPSI:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 1, 1000))
        psi = _compute_psi(s, s)
        assert psi < 0.05

    def test_very_different_distributions_high_psi(self):
        rng = np.random.default_rng(0)
        ref = pd.Series(rng.normal(0, 1, 500))
        cur = pd.Series(rng.normal(10, 1, 500))
        psi = _compute_psi(ref, cur)
        assert psi > 0.2


class TestHistory:
    def test_history_grows_with_each_detect(self, ref_df, stable_df, drifted_df):
        det = DriftDetector()
        det.detect(ref_df, stable_df)
        det.detect(ref_df, drifted_df)
        hist = det.get_history()
        assert len(hist) == 2
        assert "severity" in hist.columns
