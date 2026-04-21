"""Unit tests for DatasetDoctor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoguard.data.doctor import DatasetDoctor
from autoguard.core.config import DataConfig
from autoguard.core.exceptions import DataError


@pytest.fixture
def clean_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "age":    rng.integers(20, 80, 200).astype(float),
        "income": rng.normal(50000, 15000, 200),
        "score":  rng.uniform(0, 1, 200),
        "label":  rng.choice([0, 1], 200),
    })


@pytest.fixture
def messy_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age":     rng.integers(20, 80, 300).astype(float),
        "income":  rng.normal(50000, 15000, 300),
        "leak":    [0] * 290 + [1] * 10,         # label leak
        "constant": [42] * 300,
        "label":    [0] * 290 + [1] * 10,         # imbalanced
    })
    # Inject missing
    df.loc[df.sample(100, random_state=1).index, "age"] = np.nan
    return df


class TestDatasetDoctorClean:
    def test_returns_dict_with_keys(self, clean_df):
        doc = DatasetDoctor(DataConfig(generate_plots=False))
        report = doc.diagnose(clean_df, target="label")
        assert "risk_score" in report
        assert "issues" in report
        assert "risk_level" in report
        assert "missing_values" in report

    def test_risk_score_bounded(self, clean_df):
        doc = DatasetDoctor(DataConfig(generate_plots=False))
        report = doc.diagnose(clean_df, target="label")
        assert 0 <= report["risk_score"] <= 100

    def test_clean_data_low_risk(self, clean_df):
        doc = DatasetDoctor(DataConfig(generate_plots=False))
        report = doc.diagnose(clean_df, target="label")
        assert report["risk_level"] in ("low", "medium")

    def test_shape_reported_correctly(self, clean_df):
        doc = DatasetDoctor(DataConfig(generate_plots=False))
        report = doc.diagnose(clean_df, target="label")
        assert report["shape"]["rows"] == len(clean_df)
        assert report["shape"]["cols"] == len(clean_df.columns)


class TestDatasetDoctorMessy:
    def test_detects_missing_values(self, messy_df):
        doc = DatasetDoctor(DataConfig(generate_plots=False, missing_threshold=0.2))
        report = doc.diagnose(messy_df, target="label")
        assert len(report["missing_values"]["high_missing_columns"]) > 0

    def test_detects_class_imbalance(self, messy_df):
        doc = DatasetDoctor(DataConfig(generate_plots=False))
        report = doc.diagnose(messy_df, target="label")
        assert report["class_imbalance"]["is_imbalanced"]

    def test_detects_constant_column(self, messy_df):
        doc = DatasetDoctor(DataConfig(generate_plots=False))
        report = doc.diagnose(messy_df, target="label")
        assert "constant" in report["constant_columns"]["constant_columns"]

    def test_higher_risk_score_than_clean(self, clean_df, messy_df):
        doc = DatasetDoctor(DataConfig(generate_plots=False))
        clean_report = doc.diagnose(clean_df, target="label")
        messy_report = doc.diagnose(messy_df, target="label")
        assert messy_report["risk_score"] > clean_report["risk_score"]


class TestDatasetDoctorEdgeCases:
    def test_raises_on_missing_target(self, clean_df):
        doc = DatasetDoctor(DataConfig(generate_plots=False))
        with pytest.raises(DataError):
            doc.diagnose(clean_df, target="nonexistent")

    def test_raises_on_empty_df(self):
        doc = DatasetDoctor(DataConfig(generate_plots=False))
        with pytest.raises(DataError):
            doc.diagnose(pd.DataFrame(), target="label")

    def test_continuous_target_skips_imbalance(self, clean_df):
        # Make label continuous
        df = clean_df.copy()
        df["label"] = df["income"]  # continuous
        doc = DatasetDoctor(DataConfig(generate_plots=False))
        report = doc.diagnose(df, target="label")
        assert report["class_imbalance"]["status"] == "skipped"
