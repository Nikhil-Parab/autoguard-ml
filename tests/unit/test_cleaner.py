"""Unit tests for AutoCleaner."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoguard.data.cleaner import AutoCleaner


@pytest.fixture
def raw_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "age":      rng.integers(18, 90, 150).astype(float),
        "income":   rng.exponential(30000, 150),   # skewed
        "city":     rng.choice(["NYC", "LA", "Chicago", None], 150),
        "constant": [99] * 150,
        "label":    rng.choice([0, 1], 150),
    })
    df.loc[df.sample(30, random_state=1).index, "age"] = np.nan
    df.loc[df.sample(20, random_state=2).index, "income"] = np.nan
    return df


class TestAutoCleaner:
    def test_no_missing_after_fix(self, raw_df):
        cleaner = AutoCleaner()
        df_clean = cleaner.fit_transform(raw_df, target="label")
        assert df_clean.drop(columns=["label"]).isnull().sum().sum() == 0

    def test_drops_constant_column(self, raw_df):
        cleaner = AutoCleaner()
        df_clean = cleaner.fit_transform(raw_df, target="label")
        assert "constant" not in df_clean.columns

    def test_target_preserved(self, raw_df):
        cleaner = AutoCleaner()
        df_clean = cleaner.fit_transform(raw_df, target="label")
        assert "label" in df_clean.columns
        assert df_clean["label"].equals(raw_df["label"])

    def test_output_is_dataframe(self, raw_df):
        cleaner = AutoCleaner()
        result = cleaner.fit_transform(raw_df, target="label")
        assert isinstance(result, pd.DataFrame)

    def test_transform_on_new_data(self, raw_df):
        cleaner = AutoCleaner()
        cleaner.fit_transform(raw_df, target="label")
        new_df = raw_df.head(20).copy()
        new_df.loc[0, "age"] = np.nan
        result = cleaner.transform(new_df, target="label")
        assert isinstance(result, pd.DataFrame)
        assert result.isnull().sum().sum() == 0
