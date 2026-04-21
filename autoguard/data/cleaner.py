"""
autoguard.data.cleaner
=======================

AutoCleaner automatically preprocesses a raw dataset:
- Removes constant / all-null columns
- Fills missing numerics (median) and categoricals (mode)
- Caps extreme outliers (IQR × 3)
- Log1p-transforms highly skewed numeric columns
- Label-encodes low-cardinality categoricals
- One-hot encodes high-cardinality categoricals (if ≤ 50 uniques)
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from autoguard.core.config import DataConfig
from autoguard.core.logging import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


class AutoCleaner:
    """
    Automatic dataset cleaner / preprocessor.

    Parameters
    ----------
    config : DataConfig, optional
    """

    def __init__(self, config: Optional[DataConfig] = None) -> None:
        self.config = config or DataConfig()
        self._dropped_cols: list[str] = []
        self._fill_values: dict[str, object] = {}
        self._log_transform_cols: list[str] = []
        self._outlier_bounds: dict[str, tuple[float, float]] = {}
        self._cat_maps: dict[str, dict] = {}

    def fit_transform(self, df: pd.DataFrame, target: str = "") -> pd.DataFrame:
        """
        Fit the cleaner and return a clean copy of ``df``.

        Parameters
        ----------
        df : pd.DataFrame
        target : str
            Target column is protected from transformation.

        Returns
        -------
        pd.DataFrame
            Cleaned dataset.
        """
        df = df.copy()
        feature_cols = [c for c in df.columns if c != target]

        # 1. Drop all-null + constant columns (except target)
        for col in feature_cols:
            if df[col].isna().all() or df[col].nunique() <= 1:
                self._dropped_cols.append(col)
        if self._dropped_cols:
            logger.info(f"  Dropping {len(self._dropped_cols)} constant/empty columns: {self._dropped_cols}")
            df = df.drop(columns=self._dropped_cols)
            feature_cols = [c for c in feature_cols if c not in self._dropped_cols]

        num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

        # 2. Fill missing values
        for col in num_cols:
            fill = float(df[col].median())
            self._fill_values[col] = fill
            df[col] = df[col].fillna(fill)

        for col in cat_cols:
            mode = df[col].mode()
            fill = mode.iloc[0] if not mode.empty else "UNKNOWN"
            self._fill_values[col] = fill
            df[col] = df[col].fillna(fill)

        # 3. Cap outliers (IQR × 3) for numeric columns
        for col in num_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            self._outlier_bounds[col] = (lo, hi)
            df[col] = df[col].clip(lo, hi)

        # 4. Log-transform highly skewed columns
        for col in num_cols:
            sk = abs(float(df[col].skew()))
            if sk > self.config.skewness_threshold and df[col].min() >= 0:
                self._log_transform_cols.append(col)
                df[col] = np.log1p(df[col])

        # 5. Encode categoricals
        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique <= 15:
                # Label encode
                mapping = {v: i for i, v in enumerate(df[col].unique())}
                self._cat_maps[col] = mapping
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
            elif n_unique <= 50:
                # One-hot encode
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            else:
                # High cardinality: frequency encode
                freq = df[col].value_counts(normalize=True).to_dict()
                self._cat_maps[col] = freq
                df[col] = df[col].map(freq).fillna(0.0)

        n_missing_after = df.isnull().sum().sum()
        logger.info(
            f"  AutoCleaner: {len(self._dropped_cols)} cols dropped | "
            f"{len(self._log_transform_cols)} log-transformed | "
            f"{n_missing_after} missing cells remaining"
        )
        return df

    def transform(self, df: pd.DataFrame, target: str = "") -> pd.DataFrame:
        """
        Apply the fitted cleaner to new data.

        Parameters
        ----------
        df : pd.DataFrame
        target : str

        Returns
        -------
        pd.DataFrame
        """
        df = df.copy()
        feature_cols = [c for c in df.columns if c != target]

        df = df.drop(columns=[c for c in self._dropped_cols if c in df.columns])

        for col, val in self._fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        for col, (lo, hi) in self._outlier_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)

        for col in self._log_transform_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))

        for col, mapping in self._cat_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0)

        return df
