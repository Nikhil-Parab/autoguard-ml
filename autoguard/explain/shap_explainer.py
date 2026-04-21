"""
autoguard.explain.shap_explainer
==================================

SHAP-based explainability:
- Auto-selects best explainer (Tree, Linear, Kernel)
- Global feature importance bar + beeswarm plots
- Local single-prediction waterfall chart
- Feature importance as a ranked Series
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoguard.core.config import ExplainConfig
from autoguard.core.exceptions import ExplainError
from autoguard.core.logging import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


class ShapExplainer:
    """
    SHAP explainability wrapper.

    Parameters
    ----------
    model : sklearn-compatible estimator
    model_name : str
    config : ExplainConfig
    problem_type : str  "classification" | "regression"
    """

    def __init__(
        self,
        model: Any,
        model_name: str = "model",
        config: Optional[ExplainConfig] = None,
        problem_type: str = "classification",
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.config = config or ExplainConfig()
        self.problem_type = problem_type

        self._explainer: Any = None
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame) -> "ShapExplainer":
        """Build the SHAP explainer from background data."""
        try:
            import shap
        except ImportError:
            raise ExplainError("SHAP not installed. Run: pip install shap")

        self._feature_names = [str(c) for c in X.columns]
        n = min(self.config.max_samples, len(X))
        X_bg = X.sample(n=n, random_state=42)

        self._explainer = self._build_explainer(shap, X_bg)
        logger.info(
            f"SHAP explainer: [bold]{type(self._explainer).__name__}[/bold] "
            f"({n} background samples)"
        )
        return self

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values, returns 2D array (n_samples, n_features)."""
        if self._explainer is None:
            raise ExplainError("Call .fit() first.")

        n = min(self.config.max_samples, len(X))
        X_s = X.sample(n=n, random_state=42)

        try:
            raw = self._explainer(X_s)
        except Exception:
            # Fallback: try legacy API
            try:
                raw = self._explainer.shap_values(X_s)
            except Exception as e:
                raise ExplainError(f"SHAP computation failed: {e}") from e

        return self._unify(raw)

    def plot_global(self, X: pd.DataFrame) -> None:
        """Save global feature importance bar + summary plots."""
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        shap_vals = self.get_shap_values(X)
        importance = pd.Series(
            np.abs(shap_vals).mean(axis=0),
            index=self._feature_names[:shap_vals.shape[1]],
        ).sort_values(ascending=False)

        top = importance.head(self.config.plot_top_n_features)

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        top.sort_values().plot(kind="barh", ax=ax, color="#2196F3")
        ax.set_title(f"Global Feature Importance (SHAP) — {self.model_name}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean |SHAP Value|")
        plt.tight_layout()
        fig.savefig(out / "shap_global_importance.png", dpi=150)
        plt.close(fig)

        # Beeswarm / dot plot (best-effort)
        try:
            import shap
            n = min(self.config.max_samples, len(X))
            X_s = X.sample(n=n, random_state=42)
            top_feat = top.index.tolist()
            top_idx = [self._feature_names.index(f) for f in top_feat if f in self._feature_names]
            shap.summary_plot(
                shap_vals[:, top_idx],
                X_s[[f for f in top_feat if f in X_s.columns]].values,
                feature_names=top_feat,
                show=False,
                max_display=self.config.plot_top_n_features,
            )
            plt.tight_layout()
            plt.savefig(out / "shap_summary_plot.png", dpi=150)
            plt.close()
        except Exception as e:
            logger.debug(f"Beeswarm plot failed: {e}")

        logger.info(f"SHAP global plots → [cyan]{out}[/cyan]")

    def plot_local(self, X: pd.DataFrame, index: int = 0) -> None:
        """Save a local explanation waterfall chart for one sample."""
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        shap_vals = self.get_shap_values(X)
        n = min(self.config.max_samples, len(X))
        idx = min(index, n - 1)
        row_shap = shap_vals[idx]

        n_feat = min(self.config.plot_top_n_features, len(self._feature_names))
        top_idx = np.argsort(np.abs(row_shap))[-n_feat:]
        top_names = [self._feature_names[i] for i in top_idx if i < len(self._feature_names)]
        top_vals = row_shap[top_idx[:len(top_names)]]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in top_vals]
        ax.barh(range(len(top_names)), top_vals, color=colors)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"Local SHAP Explanation — Sample #{idx}", fontsize=13, fontweight="bold")
        ax.set_xlabel("SHAP Value")
        plt.tight_layout()
        fig.savefig(out / f"shap_local_{idx}.png", dpi=150)
        plt.close(fig)
        logger.info(f"SHAP local plot → [cyan]{out}/shap_local_{idx}.png[/cyan]")

    def get_feature_importance(self, X: pd.DataFrame) -> pd.Series:
        """Return ranked feature importance Series."""
        sv = self.get_shap_values(X)
        return pd.Series(
            np.abs(sv).mean(axis=0),
            index=self._feature_names[:sv.shape[1]],
        ).sort_values(ascending=False)

    # ── internal ─────────────────────────────────────────────────────

    def _build_explainer(self, shap: Any, X_bg: pd.DataFrame) -> Any:
        mtype = type(self.model).__name__.lower()
        tree_types = {
            "randomforestclassifier", "randomforestregressor",
            "xgbclassifier", "xgbregressor",
            "lgbmclassifier", "lgbmregressor",
            "gradientboostingclassifier", "gradientboostingregressor",
            "extratreesclassifier", "extratreesregressor",
        }
        linear_types = {
            "logisticregression", "ridge", "ridgeclassifier",
            "lasso", "elasticnet", "linearregression",
        }
        try:
            if mtype in tree_types:
                return shap.TreeExplainer(self.model)
            elif mtype in linear_types:
                return shap.LinearExplainer(self.model, X_bg)
            else:
                fn = (self.model.predict_proba
                      if self.problem_type == "classification"
                      and hasattr(self.model, "predict_proba")
                      else self.model.predict)
                return shap.KernelExplainer(fn, shap.sample(X_bg, min(50, len(X_bg))))
        except Exception as e:
            logger.warning(f"Primary SHAP explainer failed ({e}). Using KernelExplainer.")
            fn = (self.model.predict_proba
                  if hasattr(self.model, "predict_proba") else self.model.predict)
            return shap.KernelExplainer(fn, shap.sample(X_bg, min(50, len(X_bg))))

    @staticmethod
    def _unify(raw: Any) -> np.ndarray:
        """Normalise any SHAP output to 2D (n_samples, n_features)."""
        if isinstance(raw, list):
            vals = raw[1] if len(raw) == 2 else raw[0]
            return np.array(vals)
        if hasattr(raw, "values"):
            v = raw.values
            if v.ndim == 3:
                return v[:, :, 1]
            return v
        arr = np.array(raw)
        if arr.ndim == 3:
            return arr[:, :, 1]
        return arr
