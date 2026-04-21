"""
autoguard.data.doctor
=====================

DatasetDoctor runs all quality checks and returns a structured report
with a risk score (0–100). Also generates matplotlib diagnostic plots.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from autoguard.core.config import DataConfig
from autoguard.core.exceptions import DataError
from autoguard.core.logging import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


class DatasetDoctor:
    """
    Comprehensive dataset diagnostic engine.

    Checks performed
    ----------------
    - Missing values (per column + overall rate)
    - Class imbalance (for categorical targets)
    - Feature correlation (redundant feature pairs)
    - Data leakage risk (features too correlated with target)
    - Outliers (IQR or Z-score method)
    - Skewness (highly skewed numeric columns)
    - Constant / near-constant columns

    Parameters
    ----------
    config : DataConfig, optional
    """

    def __init__(self, config: Optional[DataConfig] = None) -> None:
        self.config = config or DataConfig()
        self._issues: list[dict[str, Any]] = []

    def diagnose(self, df: pd.DataFrame, target: str) -> dict[str, Any]:
        """
        Run all diagnostic checks.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset including target column.
        target : str
            Target column name.

        Returns
        -------
        dict
            Structured report: issues, risk_score, per-check results.
        """
        if df.empty:
            raise DataError("Cannot diagnose an empty DataFrame.")
        if target not in df.columns:
            raise DataError(f"Target column '{target}' not found.")

        self._issues = []
        features = df.drop(columns=[target])
        y = df[target]

        logger.info(f"Diagnosing dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")

        results = {
            "missing_values":   self._check_missing(features),
            "class_imbalance":  self._check_imbalance(y),
            "correlations":     self._check_correlations(features),
            "leakage_risk":     self._check_leakage(features, y),
            "outliers":         self._check_outliers(features),
            "skewness":         self._check_skewness(features),
            "constant_columns": self._check_constants(features),
        }

        risk = self._compute_risk(results)

        report: dict[str, Any] = {
            "shape": {"rows": df.shape[0], "cols": df.shape[1]},
            "target": target,
            "risk_score": round(risk, 2),
            "risk_level": _risk_level(risk),
            "issues": self._issues,
            **results,
        }

        if self.config.generate_plots:
            self._generate_plots(features, y)

        self._print_report(report)
        return report

    # ── checks ──────────────────────────────────────────────────────

    def _check_missing(self, features: pd.DataFrame) -> dict[str, Any]:
        ratio = features.isnull().mean()
        cols_with_missing = ratio[ratio > 0].sort_values(ascending=False).to_dict()
        high = {c: v for c, v in cols_with_missing.items() if v > self.config.missing_threshold}

        for col, pct in high.items():
            self._issue("missing_values", "high", col,
                        f"⚠ {pct:.0%} missing values in column '{col}'")

        return {
            "status": "warning" if high else "ok",
            "total_missing_cells": int(features.isnull().sum().sum()),
            "columns_with_missing": {k: round(v, 4) for k, v in cols_with_missing.items()},
            "high_missing_columns": list(high.keys()),
        }

    def _check_imbalance(self, y: pd.Series) -> dict[str, Any]:
        is_numeric_continuous = y.dtype.kind in ("f", "i") and y.nunique() > 20
        if is_numeric_continuous:
            return {"status": "skipped", "reason": "Continuous target"}

        dist = y.value_counts(normalize=True)
        minority_ratio = float(dist.min())
        minority_class = dist.idxmin()
        imbalanced = minority_ratio < self.config.imbalance_ratio_threshold

        if imbalanced:
            self._issue("class_imbalance", "high", str(y.name),
                        f"⚠ Severe class imbalance detected: "
                        f"'{minority_class}' = {minority_ratio:.1%}")

        return {
            "status": "warning" if imbalanced else "ok",
            "n_classes": int(y.nunique()),
            "class_distribution": dist.round(4).to_dict(),
            "minority_class": str(minority_class),
            "minority_ratio": round(minority_ratio, 4),
            "is_imbalanced": imbalanced,
        }

    def _check_correlations(self, features: pd.DataFrame) -> dict[str, Any]:
        num = features.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            return {"status": "skipped", "reason": "Need ≥2 numeric features"}

        corr = num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        pairs: list[dict[str, Any]] = []
        for col in upper.columns:
            for row in upper.index:
                v = upper.loc[row, col]
                if pd.notna(v) and v > self.config.correlation_threshold:
                    pairs.append({"feature_a": row, "feature_b": col, "correlation": round(float(v), 4)})
                    self._issue("correlation", "medium", f"{row} ↔ {col}",
                                f"⚠ Highly correlated features: {row} ↔ {col} ({v:.3f})")

        return {
            "status": "warning" if pairs else "ok",
            "high_correlation_pairs": pairs,
            "n_flagged": len(pairs),
        }

    def _check_leakage(self, features: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        if y.dtype.kind not in ("f", "i"):
            y_enc = pd.Series(pd.Categorical(y).codes.astype(float), index=y.index)
        else:
            y_enc = y.astype(float)

        num = features.select_dtypes(include=[np.number])
        suspects: list[dict[str, Any]] = []

        for col in num.columns:
            vals = features[col].dropna()
            y_al = y_enc.loc[vals.index]
            if len(vals) < 10:
                continue
            corr = abs(float(np.corrcoef(vals, y_al)[0, 1]))
            if corr > self.config.leakage_correlation_threshold:
                suspects.append({"column": col, "correlation_with_target": round(corr, 4)})
                self._issue("leakage_risk", "critical", col,
                            f"⚠ Possible leakage in '{col}' — correlation with target = {corr:.3f}")

        return {
            "status": "critical" if suspects else "ok",
            "suspicious_columns": suspects,
        }

    def _check_outliers(self, features: pd.DataFrame) -> dict[str, Any]:
        num = features.select_dtypes(include=[np.number])
        flagged: dict[str, Any] = {}

        for col in num.columns:
            s = num[col].dropna()
            if len(s) < 10:
                continue

            if self.config.outlier_method == "zscore":
                z = np.abs(stats.zscore(s))
                n_out = int((z > self.config.outlier_threshold).sum())
            else:  # IQR
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                fence = self.config.outlier_threshold * iqr
                n_out = int(((s < q1 - fence) | (s > q3 + fence)).sum())

            pct = n_out / len(s)
            if pct > 0.05:
                flagged[col] = {"n_outliers": n_out, "pct": round(pct, 4)}
                self._issue("outliers", "medium", col,
                            f"⚠ {n_out} outliers ({pct:.1%}) in '{col}'")

        return {
            "status": "warning" if flagged else "ok",
            "method": self.config.outlier_method,
            "flagged_columns": flagged,
        }

    def _check_skewness(self, features: pd.DataFrame) -> dict[str, Any]:
        num = features.select_dtypes(include=[np.number])
        skewed: dict[str, float] = {}

        for col in num.columns:
            s = num[col].dropna()
            if len(s) < 10:
                continue
            sk = float(s.skew())
            if abs(sk) > self.config.skewness_threshold:
                skewed[col] = round(sk, 4)

        return {
            "status": "warning" if skewed else "ok",
            "skewed_columns": skewed,
            "threshold": self.config.skewness_threshold,
        }

    def _check_constants(self, features: pd.DataFrame) -> dict[str, Any]:
        const = [c for c in features.columns if features[c].nunique() <= 1]
        for c in const:
            self._issue("constant_column", "high", c,
                        f"⚠ Column '{c}' is constant (single unique value)")
        return {
            "status": "warning" if const else "ok",
            "constant_columns": const,
        }

    # ── risk scoring ─────────────────────────────────────────────────

    def _compute_risk(self, results: dict[str, Any]) -> float:
        score = 0.0
        score += min(len(results["missing_values"].get("high_missing_columns", [])) * 5, 20)
        if results["class_imbalance"].get("is_imbalanced"):
            score += 15
        score += min(results["correlations"].get("n_flagged", 0) * 3, 15)
        score += min(len(results["leakage_risk"].get("suspicious_columns", [])) * 15, 30)
        score += min(len(results["outliers"].get("flagged_columns", {})) * 2, 10)
        score += min(len(results["skewness"].get("skewed_columns", {})), 5)
        score += min(len(results["constant_columns"].get("constant_columns", [])) * 5, 5)
        return min(score, 100.0)

    # ── plotting ─────────────────────────────────────────────────────

    def _generate_plots(self, features: pd.DataFrame, y: pd.Series) -> None:
        d = Path(self.config.plot_output_dir)
        d.mkdir(parents=True, exist_ok=True)

        for fn in [
            (self._plot_missing, features),
            (self._plot_correlation, features),
            (self._plot_target_dist, y),
            (self._plot_skewness, features),
        ]:
            try:
                fn[0](*fn[1:] if isinstance(fn[1:], tuple) else (fn[1],))  # type: ignore
            except Exception as e:
                logger.debug(f"Plot {fn[0].__name__} failed: {e}")

        logger.info(f"Diagnostic plots → [cyan]{d}[/cyan]")

    def _plot_missing(self, features: pd.DataFrame) -> None:
        miss = features.isnull().mean().sort_values(ascending=False)
        miss = miss[miss > 0]
        if miss.empty:
            return
        fig, ax = plt.subplots(figsize=(10, max(4, len(miss) * 0.35)))
        miss.plot(kind="barh", ax=ax, color="#e74c3c")
        ax.axvline(self.config.missing_threshold, color="black", ls="--",
                   label=f"Threshold ({self.config.missing_threshold:.0%})")
        ax.set_title("Missing Values by Column", fontsize=13, fontweight="bold")
        ax.set_xlabel("Missing Ratio")
        ax.legend()
        plt.tight_layout()
        fig.savefig(Path(self.config.plot_output_dir) / "missing_values.png", dpi=150)
        plt.close(fig)

    def _plot_correlation(self, features: pd.DataFrame) -> None:
        num = features.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            return
        n = min(num.shape[1], 25)
        corr = num.iloc[:, :n].corr()
        fig, ax = plt.subplots(figsize=(max(8, n * 0.5), max(6, n * 0.45)))
        sns.heatmap(corr, annot=(n <= 12), fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, square=True, linewidths=0.3)
        ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(Path(self.config.plot_output_dir) / "correlation_matrix.png", dpi=150)
        plt.close(fig)

    def _plot_target_dist(self, y: pd.Series) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        if y.nunique() <= 20:
            counts = y.value_counts().sort_index()
            counts.plot(kind="bar", ax=ax, color="#3498db", edgecolor="white")
            ax.set_title(f"Target Distribution: {y.name}", fontsize=13, fontweight="bold")
            ax.set_ylabel("Count")
            plt.xticks(rotation=30)
        else:
            y.hist(bins=50, ax=ax, color="#3498db", edgecolor="white")
            ax.set_title(f"Target Distribution: {y.name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(Path(self.config.plot_output_dir) / "target_distribution.png", dpi=150)
        plt.close(fig)

    def _plot_skewness(self, features: pd.DataFrame) -> None:
        num = features.select_dtypes(include=[np.number])
        skew = num.skew().abs().sort_values(ascending=False).head(15)
        if skew.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        skew.plot(kind="bar", ax=ax, color="#e67e22")
        ax.axhline(self.config.skewness_threshold, color="red", ls="--",
                   label=f"Threshold ({self.config.skewness_threshold})")
        ax.set_title("Feature Skewness (|Skew|)", fontsize=13, fontweight="bold")
        ax.set_ylabel("|Skewness|")
        ax.legend()
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(Path(self.config.plot_output_dir) / "skewness.png", dpi=150)
        plt.close(fig)

    # ── helpers ──────────────────────────────────────────────────────

    def _issue(self, category: str, severity: str, loc: str, msg: str) -> None:
        self._issues.append({"category": category, "severity": severity,
                              "location": loc, "message": msg})

    def _print_report(self, report: dict[str, Any]) -> None:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        score = report["risk_score"]
        level = report["risk_level"]
        color = {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}.get(level, "white")

        console.print(Panel.fit(
            f"Rows: [bold]{report['shape']['rows']:,}[/bold]  "
            f"Cols: [bold]{report['shape']['cols']}[/bold]  "
            f"Target: [cyan]{report['target']}[/cyan]\n"
            f"Risk Score: [{color}]{score:.1f} / 100  ({level.upper()})[/{color}]",
            title="[bold]🩺 Dataset Doctor[/bold]",
            border_style=color,
        ))

        if not report["issues"]:
            console.print("[green]✓ No significant issues found.[/green]")
            return

        table = Table(title="Issues Detected", show_header=True, header_style="bold magenta")
        table.add_column("Severity", width=10)
        table.add_column("Message")
        sty = {"critical": "bold red", "high": "red", "medium": "yellow", "low": "cyan"}
        for issue in report["issues"]:
            sv = issue["severity"]
            table.add_row(
                f"[{sty.get(sv,'white')}]{sv.upper()}[/{sty.get(sv,'white')}]",
                issue["message"],
            )
        console.print(table)


def _risk_level(score: float) -> str:
    if score < 20:
        return "low"
    elif score < 50:
        return "medium"
    elif score < 75:
        return "high"
    return "critical"
