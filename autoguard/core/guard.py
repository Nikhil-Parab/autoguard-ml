"""
autoguard.core.guard
====================

``AutoGuard`` is the single, clean entry-point for the entire pipeline.

Usage::

    from autoguard import AutoGuard

    ag = AutoGuard(target="label")
    ag.diagnose(df)
    df_clean = ag.auto_fix(df)
    ag.fit(df_clean)
    ag.explain()
    ag.report()
    ag.monitor(new_df)
    ag.save("model.pkl")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from autoguard.core.config import AutoGuardConfig
from autoguard.core.exceptions import NotFittedError, DataError
from autoguard.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


class AutoGuard:
    """Unified AutoML + Dataset Diagnosis + Drift Detection pipeline.

    Parameters
    ----------
    target : str
        Name of the target column in your DataFrame.
    config : AutoGuardConfig, optional
        Full configuration object.
    config_path : str or Path, optional
        Path to a YAML config file (overrides ``config``).

    Examples
    --------
    >>> ag = AutoGuard(target="label")
    >>> ag.diagnose(df)
    >>> df_clean = ag.auto_fix(df)
    >>> ag.fit(df_clean)
    >>> ag.report()
    >>> ag.monitor(new_df)
    """

    def __init__(
        self,
        target: str = "",
        config: Optional[AutoGuardConfig] = None,
        config_path: Optional[str | Path] = None,
    ) -> None:
        if config_path is not None:
            self.config = AutoGuardConfig.from_yaml(config_path)
        elif config is not None:
            self.config = config
        else:
            self.config = AutoGuardConfig()

        # Allow target to be set at constructor level OR via config YAML
        if target:
            self.config.target = target

        configure_logging(
            verbose=self.config.verbose,
            log_file=Path(self.config.output_dir) / "autoguard.log",
        )
        self.config.ensure_dirs()

        # Internal state
        self._target: str = self.config.target
        self._is_fitted: bool = False
        self._problem_type: str = ""
        self._feature_cols: list[str] = []
        self._train_df: Optional[pd.DataFrame] = None
        self._train_df_raw: Optional[pd.DataFrame] = None

        # Module outputs
        self.best_model: Any = None
        self.best_model_name: str = ""
        self.leaderboard: Optional[pd.DataFrame] = None
        self.diagnosis_report: Optional[dict[str, Any]] = None
        self._explainer: Any = None
        self._X_val: Optional[pd.DataFrame] = None

        logger.info(
            "[bold green]AutoGuard[/bold green] v0.1.0 initialized. "
            f"Output → [cyan]{self.config.output_dir}[/cyan]"
        )

    # ──────────────────────────────────────────────────────────────────
    # STEP 1 — DIAGNOSE
    # ──────────────────────────────────────────────────────────────────

    def diagnose(self, df: pd.DataFrame, target: str = "") -> dict[str, Any]:
        """
        Check your dataset for quality issues before training.

        Detects missing values, class imbalance, outliers, feature
        correlation, data leakage risk, and skewed distributions.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset including target column.
        target : str, optional
            Target column name. Overrides constructor value if provided.

        Returns
        -------
        dict
            Full diagnosis report with risk score (0–100) and per-issue details.

        Examples
        --------
        >>> ag.diagnose(df)
        # prints colored report + returns dict
        """
        target = target or self._target
        if not target:
            raise DataError("No target column specified. Pass target= to diagnose() or AutoGuard().")

        self._target = target

        from autoguard.data.doctor import DatasetDoctor
        doctor = DatasetDoctor(config=self.config.data)
        self.diagnosis_report = doctor.diagnose(df, target=target)
        return self.diagnosis_report

    # ──────────────────────────────────────────────────────────────────
    # STEP 2 — AUTO-FIX
    # ──────────────────────────────────────────────────────────────────

    def auto_fix(self, df: pd.DataFrame, target: str = "") -> pd.DataFrame:
        """
        Automatically clean and preprocess the dataset.

        Operations performed:
        - Fill missing values (median for numeric, mode for categorical)
        - Remove constant columns
        - Encode categorical variables
        - Cap extreme outliers
        - Normalize skewed numeric features (log1p)

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataset.
        target : str, optional
            Target column name (excluded from transformations).

        Returns
        -------
        pd.DataFrame
            Cleaned dataset, ready for ``fit()``.

        Examples
        --------
        >>> df_clean = ag.auto_fix(df)
        """
        target = target or self._target
        if not target and target in df.columns:
            raise DataError("Specify a target column.")

        from autoguard.data.cleaner import AutoCleaner
        cleaner = AutoCleaner(config=self.config.data)
        df_clean = cleaner.fit_transform(df, target=target)
        logger.info("[bold green]✓ auto_fix complete.[/bold green] Clean dataset ready.")
        return df_clean

    # ──────────────────────────────────────────────────────────────────
    # STEP 3 — FIT (AutoML)
    # ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        target: str = "",
        problem_type: Optional[str] = None,
    ) -> "AutoGuard":
        """
        Train the best model automatically using AutoML.

        Detects problem type, tries multiple models with HPO (Optuna),
        cross-validates, selects best model, builds SHAP explainer.

        Parameters
        ----------
        df : pd.DataFrame
            Training dataset (run ``auto_fix`` first for best results).
        target : str, optional
            Target column. Overrides constructor value if provided.
        problem_type : str, optional
            Force ``"classification"`` or ``"regression"``.

        Returns
        -------
        AutoGuard
            Self, for method chaining.

        Examples
        --------
        >>> ag.fit(df_clean)
        """
        target = target or self._target
        if not target:
            raise DataError("No target column specified.")
        self._target = target

        if target not in df.columns:
            raise DataError(f"Target column '{target}' not found in DataFrame.")

        self._train_df_raw = df.copy()
        logger.info("=" * 62)
        logger.info(f"[bold]AutoGuard.fit()[/bold] — shape={df.shape}, target=[cyan]{target}[/cyan]")

        X_raw = df.drop(columns=[target])
        y_raw = df[target]
        self._feature_cols = X_raw.columns.tolist()
        n_val = min(500, len(X_raw))
        self._X_val = X_raw.sample(n=n_val, random_state=self.config.automl.random_state)

        from autoguard.automl.engine import AutoMLEngine
        engine = AutoMLEngine(config=self.config.automl)
        engine.fit(df, target=target, problem_type=problem_type)

        self.best_model = engine.best_model
        self.best_model_name = engine.best_model_name
        self.leaderboard = engine.leaderboard
        self._problem_type = engine.problem_type
        self._train_df = engine.X_transformed_df  # preprocessed version

        # SHAP explainer
        logger.info("Building SHAP explainer...")
        from autoguard.explain.shap_explainer import ShapExplainer
        self._explainer = ShapExplainer(
            model=self.best_model,
            model_name=self.best_model_name,
            config=self.config.explain,
            problem_type=self._problem_type,
        )
        try:
            self._explainer.fit(self._X_val)
        except Exception as e:
            logger.warning(f"SHAP explainer unavailable: {e}. Run `pip install shap` to enable.")
            self._explainer = None

        self._is_fitted = True
        logger.info("[bold green]✓ AutoGuard.fit() complete.[/bold green]")
        self._print_leaderboard()
        return self

    # ──────────────────────────────────────────────────────────────────
    # STEP 4 — EXPLAIN
    # ──────────────────────────────────────────────────────────────────

    def explain(
        self,
        X: Optional[pd.DataFrame] = None,
        sample_index: int = 0,
    ) -> None:
        """
        Generate SHAP feature importance explanations.

        Produces:
        - Global feature importance bar chart
        - SHAP summary (beeswarm) plot
        - Local single-prediction waterfall chart

        Parameters
        ----------
        X : pd.DataFrame, optional
            Data to explain. Uses validation slice from training if None.
        sample_index : int
            Row index for the local (single-prediction) explanation.

        Examples
        --------
        >>> ag.explain()
        >>> ag.explain(X=new_df, sample_index=5)
        """
        self._require_fitted("explain")
        X = X if X is not None else self._X_val

        if self._explainer is None:
            from autoguard.core.exceptions import ExplainError
            raise ExplainError(
                "SHAP explainer is not available. Install shap:  pip install shap"
            )

        logger.info("Generating SHAP explanations...")
        self._explainer.plot_global(X)
        self._explainer.plot_local(X, index=sample_index)
        logger.info(
            f"SHAP plots saved → [cyan]{self.config.explain.output_dir}[/cyan]"
        )

    # ──────────────────────────────────────────────────────────────────
    # STEP 5 — MONITOR (Drift Detection)
    # ──────────────────────────────────────────────────────────────────

    def monitor(
        self,
        new_data: pd.DataFrame,
        save_report: bool = True,
    ) -> dict[str, Any]:
        """
        Detect data drift between training distribution and new data.

        Uses KS test (numeric), Chi-squared (categorical), and PSI
        to flag features with distribution shift.

        Parameters
        ----------
        new_data : pd.DataFrame
            Incoming / live data batch to compare against training data.
        save_report : bool
            If True, writes a JSON drift report to disk.

        Returns
        -------
        dict
            Per-feature drift results + overall severity score (0–100).

        Examples
        --------
        >>> ag.monitor(new_df)
        >>> for batch in stream:
        ...     ag.monitor(batch)
        """
        self._require_fitted("monitor")

        logger.info("=" * 62)
        logger.info("[bold]Drift Monitor[/bold] — comparing training vs incoming data")

        ref = self._train_df_raw[self._feature_cols]
        cur = new_data[[c for c in self._feature_cols if c in new_data.columns]]

        from autoguard.drift.detector import DriftDetector
        detector = DriftDetector(config=self.config.drift)
        report = detector.detect(reference=ref, current=cur)
        detector.print_summary(report)

        if save_report:
            detector.save_report(report)

        return report

    # ──────────────────────────────────────────────────────────────────
    # REPORT
    # ──────────────────────────────────────────────────────────────────

    def report(
        self,
        output_path: Optional[str | Path] = None,
        format: str = "html",
    ) -> dict[str, Any]:
        """
        Generate the full AutoGuard pipeline report.

        Includes dataset diagnosis, model leaderboard, best model info,
        and a shareable HTML file.

        Parameters
        ----------
        output_path : str or Path, optional
            Where to save the report.
            Defaults to ``autoguard_output/report.html``.
        format : str
            ``"html"`` (default) or ``"json"``.

        Returns
        -------
        dict
            Report data dictionary.

        Examples
        --------
        >>> ag.report()
        >>> ag.report(output_path="my_report.html")
        """
        self._require_fitted("report")

        data: dict[str, Any] = {
            "version": "0.1.0",
            "problem_type": self._problem_type,
            "target": self._target,
            "best_model": self.best_model_name,
            "features": self._feature_cols,
        }

        if self.diagnosis_report:
            data["diagnosis"] = self.diagnosis_report

        if self.leaderboard is not None:
            data["leaderboard"] = self.leaderboard.to_dict(orient="records")

        if output_path is None:
            ext = "html" if format == "html" else "json"
            output_path = Path(self.config.output_dir) / f"report.{ext}"

        output_path = Path(output_path)

        if format == "html":
            from autoguard.utils.report import HTMLReportGenerator
            gen = HTMLReportGenerator()
            gen.save(data, output_path)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        logger.info(f"Report saved → [cyan]{output_path}[/cyan]")
        self._print_summary(data)
        return data

    # ──────────────────────────────────────────────────────────────────
    # PREDICT
    # ──────────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> Any:
        """
        Run inference with the best trained model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data (same schema as training features).

        Returns
        -------
        np.ndarray
            Predictions.
        """
        self._require_fitted("predict")
        cols = [c for c in self._feature_cols if c in X.columns]
        return self.best_model.predict(X[cols])

    def predict_proba(self, X: pd.DataFrame) -> Any:
        """
        Probabilistic inference (classification only).

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray
            Class probability matrix, shape (n_samples, n_classes).
        """
        self._require_fitted("predict_proba")
        if self._problem_type != "classification":
            raise NotFittedError("predict_proba only available for classification.")
        cols = [c for c in self._feature_cols if c in X.columns]
        return self.best_model.predict_proba(X[cols])

    # ──────────────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ──────────────────────────────────────────────────────────────────

    def save(self, path: str | Path = "model.pkl") -> Path:
        """
        Save the fitted AutoGuard instance to disk.

        Parameters
        ----------
        path : str or Path
            Target ``.pkl`` file path.

        Returns
        -------
        Path

        Examples
        --------
        >>> ag.save("model.pkl")
        """
        self._require_fitted("save")
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"AutoGuard instance saved → [cyan]{path}[/cyan]")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "AutoGuard":
        """
        Load a previously saved AutoGuard instance.

        Parameters
        ----------
        path : str or Path
            Path to the ``.pkl`` file produced by ``save()``.

        Returns
        -------
        AutoGuard

        Examples
        --------
        >>> ag = AutoGuard.load("model.pkl")
        >>> ag.predict(new_df)
        """
        import joblib
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        instance = joblib.load(path)
        logger.info(f"AutoGuard instance loaded from [cyan]{path}[/cyan]")
        return instance

    # ──────────────────────────────────────────────────────────────────
    # PROPERTIES
    # ──────────────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        """Whether the pipeline has been trained."""
        return self._is_fitted

    @property
    def problem_type(self) -> str:
        """Detected problem type: 'classification' or 'regression'."""
        return self._problem_type

    @property
    def feature_cols(self) -> list[str]:
        """Feature column names used during training."""
        return self._feature_cols

    # ──────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────────────

    def _require_fitted(self, method: str = "") -> None:
        if not self._is_fitted:
            raise NotFittedError(
                f"AutoGuard is not fitted yet. Call .fit(df) before .{method}()."
            )

    def _print_leaderboard(self) -> None:
        from rich.console import Console
        from rich.table import Table

        if self.leaderboard is None:
            return

        console = Console()
        table = Table(title="🏆 AutoML Leaderboard", show_header=True, header_style="bold cyan")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Model", style="bold")
        table.add_column("CV Score", justify="right")
        table.add_column("Metric")

        for i, row in self.leaderboard.iterrows():
            marker = "⭐ " if row["model"] == self.best_model_name else "   "
            table.add_row(
                str(i),
                marker + row["model"],
                f"{row['cv_score']:.5f}",
                row["metric"],
            )
        console.print(table)

    def _print_summary(self, data: dict[str, Any]) -> None:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        lines = [
            f"[bold]Problem[/bold]    : {data['problem_type']}",
            f"[bold]Target[/bold]     : {data['target']}",
            f"[bold]Best model[/bold] : [green]{data['best_model']}[/green]",
        ]
        if "diagnosis" in data:
            score = data["diagnosis"]["risk_score"]
            level = data["diagnosis"]["risk_level"]
            color = {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}.get(level, "white")
            lines.append(f"[bold]Data risk[/bold]  : [{color}]{score:.1f}/100 ({level})[/{color}]")

        console.print(Panel("\n".join(lines), title="[bold]AutoGuard Report[/bold]", border_style="green"))
