"""
autoguard.automl.getbest
=========================

GetBestEngine — fast pre-training model recommender.

Runs a lightweight benchmark on a **sample** of your dataset using every
registered model with a small number of Optuna trials, then combines the
benchmark score with dataset heuristics to surface the best algorithm to
use *before* committing to a full training run.

Usage (CLI)::

    autoguard getbest data.csv --target label
    autoguard getbest data.csv --target price --problem-type regression

Usage (Python)::

    from autoguard.automl.getbest import GetBestEngine
    engine = GetBestEngine()
    results = engine.run(df, target="label")
"""
from __future__ import annotations

import copy
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import optuna
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from autoguard.automl.registry import ModelRegistry
import autoguard.automl.registry  # noqa: F401 — registers built-ins
from autoguard.core.config import AutoMLConfig
from autoguard.core.logging import get_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def _make_console() -> Console:
    """Create a Rich Console that works on Windows (avoids legacy cp1252 renderer)."""
    import sys
    import io
    # Wrap stdout in a UTF-8 TextIOWrapper so Rich never hits the legacy
    # Windows console renderer, which can't encode non-cp1252 characters.
    try:
        utf8_stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
        return Console(file=utf8_stdout, highlight=False)
    except AttributeError:
        # stdout has no .buffer (e.g. pytest capture) — fall back
        return Console(highlight=False)


console = _make_console()


# ─────────────────────────────────────────────────────────────────────
# Heuristics
# ─────────────────────────────────────────────────────────────────────

_HEURISTIC_BOOSTS: dict[str, dict[str, float]] = {
    # model_key → {condition: score_boost}
    "random_forest":      {"large": 0.05, "many_features": 0.03},
    "xgboost":            {"large": 0.06, "many_features": 0.04, "moderate_imbalance": 0.03},
    "lightgbm":           {"large": 0.07, "high_cardinality_cat": 0.05, "many_features": 0.04},
    "catboost":           {"high_cardinality_cat": 0.08, "large": 0.04},
    "gradient_boosting":  {"moderate": 0.04, "low_noise": 0.03},
    "extra_trees":        {"large": 0.04, "many_features": 0.03, "noisy": 0.05},
    "svm":                {"small": 0.07, "few_features": 0.05},
    "knn":                {"small": 0.06, "low_dimensionality": 0.04},
    "decision_tree":      {"small": 0.03, "interpretability_needed": 0.05},
    "logistic_regression": {"small": 0.04, "few_features": 0.03, "linear": 0.06},
    "lasso":              {"sparse": 0.06, "regression_linear": 0.04},
    "ridge":              {"small": 0.03, "few_features": 0.04},
}

_MODEL_DESCRIPTIONS: dict[str, str] = {
    "random_forest":      "Fast ensemble, handles noise well, great default",
    "xgboost":            "High-accuracy gradient boosting, tabular champion",
    "lightgbm":           "Fast gradient boosting, great for large datasets",
    "catboost":           "Native categorical support, often no tuning needed",
    "gradient_boosting":  "sklearn GBM, robust but slower to train",
    "extra_trees":        "Extremely randomized trees, fast and robust",
    "svm":                "Kernel-based, strong on small/medium datasets",
    "knn":                "Instance-based, simple but effective on small data",
    "decision_tree":      "Highly interpretable single-tree baseline",
    "logistic_regression": "Fast linear model, great for linearly separable data",
    "lasso":              "Linear model with feature selection (L1 sparsity)",
    "ridge":              "Regularized linear model, excellent baseline",
}


def _compute_dataset_tags(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
) -> set[str]:
    """Return a set of string tags describing the dataset characteristics."""
    tags: set[str] = set()
    n_rows, n_cols = df.shape
    n_features = n_cols - 1  # exclude target

    # Size
    if n_rows < 1_000:
        tags.add("small")
    elif n_rows < 50_000:
        tags.add("moderate")
    else:
        tags.add("large")

    # Dimensionality
    if n_features <= 15:
        tags.add("few_features")
        tags.add("low_dimensionality")
    elif n_features > 100:
        tags.add("many_features")

    # Categorical cardinality
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if target in cat_cols:
        cat_cols = [c for c in cat_cols if c != target]
    if cat_cols:
        max_cardinality = max(df[c].nunique() for c in cat_cols)
        if max_cardinality > 20:
            tags.add("high_cardinality_cat")

    # Class imbalance (classification only)
    if problem_type == "classification" and target in df.columns:
        try:
            vc = df[target].value_counts(normalize=True)
            min_frac = vc.min()
            if min_frac < 0.10:
                tags.add("moderate_imbalance")
            elif min_frac < 0.02:
                tags.add("severe_imbalance")
        except Exception:
            pass

    # Sparsity: many zeros (useful for lasso)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols:
        num_cols = [c for c in num_cols if c != target]
    if num_cols:
        zero_frac = (df[num_cols] == 0).sum().sum() / max(df[num_cols].size, 1)
        if zero_frac > 0.30:
            tags.add("sparse")

    if problem_type == "regression":
        tags.add("regression_linear")

    return tags


def _heuristic_bonus(model_key: str, tags: set[str]) -> float:
    """Return a 0–1 bonus score based on dataset tags."""
    boosts = _HEURISTIC_BOOSTS.get(model_key, {})
    return sum(v for k, v in boosts.items() if k in tags)


# ─────────────────────────────────────────────────────────────────────
# GetBestEngine
# ─────────────────────────────────────────────────────────────────────

class GetBestEngine:
    """
    Fast model recommender — runs BEFORE full training.

    Parameters
    ----------
    config : AutoMLConfig, optional
        AutoML config (uses ``getbest_sample_frac`` and ``getbest_n_trials``).
    """

    def __init__(self, config: Optional[AutoMLConfig] = None) -> None:
        self.config = config or AutoMLConfig()

    def run(
        self,
        df: pd.DataFrame,
        target: str,
        problem_type: Optional[str] = None,
        sample_frac: Optional[float] = None,
        n_trials: Optional[int] = None,
        cv_folds: int = 3,
        models: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Run fast benchmark + heuristic analysis and return ranked results.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset including target column.
        target : str
            Name of the target column.
        problem_type : str, optional
            ``"classification"`` or ``"regression"``. Auto-detected if None.
        sample_frac : float, optional
            Fraction of data to use for benchmarking. Default from config (0.15).
        n_trials : int, optional
            Optuna trials per model. Default from config (5).
        cv_folds : int
            Number of cross-validation folds. Default 3.
        models : list[str], optional
            Subset of model keys to evaluate. Defaults to all registered models.

        Returns
        -------
        list[dict]
            Ranked list of result dicts, each with keys:
            rank, model, cv_score, heuristic_bonus, combined_score,
            metric, time_s, tags, recommended, description.
        """
        sample_frac = sample_frac or self.config.getbest_sample_frac
        n_trials = n_trials or self.config.getbest_n_trials

        # Auto-detect problem type
        if problem_type is None:
            problem_type = self._detect_problem_type(df[target])
        logger.info(f"[bold]GetBest[/bold] — problem type: [cyan]{problem_type}[/cyan]")

        # Sample
        n_sample = max(100, int(len(df) * sample_frac))
        n_sample = min(n_sample, len(df))
        df_sample = df.sample(n=n_sample, random_state=self.config.random_state)
        console.print(
            f"\n[bold cyan]GetBest[/bold cyan] sampling "
            f"[bold]{n_sample:,}[/bold] / [bold]{len(df):,}[/bold] rows "
            f"([dim]{sample_frac*100:.0f}%[/dim])"
        )

        # Dataset tags for heuristics
        tags = _compute_dataset_tags(df, target, problem_type)
        logger.info(f"Dataset tags: {tags}")

        # Preprocess sample
        X, y = self._preprocess(df_sample, target, problem_type)

        # Scoring metric
        scoring = (
            self.config.scoring_classification
            if problem_type == "classification"
            else self.config.scoring_regression
        )

        # Models to evaluate
        all_models = ModelRegistry.available()
        models_to_try = models if models else [m for m in self.config.models if m in all_models]
        # Also include catboost if registered but not in default list
        for m in all_models:
            if m not in models_to_try:
                models_to_try.append(m)

        console.print(
            f"[bold]Benchmarking {len(models_to_try)} models[/bold] | "
            f"Trials/model: {n_trials} | CV: {cv_folds}-fold | "
            f"Metric: [cyan]{scoring}[/cyan]\n"
        )

        rows: list[dict[str, Any]] = []

        from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

        cv = (
            StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config.random_state)
            if problem_type == "classification"
            else KFold(n_splits=cv_folds, shuffle=True, random_state=self.config.random_state)
        )

        for name in models_to_try:
            t0 = time.time()
            try:
                factory = ModelRegistry.get(name)
                best_ref: dict[str, Any] = {"score": -np.inf, "model": None}

                def _make_objective(
                    _factory=factory,
                    _cv=cv,
                    _X=X,
                    _y=y,
                    _scoring=scoring,
                    _problem_type=problem_type,
                    _best_ref=best_ref,
                ):
                    def objective(trial: optuna.Trial) -> float:
                        model = _factory(trial, _problem_type)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            scores = cross_val_score(
                                model, _X, _y, cv=_cv, scoring=_scoring, n_jobs=-1
                            )
                        s = float(np.mean(scores))
                        if s > _best_ref["score"]:
                            _best_ref["score"] = s
                            fitted = copy.deepcopy(model)
                            fitted.fit(_X, _y)
                            _best_ref["model"] = fitted
                        return s
                    return objective

                study = optuna.create_study(direction="maximize", study_name=name)
                study.optimize(
                    _make_objective(),
                    n_trials=n_trials,
                    timeout=30,
                    show_progress_bar=False,
                )

                cv_score = best_ref["score"]
                elapsed = time.time() - t0

                heuristic_bonus = _heuristic_bonus(name, tags)
                # Normalize cv_score to [0, 1] range for combining
                # For f1_weighted: already [0,1]; for neg_rmse: negative
                norm_cv = (
                    cv_score
                    if cv_score >= 0
                    else max(0.0, 1.0 + cv_score / max(abs(cv_score), 1.0))
                )
                combined = round(norm_cv + heuristic_bonus, 5)

                rows.append({
                    "model": name,
                    "cv_score": round(float(cv_score), 5),
                    "heuristic_bonus": round(heuristic_bonus, 4),
                    "combined_score": combined,
                    "metric": scoring,
                    "time_s": round(elapsed, 1),
                    "tags_matched": [k for k, v in _HEURISTIC_BOOSTS.get(name, {}).items() if k in tags],
                    "description": _MODEL_DESCRIPTIONS.get(name, ""),
                    "recommended": False,
                })
                console.print(
                    f"  [cyan]{name:25s}[/cyan]  cv=[bold]{cv_score:.4f}[/bold]  "
                    f"bonus=[dim]+{heuristic_bonus:.3f}[/dim]  "
                    f"combined=[bold green]{combined:.4f}[/bold green]  ({elapsed:.0f}s)"
                )
            except Exception as e:
                elapsed = time.time() - t0
                logger.warning(f"  [red]{name}[/red] failed: {e}")
                rows.append({
                    "model": name,
                    "cv_score": float("nan"),
                    "heuristic_bonus": 0.0,
                    "combined_score": float("nan"),
                    "metric": scoring,
                    "time_s": round(elapsed, 1),
                    "tags_matched": [],
                    "description": _MODEL_DESCRIPTIONS.get(name, ""),
                    "recommended": False,
                })

        # Sort by combined_score descending, NaN last
        rows.sort(key=lambda r: r["combined_score"] if not np.isnan(r["combined_score"]) else -np.inf, reverse=True)

        # Mark top 3 as recommended
        valid = [r for r in rows if not np.isnan(r["combined_score"])]
        for r in valid[:3]:
            r["recommended"] = True

        # Assign ranks
        for i, r in enumerate(rows):
            r["rank"] = i + 1

        return rows

    # ─────────────────────────────────────────────────────────────────

    def print_recommendations(
        self,
        results: list[dict[str, Any]],
        top_n: int = 12,
    ) -> None:
        """Pretty-print the getbest results table."""
        console.print()
        table = Table(
            title="AutoGuard - Model Recommendation Report",
            show_header=True,
            header_style="bold cyan",
            box=box.ROUNDED,
            show_lines=True,
        )
        table.add_column("Rank", style="dim", width=5, justify="center")
        table.add_column("Model", style="bold", min_width=22)
        table.add_column("CV Score", justify="right", min_width=9)
        table.add_column("Heuristic", justify="right", min_width=9)
        table.add_column("Combined", justify="right", min_width=9)
        table.add_column("Time", justify="right", min_width=6)
        table.add_column("Why Good?", min_width=20)

        for r in results[:top_n]:
            rank = r["rank"]
            is_best = rank == 1
            is_rec = r["recommended"]

            if is_best:
                badge = "[1] "
                row_style = "bold green"
            elif is_rec:
                badge = "[*] "
                row_style = "yellow"
            else:
                badge = "    "
                row_style = ""

            cv = f"{r['cv_score']:.5f}" if not np.isnan(r["cv_score"]) else "failed"
            bonus = f"+{r['heuristic_bonus']:.3f}"
            combined = f"{r['combined_score']:.5f}" if not np.isnan(r["combined_score"]) else "—"
            why = ", ".join(r.get("tags_matched", [])[:3]) or r.get("description", "")[:35]

            table.add_row(
                str(rank),
                badge + r["model"],
                cv,
                bonus,
                combined,
                f"{r['time_s']}s",
                why,
                style=row_style,
            )

        console.print(table)

        # Best model panel
        best = next((r for r in results if r["rank"] == 1 and not np.isnan(r["combined_score"])), None)
        if best:
            desc = _MODEL_DESCRIPTIONS.get(best["model"], "")
            console.print(
                Panel(
                    f"[bold green]BEST:[/bold green] [bold]{best['model']}[/bold]\n"
                    f"[dim]{desc}[/dim]\n\n"
                    f"[bold]Next step:[/bold]  "
                    f"[cyan]autoguard train data.csv --target <col>[/cyan]",
                    title="[bold]GetBest Result[/bold]",
                    border_style="green",
                    padding=(1, 2),
                )
            )

        # Top-3 summary
        recs = [r for r in results if r["recommended"] and not np.isnan(r["combined_score"])]
        if recs:
            top_str = "  ".join(
                f"[{'green' if r['rank']==1 else 'yellow'}]#{r['rank']} {r['model']}[/{'green' if r['rank']==1 else 'yellow'}]"
                for r in recs
            )
            console.print(f"[bold]Top recommendations:[/bold] {top_str}")
        console.print()

    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_problem_type(y: pd.Series) -> str:
        if y.dtype.kind in ("O", "b") or y.dtype.name == "category":
            return "classification"
        n_unique, n_total = y.nunique(), len(y)
        if n_unique <= 20 and n_unique / n_total < 0.05:
            return "classification"
        return "regression"

    def _preprocess(
        self,
        df: pd.DataFrame,
        target: str,
        problem_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Light preprocessing identical to AutoMLEngine."""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

        X_raw = df.drop(columns=[target])
        y_raw = df[target]

        num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()

        transformers = []
        if num_cols:
            transformers.append(("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
            ]), num_cols))
        if cat_cols:
            transformers.append(("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols))

        preprocessor = Pipeline([
            ("ct", ColumnTransformer(transformers, remainder="drop")),
        ])
        X_t = preprocessor.fit_transform(X_raw)

        y_arr = y_raw.to_numpy()
        if problem_type == "classification":
            le = LabelEncoder()
            y_arr = le.fit_transform(y_arr)

        return X_t, y_arr
