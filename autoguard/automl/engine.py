"""
autoguard.automl.engine
========================

AutoMLEngine orchestrates:
1. Problem-type detection
2. Preprocessing (sklearn Pipeline)
3. Per-model Optuna HPO + k-fold CV
4. Leaderboard + best model selection
5. Model export to .pkl
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from autoguard.automl.registry import ModelRegistry
import autoguard.automl.registry  # noqa: F401 – registers built-ins
from autoguard.core.config import AutoMLConfig
from autoguard.core.exceptions import AutoMLError
from autoguard.core.logging import get_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
logger = get_logger(__name__)


class AutoMLEngine:
    """
    AutoML engine: tries multiple models, tunes HPO, returns best.

    Attributes
    ----------
    problem_type : str
    best_model : sklearn estimator
    best_model_name : str
    leaderboard : pd.DataFrame
    feature_cols : list[str]
    X_transformed_df : pd.DataFrame   preprocessed features (for SHAP)
    """

    def __init__(self, config: Optional[AutoMLConfig] = None) -> None:
        self.config = config or AutoMLConfig()

        self.problem_type: str = ""
        self.best_model: Any = None
        self.best_model_name: str = ""
        self.leaderboard: Optional[pd.DataFrame] = None
        self.feature_cols: list[str] = []
        self.X_transformed_df: Optional[pd.DataFrame] = None

        self._le: Optional[LabelEncoder] = None
        self._preprocessor: Optional[Pipeline] = None
        self._rows: list[dict[str, Any]] = []

    # ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        target: str,
        problem_type: Optional[str] = None,
    ) -> "AutoMLEngine":
        X_raw = df.drop(columns=[target])
        y_raw = df[target]
        self.feature_cols = X_raw.columns.tolist()

        self.problem_type = problem_type or self._detect_problem_type(y_raw)
        logger.info(f"Problem type: [bold]{self.problem_type}[/bold]")

        # Build sklearn preprocessing pipeline
        X, y = self._preprocess(X_raw, y_raw)

        # Store preprocessed df for SHAP (keep column names)
        try:
            feat_names = self._get_feature_names()
            self.X_transformed_df = pd.DataFrame(
                X[:min(500, len(X))],
                columns=feat_names[:X.shape[1]] if len(feat_names) >= X.shape[1] else [f"f{i}" for i in range(X.shape[1])]
            )
        except Exception:
            self.X_transformed_df = pd.DataFrame(X[:min(500, len(X))])

        scoring = (
            self.config.scoring_classification
            if self.problem_type == "classification"
            else self.config.scoring_regression
        )

        models_to_try = [m for m in self.config.models if m in ModelRegistry.available()]
        if not models_to_try:
            raise AutoMLError(f"No valid models in config. Available: {ModelRegistry.available()}")

        logger.info(
            f"Models: {models_to_try} | Trials: {self.config.n_trials} | "
            f"CV: {self.config.cv_folds}-fold | Scoring: {scoring}"
        )

        best_score = -np.inf
        for name in models_to_try:
            try:
                t0 = time.time()
                score, model = self._optimize(name, X, y, scoring)
                elapsed = time.time() - t0
                self._rows.append({"rank": 0, "model": name,
                                   "cv_score": round(float(score), 5),
                                   "metric": scoring, "time_s": round(elapsed, 1)})
                logger.info(
                    f"  [cyan]{name:25s}[/cyan] → [bold]{score:.5f}[/bold] "
                    f"({elapsed:.0f}s)"
                )
                if score > best_score:
                    best_score = score
                    self.best_model = model
                    self.best_model_name = name
            except Exception as e:
                logger.warning(f"  [red]{name}[/red] failed: {e}")
                self._rows.append({"rank": 0, "model": name,
                                   "cv_score": float("nan"),
                                   "metric": scoring, "time_s": 0})

        if self.best_model is None:
            raise AutoMLError("All models failed. See logs above.")

        self.leaderboard = (
            pd.DataFrame(self._rows)
            .sort_values("cv_score", ascending=False)
            .reset_index(drop=True)
        )
        self.leaderboard["rank"] = self.leaderboard.index + 1

        # Save
        path = Path(self.config.export_dir) / "best_model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(self.best_model, path)
        logger.info(f"Best model saved → [cyan]{path}[/cyan]")

        return self

    # ─────────────────────────────────────────────────────────────────

    def _optimize(
        self, name: str, X: np.ndarray, y: np.ndarray, scoring: str
    ) -> tuple[float, Any]:
        factory = ModelRegistry.get(name)
        pt = self.problem_type
        rs = self.config.random_state

        cv = (
            StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=rs)
            if pt == "classification"
            else KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=rs)
        )

        best: dict[str, Any] = {"score": -np.inf, "model": None}

        def objective(trial: optuna.Trial) -> float:
            model = factory(trial, pt)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=self.config.n_jobs)
            s = float(np.mean(scores))
            if s > best["score"]:
                best["score"] = s
                fitted = copy.deepcopy(model)
                fitted.fit(X, y)
                best["model"] = fitted
            return s

        study = optuna.create_study(direction="maximize", study_name=name)
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_per_model,
            show_progress_bar=False,
        )
        return best["score"], best["model"]

    # ─────────────────────────────────────────────────────────────────

    def _preprocess(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

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

        self._preprocessor = Pipeline([
            ("ct", ColumnTransformer(transformers, remainder="drop")),
        ])

        X_t = self._preprocessor.fit_transform(X)

        y_arr = y.to_numpy()
        if self.problem_type == "classification":
            self._le = LabelEncoder()
            y_arr = self._le.fit_transform(y_arr)

            # Optional SMOTE
            if self.config.handle_imbalance:
                try:
                    from imblearn.over_sampling import SMOTE
                    counts = np.bincount(y_arr.astype(int))
                    if counts.min() >= 6:
                        sm = SMOTE(random_state=self.config.random_state)
                        X_t, y_arr = sm.fit_resample(X_t, y_arr)
                        logger.info(f"  SMOTE: {len(y_arr)} samples after resampling")
                except Exception as e:
                    logger.debug(f"SMOTE skipped: {e}")

        return X_t, y_arr

    def _get_feature_names(self) -> list[str]:
        ct = self._preprocessor.named_steps["ct"]
        names: list[str] = []
        for tname, transformer, cols in ct.transformers_:
            if tname == "num":
                names.extend(cols)
            elif tname == "cat":
                enc = transformer.named_steps["enc"]
                try:
                    names.extend(enc.get_feature_names_out(cols).tolist())
                except Exception:
                    names.extend([f"{c}_enc" for c in cols])
        return names

    @staticmethod
    def _detect_problem_type(y: pd.Series) -> str:
        if y.dtype.kind in ("O", "b") or y.dtype.name == "category":
            return "classification"
        n_unique, n_total = y.nunique(), len(y)
        if n_unique <= 20 and n_unique / n_total < 0.05:
            return "classification"
        return "regression"
