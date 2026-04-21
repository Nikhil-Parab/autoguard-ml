"""
autoguard.automl.registry
==========================

Plugin-based model registry. Register any sklearn-compatible model::

    from autoguard.automl.registry import ModelRegistry

    @ModelRegistry.register("my_svm")
    def build_svm(trial, problem_type):
        C = trial.suggest_float("C", 0.01, 100, log=True)
        from sklearn.svm import SVC, SVR
        return SVC(C=C) if problem_type == "classification" else SVR(C=C)
"""
from __future__ import annotations

from typing import Any, Callable

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

ModelFactory = Callable[[optuna.Trial, str], Any]
_REGISTRY: dict[str, ModelFactory] = {}


class ModelRegistry:
    """Central registry for AutoML model factories."""

    @classmethod
    def register(cls, name: str) -> Callable[[ModelFactory], ModelFactory]:
        """Decorator to register a model factory."""
        def decorator(fn: ModelFactory) -> ModelFactory:
            _REGISTRY[name] = fn
            return fn
        return decorator

    @classmethod
    def get(cls, name: str) -> ModelFactory:
        if name not in _REGISTRY:
            raise KeyError(f"Model '{name}' not registered. Available: {list(_REGISTRY)}")
        return _REGISTRY[name]

    @classmethod
    def available(cls) -> list[str]:
        return list(_REGISTRY.keys())


# ── Built-in models ───────────────────────────────────────────────────

@ModelRegistry.register("random_forest")
def _rf(trial: optuna.Trial, problem_type: str) -> Any:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    p = dict(
        n_estimators=trial.suggest_int("rf_n", 50, 400),
        max_depth=trial.suggest_int("rf_depth", 3, 20),
        min_samples_split=trial.suggest_int("rf_mss", 2, 20),
        min_samples_leaf=trial.suggest_int("rf_msl", 1, 10),
        max_features=trial.suggest_categorical("rf_feat", ["sqrt", "log2", 0.5]),
        n_jobs=-1, random_state=42,
    )
    Cls = RandomForestClassifier if problem_type == "classification" else RandomForestRegressor
    return Cls(**p)


@ModelRegistry.register("xgboost")
def _xgb(trial: optuna.Trial, problem_type: str) -> Any:
    from xgboost import XGBClassifier, XGBRegressor
    p = dict(
        n_estimators=trial.suggest_int("xgb_n", 50, 400),
        learning_rate=trial.suggest_float("xgb_lr", 1e-3, 0.3, log=True),
        max_depth=trial.suggest_int("xgb_depth", 3, 10),
        subsample=trial.suggest_float("xgb_sub", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("xgb_col", 0.5, 1.0),
        reg_alpha=trial.suggest_float("xgb_a", 1e-8, 1.0, log=True),
        reg_lambda=trial.suggest_float("xgb_l", 1e-8, 1.0, log=True),
        random_state=42, verbosity=0, tree_method="hist", device="cuda",
        eval_metric="logloss" if problem_type == "classification" else "rmse",
    )
    Cls = XGBClassifier if problem_type == "classification" else XGBRegressor
    return Cls(**p)


@ModelRegistry.register("lightgbm")
def _lgbm(trial: optuna.Trial, problem_type: str) -> Any:
    from lightgbm import LGBMClassifier, LGBMRegressor
    p = dict(
        n_estimators=trial.suggest_int("lgbm_n", 50, 400),
        learning_rate=trial.suggest_float("lgbm_lr", 1e-3, 0.3, log=True),
        max_depth=trial.suggest_int("lgbm_depth", 3, 12),
        num_leaves=trial.suggest_int("lgbm_leaves", 20, 150),
        subsample=trial.suggest_float("lgbm_sub", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("lgbm_col", 0.5, 1.0),
        random_state=42, verbose=-1, n_jobs=1, device="gpu",
    )
    Cls = LGBMClassifier if problem_type == "classification" else LGBMRegressor
    return Cls(**p)


@ModelRegistry.register("logistic_regression")
def _lr(trial: optuna.Trial, problem_type: str) -> Any:
    from sklearn.linear_model import LogisticRegression
    C = trial.suggest_float("lr_C", 1e-4, 100, log=True)
    solver = trial.suggest_categorical("lr_solver", ["lbfgs", "saga"])
    return LogisticRegression(C=C, solver=solver, max_iter=2000, random_state=42, n_jobs=-1)


@ModelRegistry.register("ridge")
def _ridge(trial: optuna.Trial, problem_type: str) -> Any:
    from sklearn.linear_model import Ridge, RidgeClassifier
    alpha = trial.suggest_float("ridge_alpha", 1e-3, 100, log=True)
    return RidgeClassifier(alpha=alpha) if problem_type == "classification" else Ridge(alpha=alpha)
