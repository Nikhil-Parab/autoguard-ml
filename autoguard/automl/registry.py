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
        random_state=42, verbosity=0,
        eval_metric="logloss" if problem_type == "classification" else "rmse",
    )
    # Try GPU first, fall back to CPU gracefully
    try:
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "from xgboost import XGBClassifier; XGBClassifier(device='cuda').fit([[1,2],[3,4]],[0,1])"],
            capture_output=True, timeout=5
        )
        p["device"] = "cuda" if result.returncode == 0 else "cpu"
    except Exception:
        p["device"] = "cpu"
    p["tree_method"] = "hist"
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
        random_state=42, verbose=-1, n_jobs=-1,
    )
    # Try GPU, fall back silently to CPU
    try:
        import lightgbm as lgb
        test = LGBMClassifier(device="gpu", n_estimators=2, verbose=-1)
        import numpy as np
        test.fit(np.array([[1, 2], [3, 4]]), [0, 1])
        p["device"] = "gpu"
    except Exception:
        p["device"] = "cpu"
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


# ── New models ────────────────────────────────────────────────────────

@ModelRegistry.register("gradient_boosting")
def _gbm(trial: optuna.Trial, problem_type: str) -> Any:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    p = dict(
        n_estimators=trial.suggest_int("gbm_n", 50, 300),
        learning_rate=trial.suggest_float("gbm_lr", 1e-3, 0.3, log=True),
        max_depth=trial.suggest_int("gbm_depth", 2, 8),
        min_samples_split=trial.suggest_int("gbm_mss", 2, 20),
        min_samples_leaf=trial.suggest_int("gbm_msl", 1, 10),
        subsample=trial.suggest_float("gbm_sub", 0.5, 1.0),
        max_features=trial.suggest_categorical("gbm_feat", ["sqrt", "log2", None]),
        random_state=42,
    )
    Cls = GradientBoostingClassifier if problem_type == "classification" else GradientBoostingRegressor
    return Cls(**p)


@ModelRegistry.register("extra_trees")
def _et(trial: optuna.Trial, problem_type: str) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    p = dict(
        n_estimators=trial.suggest_int("et_n", 50, 400),
        max_depth=trial.suggest_int("et_depth", 3, 25),
        min_samples_split=trial.suggest_int("et_mss", 2, 20),
        min_samples_leaf=trial.suggest_int("et_msl", 1, 10),
        max_features=trial.suggest_categorical("et_feat", ["sqrt", "log2", 0.5]),
        n_jobs=-1, random_state=42,
    )
    Cls = ExtraTreesClassifier if problem_type == "classification" else ExtraTreesRegressor
    return Cls(**p)


@ModelRegistry.register("svm")
def _svm(trial: optuna.Trial, problem_type: str) -> Any:
    from sklearn.svm import SVC, SVR
    kernel = trial.suggest_categorical("svm_kernel", ["rbf", "poly", "sigmoid"])
    C = trial.suggest_float("svm_C", 1e-2, 100, log=True)
    gamma = trial.suggest_categorical("svm_gamma", ["scale", "auto"])
    if problem_type == "classification":
        return SVC(
            kernel=kernel, C=C, gamma=gamma,
            probability=True, random_state=42,
            class_weight="balanced",
        )
    else:
        epsilon = trial.suggest_float("svm_eps", 1e-3, 1.0, log=True)
        return SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)


@ModelRegistry.register("knn")
def _knn(trial: optuna.Trial, problem_type: str) -> Any:
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    p = dict(
        n_neighbors=trial.suggest_int("knn_k", 3, 30),
        weights=trial.suggest_categorical("knn_w", ["uniform", "distance"]),
        metric=trial.suggest_categorical("knn_metric", ["euclidean", "manhattan", "minkowski"]),
        n_jobs=-1,
    )
    Cls = KNeighborsClassifier if problem_type == "classification" else KNeighborsRegressor
    return Cls(**p)


@ModelRegistry.register("decision_tree")
def _dt(trial: optuna.Trial, problem_type: str) -> Any:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    p = dict(
        max_depth=trial.suggest_int("dt_depth", 2, 20),
        min_samples_split=trial.suggest_int("dt_mss", 2, 30),
        min_samples_leaf=trial.suggest_int("dt_msl", 1, 20),
        max_features=trial.suggest_categorical("dt_feat", ["sqrt", "log2", None]),
        random_state=42,
    )
    if problem_type == "classification":
        p["criterion"] = trial.suggest_categorical("dt_crit", ["gini", "entropy"])
        p["class_weight"] = "balanced"
        return DecisionTreeClassifier(**p)
    else:
        p["criterion"] = trial.suggest_categorical("dt_crit_r", ["squared_error", "friedman_mse", "absolute_error"])
        return DecisionTreeRegressor(**p)


@ModelRegistry.register("lasso")
def _lasso(trial: optuna.Trial, problem_type: str) -> Any:
    if problem_type == "classification":
        # L1-penalized logistic regression as the classification analog
        from sklearn.linear_model import LogisticRegression
        C = trial.suggest_float("lasso_C", 1e-4, 10, log=True)
        return LogisticRegression(
            penalty="l1", C=C, solver="saga",
            max_iter=2000, random_state=42, n_jobs=-1,
        )
    else:
        from sklearn.linear_model import Lasso
        alpha = trial.suggest_float("lasso_alpha", 1e-4, 10, log=True)
        max_iter = trial.suggest_int("lasso_iter", 500, 3000)
        return Lasso(alpha=alpha, max_iter=max_iter, random_state=42)


@ModelRegistry.register("catboost")
def _catboost(trial: optuna.Trial, problem_type: str) -> Any:
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError:
        raise ImportError(
            "CatBoost is not installed. Run: pip install catboost"
        )
    p = dict(
        iterations=trial.suggest_int("cb_iter", 50, 400),
        learning_rate=trial.suggest_float("cb_lr", 1e-3, 0.3, log=True),
        depth=trial.suggest_int("cb_depth", 3, 10),
        l2_leaf_reg=trial.suggest_float("cb_l2", 1e-3, 10, log=True),
        border_count=trial.suggest_int("cb_border", 32, 255),
        verbose=0, random_seed=42,
        task_type="CPU",  # safe default; set to "GPU" if available
    )
    Cls = CatBoostClassifier if problem_type == "classification" else CatBoostRegressor
    return Cls(**p)
