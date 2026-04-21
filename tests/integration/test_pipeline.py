"""
Integration tests: full AutoGuard pipeline end-to-end.

These tests run the real fit → report → monitor chain on small
synthetic datasets and assert on structure (not exact numeric values).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoguard import AutoGuard, AutoGuardConfig
from autoguard.core.config import AutoMLConfig
from autoguard.core.exceptions import NotFittedError


@pytest.fixture(scope="module")
def clf_df() -> pd.DataFrame:
    """Small classification dataset."""
    rng = np.random.default_rng(0)
    n = 300
    age    = rng.integers(20, 70, n).astype(float)
    income = rng.normal(50000, 12000, n)
    city   = rng.choice(["NYC", "LA", "Chicago"], n)
    label  = (age > 40).astype(int)
    return pd.DataFrame({"age": age, "income": income, "city": city, "label": label})


@pytest.fixture(scope="module")
def reg_df() -> pd.DataFrame:
    """Small regression dataset."""
    rng = np.random.default_rng(1)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    y  = 3 * x1 + 2 * x2 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "price": y})


@pytest.fixture
def fast_config() -> AutoGuardConfig:
    cfg = AutoGuardConfig()
    cfg.automl.n_trials = 3
    cfg.automl.cv_folds = 2
    cfg.automl.timeout_per_model = 30
    cfg.automl.models = ["random_forest", "logistic_regression"]
    cfg.data.generate_plots = False
    cfg.explain.max_samples = 50
    return cfg


class TestClassificationPipeline:
    def test_fit_returns_self(self, clf_df, fast_config):
        ag = AutoGuard(target="label", config=fast_config)
        result = ag.fit(clf_df)
        assert result is ag

    def test_is_fitted_after_fit(self, clf_df, fast_config):
        ag = AutoGuard(target="label", config=fast_config)
        ag.fit(clf_df)
        assert ag.is_fitted

    def test_problem_type_detected(self, clf_df, fast_config):
        ag = AutoGuard(target="label", config=fast_config)
        ag.fit(clf_df)
        assert ag.problem_type == "classification"

    def test_leaderboard_populated(self, clf_df, fast_config):
        ag = AutoGuard(target="label", config=fast_config)
        ag.fit(clf_df)
        assert ag.leaderboard is not None
        assert len(ag.leaderboard) > 0
        assert "model" in ag.leaderboard.columns
        assert "cv_score" in ag.leaderboard.columns

    def test_best_model_set(self, clf_df, fast_config):
        ag = AutoGuard(target="label", config=fast_config)
        ag.fit(clf_df)
        assert ag.best_model is not None
        assert ag.best_model_name != ""

    def test_predict_returns_array(self, clf_df, fast_config):
        ag = AutoGuard(target="label", config=fast_config)
        ag.fit(clf_df)
        X = clf_df.drop(columns=["label"]).head(10)
        preds = ag.predict(X)
        assert len(preds) == 10

    def test_predict_proba_shape(self, clf_df, fast_config):
        ag = AutoGuard(target="label", config=fast_config)
        ag.fit(clf_df)
        X = clf_df.drop(columns=["label"]).head(10)
        proba = ag.predict_proba(X)
        assert proba.shape[0] == 10
        assert proba.shape[1] >= 2


class TestRegressionPipeline:
    def test_problem_type_regression(self, reg_df, fast_config):
        cfg = AutoGuardConfig()
        cfg.automl.n_trials = 3
        cfg.automl.cv_folds = 2
        cfg.automl.models = ["random_forest", "ridge"]
        cfg.data.generate_plots = False
        cfg.explain.max_samples = 50
        ag = AutoGuard(target="price", config=cfg)
        ag.fit(reg_df)
        assert ag.problem_type == "regression"


class TestDiagnoseThenFit:
    def test_diagnose_before_fit(self, clf_df, fast_config):
        ag = AutoGuard(target="label", config=fast_config)
        report = ag.diagnose(clf_df)
        assert "risk_score" in report
        ag.fit(clf_df)
        assert ag.is_fitted

    def test_auto_fix_then_fit(self, clf_df, fast_config):
        ag = AutoGuard(target="label", config=fast_config)
        df_clean = ag.auto_fix(clf_df)
        assert isinstance(df_clean, pd.DataFrame)
        assert df_clean.isnull().sum().sum() == 0


class TestMonitor:
    def test_monitor_returns_dict(self, clf_df, fast_config):
        ag = AutoGuard(target="label", config=fast_config)
        ag.fit(clf_df)
        new_data = clf_df.drop(columns=["label"]).head(50)
        report = ag.monitor(new_data, save_report=False)
        assert "overall_drift_severity" in report
        assert "drift_level" in report
        assert "drifted_features" in report

    def test_drift_severity_bounded(self, clf_df, fast_config):
        ag = AutoGuard(target="label", config=fast_config)
        ag.fit(clf_df)
        new_data = clf_df.drop(columns=["label"]).head(50)
        report = ag.monitor(new_data, save_report=False)
        assert 0 <= report["overall_drift_severity"] <= 100


class TestSaveLoad:
    def test_save_and_load(self, clf_df, fast_config):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            ag = AutoGuard(target="label", config=fast_config)
            ag.fit(clf_df)
            ag.save(path)
            assert path.exists()

            ag2 = AutoGuard.load(path)
            assert ag2.is_fitted
            assert ag2.best_model_name == ag.best_model_name

    def test_loaded_model_can_predict(self, clf_df, fast_config):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            ag = AutoGuard(target="label", config=fast_config)
            ag.fit(clf_df)
            ag.save(path)

            ag2 = AutoGuard.load(path)
            X = clf_df.drop(columns=["label"]).head(5)
            preds = ag2.predict(X)
            assert len(preds) == 5


class TestReport:
    def test_report_returns_dict(self, clf_df, fast_config):
        with tempfile.TemporaryDirectory() as tmp:
            fast_config.output_dir = str(Path(tmp) / "out")
            fast_config.ensure_dirs()
            ag = AutoGuard(target="label", config=fast_config)
            ag.diagnose(clf_df)
            ag.fit(clf_df)
            report = ag.report(output_path=Path(tmp) / "report.html")
            assert "best_model" in report
            assert "leaderboard" in report

    def test_report_html_file_created(self, clf_df, fast_config):
        with tempfile.TemporaryDirectory() as tmp:
            fast_config.output_dir = str(Path(tmp) / "out")
            fast_config.ensure_dirs()
            ag = AutoGuard(target="label", config=fast_config)
            ag.fit(clf_df)
            out = Path(tmp) / "report.html"
            ag.report(output_path=out, format="html")
            assert out.exists()
            content = out.read_text()
            assert "AutoGuard" in content


class TestNotFittedErrors:
    def test_predict_raises_before_fit(self, clf_df):
        ag = AutoGuard(target="label")
        with pytest.raises(NotFittedError):
            ag.predict(clf_df)

    def test_monitor_raises_before_fit(self, clf_df):
        ag = AutoGuard(target="label")
        with pytest.raises(NotFittedError):
            ag.monitor(clf_df)

    def test_report_raises_before_fit(self):
        ag = AutoGuard(target="label")
        with pytest.raises(NotFittedError):
            ag.report()
