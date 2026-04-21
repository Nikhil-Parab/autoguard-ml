"""
autoguard.core.config
=====================
YAML-driven configuration for the full AutoGuard pipeline.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    missing_threshold: float = 0.30
    correlation_threshold: float = 0.95
    outlier_method: str = "iqr"          # "iqr" | "zscore"
    outlier_threshold: float = 3.0
    skewness_threshold: float = 1.0
    imbalance_ratio_threshold: float = 0.10
    leakage_correlation_threshold: float = 0.98
    generate_plots: bool = True
    plot_output_dir: str = "autoguard_output/plots"


@dataclass
class AutoMLConfig:
    models: list[str] = field(default_factory=lambda: [
        "random_forest", "xgboost", "lightgbm",
        "logistic_regression", "ridge",
    ])
    n_trials: int = 30
    cv_folds: int = 5
    scoring_classification: str = "f1_weighted"
    scoring_regression: str = "neg_root_mean_squared_error"
    timeout_per_model: int = 120
    feature_engineering: bool = True
    handle_imbalance: bool = True
    export_dir: str = "autoguard_output/models"
    n_jobs: int = -1
    random_state: int = 42


@dataclass
class ExplainConfig:
    max_samples: int = 300
    plot_top_n_features: int = 15
    output_dir: str = "autoguard_output/explain"


@dataclass
class DriftConfig:
    ks_pvalue_threshold: float = 0.05
    psi_threshold_warning: float = 0.10
    psi_threshold_alert: float = 0.20
    alert_email: Optional[str] = None
    log_dir: str = "autoguard_output/drift_logs"
    report_dir: str = "autoguard_output/drift_reports"


@dataclass
class AutoGuardConfig:
    """
    Master configuration for AutoGuard.

    Load from YAML::

        cfg = AutoGuardConfig.from_yaml("config.yaml")

    Or programmatically::

        cfg = AutoGuardConfig(automl=AutoMLConfig(n_trials=100))
    """
    target: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    automl: AutoMLConfig = field(default_factory=AutoMLConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    verbose: bool = True
    output_dir: str = "autoguard_output"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AutoGuardConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(
            target=raw.get("target", ""),
            data=DataConfig(**raw.get("data", {})),
            automl=AutoMLConfig(**raw.get("automl", {})),
            explain=ExplainConfig(**raw.get("explain", {})),
            drift=DriftConfig(**raw.get("drift", {})),
            verbose=raw.get("verbose", True),
            output_dir=raw.get("output_dir", "autoguard_output"),
        )

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False, sort_keys=False)

    def ensure_dirs(self) -> None:
        for d in [
            self.output_dir,
            self.data.plot_output_dir,
            self.automl.export_dir,
            self.explain.output_dir,
            self.drift.log_dir,
            self.drift.report_dir,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)
