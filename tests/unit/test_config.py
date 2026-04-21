"""Unit tests for AutoGuardConfig."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from autoguard.core.config import AutoGuardConfig, DataConfig, AutoMLConfig


class TestAutoGuardConfig:
    def test_default_instantiation(self):
        cfg = AutoGuardConfig()
        assert cfg.output_dir == "autoguard_output"
        assert cfg.verbose is True

    def test_dataconfig_defaults(self):
        dc = DataConfig()
        assert 0 < dc.missing_threshold < 1
        assert 0 < dc.correlation_threshold < 1

    def test_automl_model_list_not_empty(self):
        cfg = AutoGuardConfig()
        assert len(cfg.automl.models) > 0

    def test_round_trip_yaml(self):
        cfg = AutoGuardConfig()
        cfg.automl.n_trials = 77
        cfg.data.missing_threshold = 0.42

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            cfg.to_yaml(path)
            loaded = AutoGuardConfig.from_yaml(path)

        assert loaded.automl.n_trials == 77
        assert abs(loaded.data.missing_threshold - 0.42) < 1e-6

    def test_from_yaml_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            AutoGuardConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_ensure_dirs_creates_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AutoGuardConfig(output_dir=str(Path(tmp) / "output"))
            cfg.data.plot_output_dir = str(Path(tmp) / "output" / "plots")
            cfg.automl.export_dir = str(Path(tmp) / "output" / "models")
            cfg.explain.output_dir = str(Path(tmp) / "output" / "explain")
            cfg.drift.log_dir = str(Path(tmp) / "output" / "drift_logs")
            cfg.drift.report_dir = str(Path(tmp) / "output" / "drift_reports")
            cfg.ensure_dirs()
            assert Path(cfg.output_dir).exists()
            assert Path(cfg.automl.export_dir).exists()
