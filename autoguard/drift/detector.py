"""
autoguard.drift.detector
=========================

DriftDetector compares training reference data vs. incoming live data.

Methods per feature type
------------------------
Numeric   : KS test (p-value) + PSI (Population Stability Index)
Categorical: Chi-squared test + PSI

PSI thresholds
--------------
< 0.10  → No drift         (green)
0.10–0.20 → Moderate drift   (yellow)
> 0.20  → Severe drift      (red)

Overall severity score 0–100 is computed from per-feature findings.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from autoguard.core.config import DriftConfig
from autoguard.core.exceptions import DriftError
from autoguard.core.logging import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


class DriftDetector:
    """
    Detects data drift between a reference and a current dataset.

    Parameters
    ----------
    config : DriftConfig, optional
    """

    def __init__(self, config: Optional[DriftConfig] = None) -> None:
        self.config = config or DriftConfig()
        self._history: list[dict[str, Any]] = []

    def detect(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Run drift detection on all shared columns.

        Parameters
        ----------
        reference : pd.DataFrame
            Training / reference distribution.
        current : pd.DataFrame
            Live / incoming data.

        Returns
        -------
        dict
            Full drift report: per-feature results + overall severity.
        """
        common = [c for c in reference.columns if c in current.columns]
        if not common:
            raise DriftError("No common columns between reference and current data.")

        missing_in_cur = set(reference.columns) - set(current.columns)
        if missing_in_cur:
            logger.warning(f"Schema mismatch: {len(missing_in_cur)} cols missing in current: {missing_in_cur}")

        features: dict[str, Any] = {}
        drifted: list[str] = []

        for col in common:
            ref_s = reference[col].dropna()
            cur_s = current[col].dropna()
            if len(ref_s) < 10 or len(cur_s) < 10:
                continue

            # Safely detect type — avoid mixed int/str crash
            try:
                ref_s_num = pd.to_numeric(ref_s, errors='raise')
                cur_s_num = pd.to_numeric(cur_s, errors='raise')
                result = self._detect_numeric(col, ref_s_num, cur_s_num)
            except (ValueError, TypeError):
                result = self._detect_categorical(
                    col, ref_s.astype(str), cur_s.astype(str)
                )

            features[col] = result
            if result.get("drifted"):
                drifted.append(col)

        severity = self._overall_severity(features)
        report: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_size": int(len(reference)),
            "current_size": int(len(current)),
            "n_features_analyzed": len(features),
            "n_features_drifted": len(drifted),
            "drifted_features": drifted,
            "overall_drift_severity": round(severity, 2),
            "drift_level": _severity_label(severity),
            "features": features,
        }

        self._history.append({
            "timestamp": report["timestamp"],
            "severity": severity,
            "n_drifted": len(drifted),
        })

        if drifted:
            self._alert(report)

        return report

    def print_summary(self, report: dict[str, Any]) -> None:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        level = report["drift_level"]
        sev = report["overall_drift_severity"]
        color = {"none": "green", "low": "cyan", "moderate": "yellow",
                 "high": "red", "critical": "bold red"}.get(level, "white")

        console.print(Panel.fit(
            f"Ref N: [bold]{report['reference_size']:,}[/bold]  "
            f"Cur N: [bold]{report['current_size']:,}[/bold]\n"
            f"Features analyzed: {report['n_features_analyzed']}  "
            f"Drifted: [bold]{report['n_features_drifted']}[/bold]\n"
            f"Severity: [{color}]{sev:.1f}/100  ({level.upper()})[/{color}]",
            title="[bold]📡 Drift Monitor[/bold]",
            border_style=color,
        ))

        if report["drifted_features"]:
            table = Table(title="Drifted Features", show_header=True)
            table.add_column("Feature", style="cyan")
            table.add_column("Method")
            table.add_column("Statistic", justify="right")
            table.add_column("PSI", justify="right")
            table.add_column("Severity")
            sty = {"low": "cyan", "moderate": "yellow", "high": "red", "critical": "bold red"}

            for f in report["drifted_features"]:
                r = report["features"][f]
                sv = r.get("severity", "?")
                table.add_row(
                    f,
                    r.get("method", ""),
                    f"{r.get('statistic', 0):.4f}",
                    f"{r.get('psi', 0):.4f}",
                    f"[{sty.get(sv,'white')}]{sv.upper()}[/{sty.get(sv,'white')}]",
                )
            console.print(table)
        else:
            console.print("[green]✓ No significant drift detected.[/green]")

    def save_report(
        self,
        report: dict[str, Any],
        path: Optional[str | Path] = None,
    ) -> Path:
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(self.config.report_dir) / f"drift_{ts}.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Drift report → [cyan]{path}[/cyan]")
        return path

    def get_history(self) -> pd.DataFrame:
        """Return historical drift severity scores as a DataFrame."""
        return pd.DataFrame(self._history)

    # ── detection ────────────────────────────────────────────────────

    def _detect_numeric(
        self, col: str, ref: pd.Series, cur: pd.Series
    ) -> dict[str, Any]:
        ks_stat, ks_pval = stats.ks_2samp(ref.values, cur.values)
        psi = _compute_psi(ref, cur)
        ks_drift = ks_pval < self.config.ks_pvalue_threshold
        psi_drift = psi > self.config.psi_threshold_warning
        drifted = ks_drift or psi_drift

        if psi >= self.config.psi_threshold_alert:
            severity = "high" if psi < 0.5 else "critical"
        elif psi_drift or ks_drift:
            severity = "moderate"
        else:
            severity = "low" if ks_stat > 0.1 else "none"

        return {
            "method": "KS + PSI",
            "statistic": round(float(ks_stat), 5),
            "ks_pvalue": round(float(ks_pval), 5),
            "psi": round(float(psi), 5),
            "drifted": bool(drifted),
            "severity": severity,
            "ref_mean": round(float(ref.mean()), 4),
            "cur_mean": round(float(cur.mean()), 4),
            "ref_std": round(float(ref.std()), 4),
            "cur_std": round(float(cur.std()), 4),
        }

    def _detect_categorical(
        self, col: str, ref: pd.Series, cur: pd.Series
    ) -> dict[str, Any]:
        all_cats = list(set(ref.unique()) | set(cur.unique()))
        ref_cnt = np.array([ref.eq(c).sum() for c in all_cats], dtype=float)
        cur_cnt = np.array([cur.eq(c).sum() for c in all_cats], dtype=float)

        ref_p = ref_cnt / (ref_cnt.sum() + 1e-9)
        cur_cnt_safe = np.where(cur_cnt == 0, 1e-6, cur_cnt)
        expected = ref_p * cur_cnt.sum()
        expected = np.where(expected == 0, 1e-6, expected)

        chi2, pval = stats.chisquare(f_obs=cur_cnt_safe, f_exp=expected)
        cur_p = cur_cnt / (cur_cnt.sum() + 1e-9)
        psi = float(abs(np.sum((cur_p - ref_p) * np.log((cur_p + 1e-8) / (ref_p + 1e-8)))))

        drifted = pval < self.config.ks_pvalue_threshold or psi > self.config.psi_threshold_warning
        severity = ("high" if psi >= self.config.psi_threshold_alert
                    else "moderate" if drifted else "none")

        return {
            "method": "Chi-squared + PSI",
            "statistic": round(float(chi2), 5),
            "chi2_pvalue": round(float(pval), 5),
            "psi": round(float(psi), 5),
            "drifted": bool(drifted),
            "severity": severity,
            "ref_n_categories": int(ref.nunique()),
            "cur_n_categories": int(cur.nunique()),
        }

    # ── scoring + alerting ───────────────────────────────────────────

    def _overall_severity(self, features: dict[str, Any]) -> float:
        if not features:
            return 0.0
        sev_w = {"none": 0, "low": 10, "moderate": 40, "high": 70, "critical": 100}
        scores = [sev_w.get(r.get("severity", "none"), 0) for r in features.values()]
        pct_drifted = sum(1 for r in features.values() if r.get("drifted")) / len(features)
        return min(0.5 * float(np.mean(scores)) + 0.5 * pct_drifted * 100, 100.0)

    def _alert(self, report: dict[str, Any]) -> None:
        level = report["drift_level"]
        sev = report["overall_drift_severity"]
        nd = report["n_features_drifted"]
        feats = report["drifted_features"]

        logger.warning(
            f"🚨 [bold red]DRIFT ALERT[/bold red] — severity={sev:.1f}/100 ({level.upper()}) | "
            f"{nd} feature(s): {feats}"
        )
        log_path = Path(self.config.log_dir) / "drift_alerts.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "timestamp": report["timestamp"],
                "level": level,
                "severity": sev,
                "drifted_features": feats,
            }) + "\n")

        if self.config.alert_email:
            logger.info(f"[email stub] Alert would be sent to {self.config.alert_email}")


# ── helpers ───────────────────────────────────────────────────────────

def _compute_psi(ref: pd.Series, cur: pd.Series, n_bins: int = 10) -> float:
    breaks = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    breaks = np.unique(breaks)
    if len(breaks) < 2:
        return 0.0

    ref_cnt, _ = np.histogram(ref, bins=breaks)
    cur_cnt, _ = np.histogram(cur, bins=breaks)
    ref_p = ref_cnt / (ref_cnt.sum() + 1e-9)
    cur_p = cur_cnt / (cur_cnt.sum() + 1e-9)
    ref_p = np.where(ref_p == 0, 1e-8, ref_p)
    cur_p = np.where(cur_p == 0, 1e-8, cur_p)
    return float(abs(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p))))


def _severity_label(score: float) -> str:
    if score < 5:   return "none"
    if score < 20:  return "low"
    if score < 50:  return "moderate"
    if score < 75:  return "high"
    return "critical"
