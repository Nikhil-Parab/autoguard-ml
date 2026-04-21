"""
AutoGuard ML — Unified AutoML + Data Diagnosis + Drift Detection
================================================================

Quick start::

    from autoguard import AutoGuard

    ag = AutoGuard(target="label")
    ag.diagnose(df)
    df_clean = ag.auto_fix(df)
    ag.fit(df_clean)
    ag.report()
    ag.monitor(new_df)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "AutoGuard Contributors"
__license__ = "MIT"

from autoguard.core.guard import AutoGuard
from autoguard.core.config import AutoGuardConfig

__all__ = ["AutoGuard", "AutoGuardConfig", "__version__"]
