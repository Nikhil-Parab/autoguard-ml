"""
examples/quickstart.py
=======================

Complete AutoGuard ML demo — runs the full pipeline:
  1. Dataset Diagnosis
  2. Auto-Fix (cleaning)
  3. AutoML training
  4. Explainability (SHAP)
  5. Report generation
  6. Drift monitoring

Run:
    python examples/quickstart.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path so this works without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_demo() -> None:
    from rich.console import Console
    from rich.rule import Rule

    console = Console()

    # ── 0. Generate sample data ─────────────────────────────────────
    console.print(Rule("[bold green]AutoGuard ML — Full Demo[/bold green]"))
    console.print()

    from examples.generate_sample_data import make_churn_dataset, make_drifted_data
    train_df = make_churn_dataset(n=800, seed=0)
    new_data  = make_drifted_data(train_df, n=300)

    console.print(f"[bold]Train dataset[/bold]: {train_df.shape[0]:,} rows × {train_df.shape[1]} cols")
    console.print(f"[bold]Live dataset[/bold] : {new_data.shape[0]:,} rows (shifted distribution)")
    console.print()

    # ── 1. Initialise AutoGuard ──────────────────────────────────────
    from autoguard import AutoGuard, AutoGuardConfig
    from autoguard.core.config import AutoMLConfig

    cfg = AutoGuardConfig(
        automl=AutoMLConfig(
            n_trials=10,        # keep short for demo — use 50+ in production
            cv_folds=3,
            models=["random_forest", "xgboost", "logistic_regression"],
        )
    )
    cfg.data.generate_plots = True

    ag = AutoGuard(target="churn", config=cfg)

    # ── 2. Diagnose ──────────────────────────────────────────────────
    console.print(Rule("[bold]Step 1 — Dataset Diagnosis[/bold]"))
    diag = ag.diagnose(train_df)
    console.print(f"\nRisk score: [bold]{diag['risk_score']:.1f}/100[/bold]  "
                  f"({diag['risk_level']})\n")

    # ── 3. Auto-Fix ──────────────────────────────────────────────────
    console.print(Rule("[bold]Step 2 — Auto-Fix[/bold]"))
    df_clean = ag.auto_fix(train_df)
    console.print(f"Clean shape: {df_clean.shape}  "
                  f"(missing cells: {df_clean.isnull().sum().sum()})\n")

    # ── 4. Fit (AutoML) ──────────────────────────────────────────────
    console.print(Rule("[bold]Step 3 — AutoML Training[/bold]"))
    ag.fit(df_clean, target="churn")

    console.print(f"\nBest model : [bold green]{ag.best_model_name}[/bold green]")
    console.print(f"Problem    : {ag.problem_type}")
    console.print()

    # ── 5. Explain ───────────────────────────────────────────────────
    console.print(Rule("[bold]Step 4 — SHAP Explainability[/bold]"))
    X_val = df_clean.drop(columns=["churn"]).head(200)
    ag.explain(X=X_val, sample_index=0)
    console.print()

    # ── 6. Report ────────────────────────────────────────────────────
    console.print(Rule("[bold]Step 5 — Report[/bold]"))
    report = ag.report(output_path="autoguard_output/report.html")
    console.print()

    # ── 7. Save model ────────────────────────────────────────────────
    ag.save("autoguard_output/churn_model.pkl")
    console.print()

    # ── 8. Drift monitoring ──────────────────────────────────────────
    console.print(Rule("[bold]Step 6 — Drift Monitoring[/bold]"))
    drift_report = ag.monitor(new_data, save_report=True)
    console.print(
        f"\nDrift severity: [bold]{drift_report['overall_drift_severity']:.1f}/100[/bold]  "
        f"({drift_report['drift_level']})"
    )
    if drift_report["drifted_features"]:
        console.print(f"Drifted features: {drift_report['drifted_features']}")
    console.print()

    # ── 9. Load & predict ────────────────────────────────────────────
    console.print(Rule("[bold]Step 7 — Load & Predict[/bold]"))
    ag2 = AutoGuard.load("autoguard_output/churn_model.pkl")
    sample = df_clean.drop(columns=["churn"]).head(5)
    preds = ag2.predict(sample)
    proba = ag2.predict_proba(sample)
    console.print(f"Predictions     : {preds.tolist()}")
    console.print(f"Max probability : {proba.max(axis=1).round(3).tolist()}")
    console.print()

    console.print(Rule("[bold green]✓ Demo complete![/bold green]"))
    console.print("\nOutputs in: [cyan]autoguard_output/[/cyan]")
    console.print("  • report.html         — shareable HTML report")
    console.print("  • churn_model.pkl     — saved model")
    console.print("  • plots/              — diagnostic plots")
    console.print("  • explain/            — SHAP plots")
    console.print("  • drift_reports/      — drift JSON reports")
    console.print()
    console.print("Next steps:")
    console.print("  [cyan]autoguard serve autoguard_output/churn_model.pkl[/cyan]")
    console.print("  → opens REST API at http://localhost:8000/docs")


if __name__ == "__main__":
    run_demo()
