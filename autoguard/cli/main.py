"""
autoguard.cli.main
==================

Command-line interface for AutoGuard ML.

Commands
--------
autoguard train    data.csv --target label
autoguard diagnose data.csv --target label
autoguard fix      data.csv --target label --output clean.csv
autoguard monitor  new.csv  --baseline train.csv
autoguard explain  --model  model.pkl --data data.csv
autoguard serve    model.pkl --host 0.0.0.0 --port 8000
autoguard report   --model  model.pkl --output report.html
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import click
from autoguard.core.logging import get_logger, configure_logging, _make_utf8_console

if TYPE_CHECKING:
    import pandas as pd
    from autoguard import AutoGuard

console = _make_utf8_console()

# ─────────────────────────────────────────────────────────────────────
# ROOT GROUP
# ─────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.2.2", prog_name="autoguard")
def cli() -> None:
    """
    \b
    AutoGuard ML  v0.2.0
    AutoML + Diagnosis + Drift Detection

    Run `autoguard COMMAND --help` for details on any command.
    """


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> pd.DataFrame:
    import pandas as pd
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        sys.exit(1)
    return pd.read_csv(p)


def _load_ag(model_path: str) -> AutoGuard:
    from autoguard import AutoGuard
    return AutoGuard.load(model_path)


def _make_ag(target: str, config: Optional[str]) -> AutoGuard:
    from autoguard import AutoGuard, AutoGuardConfig
    if config:
        cfg = AutoGuardConfig.from_yaml(config)
        cfg.target = target or cfg.target
        return AutoGuard(config=cfg)
    return AutoGuard(target=target)


# ─────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────

@cli.command("train")
@click.argument("data", type=click.Path(exists=True))
@click.option("--target", "-t", required=True, help="Target column name")
@click.option("--output", "-o", default="model.pkl", show_default=True,
              help="Output path for saved model (.pkl)")
@click.option("--config", "-c", default=None, help="Path to YAML config file")
@click.option("--no-diagnose", is_flag=True, default=False,
              help="Skip dataset diagnosis step")
@click.option("--no-fix", is_flag=True, default=False,
              help="Skip auto-cleaning step")
@click.option("--report", "gen_report", is_flag=True, default=False,
              help="Generate HTML report after training")
@click.option("--problem-type", default=None,
              type=click.Choice(["classification", "regression"]),
              help="Override auto-detected problem type")
def cmd_train(
    data: str,
    target: str,
    output: str,
    config: Optional[str],
    no_diagnose: bool,
    no_fix: bool,
    gen_report: bool,
    problem_type: Optional[str],
) -> None:
    """
    Train an AutoML model on DATA.

    \b
    Examples:
      autoguard train data.csv --target label
      autoguard train data.csv --target price --problem-type regression
      autoguard train data.csv --target label --config myconfig.yaml --report
    """
    df = _load_csv(data)
    ag = _make_ag(target, config)

    console.print(f"\n[bold cyan]Dataset:[/bold cyan] {data}  "
                  f"({df.shape[0]:,} rows × {df.shape[1]} cols)")

    if not no_diagnose:
        console.rule("[bold]Step 1 / 3 — Dataset Diagnosis[/bold]")
        ag.diagnose(df)

    if not no_fix:
        console.rule("[bold]Step 2 / 3 — Auto-Fix (cleaning)[/bold]")
        df = ag.auto_fix(df)

    console.rule("[bold]Step 3 / 3 — AutoML Training[/bold]")
    ag.fit(df, target=target, problem_type=problem_type)

    ag.save(output)
    console.print(f"\n[bold green]✓ Model saved:[/bold green] [cyan]{output}[/cyan]")

    if gen_report:
        report_path = Path(output).with_suffix(".html")
        ag.report(output_path=report_path)
        console.print(f"[bold green]✓ Report saved:[/bold green] [cyan]{report_path}[/cyan]")


# ─────────────────────────────────────────────────────────────────────
# DIAGNOSE
# ─────────────────────────────────────────────────────────────────────

@cli.command("diagnose")
@click.argument("data", type=click.Path(exists=True))
@click.option("--target", "-t", required=True, help="Target column name")
@click.option("--config", "-c", default=None, help="YAML config path")
@click.option("--output", "-o", default=None,
              help="Save JSON report to this path")
@click.option("--no-plots", is_flag=True, default=False,
              help="Disable diagnostic plot generation")
def cmd_diagnose(
    data: str,
    target: str,
    config: Optional[str],
    output: Optional[str],
    no_plots: bool,
) -> None:
    """
    Diagnose a dataset for quality issues.

    \b
    Examples:
      autoguard diagnose data.csv --target label
      autoguard diagnose data.csv --target label --output diag.json
    """
    import json
    from autoguard import AutoGuard, AutoGuardConfig

    df = _load_csv(data)

    if config:
        cfg = AutoGuardConfig.from_yaml(config)
        cfg.target = target
    else:
        cfg = AutoGuardConfig(target=target)

    if no_plots:
        cfg.data.generate_plots = False

    ag = AutoGuard(target=target, config=cfg)
    report = ag.diagnose(df)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        console.print(f"\n[bold green]✓ Diagnosis report saved:[/bold green] [cyan]{output}[/cyan]")


# ─────────────────────────────────────────────────────────────────────
# FIX
# ─────────────────────────────────────────────────────────────────────

@cli.command("fix")
@click.argument("data", type=click.Path(exists=True))
@click.option("--target", "-t", required=True, help="Target column name")
@click.option("--output", "-o", default="cleaned.csv", show_default=True,
              help="Output path for cleaned CSV")
def cmd_fix(data: str, target: str, output: str) -> None:
    """
    Auto-clean a dataset (fill missing, encode, normalize).

    \b
    Examples:
      autoguard fix data.csv --target label --output clean.csv
    """
    from autoguard import AutoGuard
    df = _load_csv(data)
    ag = AutoGuard(target=target)
    df_clean = ag.auto_fix(df)
    df_clean.to_csv(output, index=False)
    console.print(f"\n[bold green]✓ Clean dataset saved:[/bold green] [cyan]{output}[/cyan]  "
                  f"({df_clean.shape[0]:,} rows × {df_clean.shape[1]} cols)")


# ─────────────────────────────────────────────────────────────────────
# MONITOR
# ─────────────────────────────────────────────────────────────────────

@cli.command("monitor")
@click.argument("new_data", type=click.Path(exists=True))
@click.option("--model", "-m", "model_path", required=True,
              type=click.Path(exists=True), help="Path to saved .pkl model")
@click.option("--output", "-o", default=None,
              help="Save drift report JSON to this path")
def cmd_monitor(new_data: str, model_path: str, output: Optional[str]) -> None:
    """
    Detect drift in NEW_DATA vs. training distribution.

    \b
    Examples:
      autoguard monitor new.csv --model model.pkl
      autoguard monitor new.csv --model model.pkl --output drift.json
    """
    df = _load_csv(new_data)
    ag = _load_ag(model_path)

    report = ag.monitor(df, save_report=(output is None))

    if output:
        from autoguard.drift.detector import DriftDetector
        det = DriftDetector()
        det.save_report(report, path=output)
        console.print(f"\n[bold green]✓ Drift report saved:[/bold green] [cyan]{output}[/cyan]")


# ─────────────────────────────────────────────────────────────────────
# EXPLAIN
# ─────────────────────────────────────────────────────────────────────

@cli.command("explain")
@click.option("--model", "-m", "model_path", required=True,
              type=click.Path(exists=True), help="Path to saved .pkl model")
@click.option("--data", "-d", "data_path", required=True,
              type=click.Path(exists=True), help="Data CSV for SHAP computation")
@click.option("--sample-index", default=0, show_default=True,
              help="Row index for local explanation")
def cmd_explain(model_path: str, data_path: str, sample_index: int) -> None:
    """
    Generate SHAP explanations for a trained model.

    \b
    Examples:
      autoguard explain --model model.pkl --data data.csv
      autoguard explain --model model.pkl --data data.csv --sample-index 5
    """
    ag = _load_ag(model_path)
    df = _load_csv(data_path)

    target = ag._target
    X = df.drop(columns=[target]) if target in df.columns else df

    ag.explain(X=X, sample_index=sample_index)
    console.print(f"\n[bold green]✓ SHAP plots saved to:[/bold green] "
                  f"[cyan]{ag.config.explain.output_dir}[/cyan]")


# ─────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────

@cli.command("report")
@click.option("--model", "-m", "model_path", required=True,
              type=click.Path(exists=True), help="Path to saved .pkl model")
@click.option("--output", "-o", default="report.html", show_default=True,
              help="Output path for the HTML report")
@click.option("--format", "fmt", default="html",
              type=click.Choice(["html", "json"]), show_default=True,
              help="Report format")
def cmd_report(model_path: str, output: str, fmt: str) -> None:
    """
    Generate an HTML/JSON report for a saved model.

    \b
    Examples:
      autoguard report --model model.pkl
      autoguard report --model model.pkl --output my_report.html
    """
    ag = _load_ag(model_path)
    ag.report(output_path=output, format=fmt)
    console.print(f"\n[bold green]✓ Report saved:[/bold green] [cyan]{output}[/cyan]")


# ─────────────────────────────────────────────────────────────────────
# SERVE
# ─────────────────────────────────────────────────────────────────────

@cli.command("serve")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option("--reload", is_flag=True, default=False,
              help="Enable auto-reload (dev mode)")
def cmd_serve(model_path: str, host: str, port: int, reload: bool) -> None:
    """
    Deploy a trained model as a REST API.

    \b
    Examples:
      autoguard serve model.pkl
      autoguard serve model.pkl --host 127.0.0.1 --port 9000

    Endpoints:
      GET  /health        → health check
      GET  /model/info    → leaderboard + features
      POST /predict       → batch inference
      POST /predict/proba → probabilities
      POST /monitor       → drift detection
      GET  /docs          → Swagger UI
    """
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed.[/red] Run: pip install autoguard-ml[api]")
        sys.exit(1)

    ag = _load_ag(model_path)

    from autoguard.api.server import create_app
    app = create_app(ag)

    console.print(f"\n[bold green]AutoGuard API starting[/bold green]")
    console.print(f"  Host   : [cyan]http://{host}:{port}[/cyan]")
    console.print(f"  Model  : [cyan]{model_path}[/cyan]  ({ag.best_model_name})")
    console.print(f"  Docs   : [cyan]http://{host}:{port}/docs[/cyan]")
    console.print()

    uvicorn.run(app, host=host, port=port, reload=reload)


# ─────────────────────────────────────────────────────────────────────
# CONFIG INIT
# ─────────────────────────────────────────────────────────────────────

@cli.command("init")
@click.option("--output", "-o", default="autoguard_config.yaml", show_default=True,
              help="Path for the generated config file")
def cmd_init(output: str) -> None:
    """
    Generate a starter config.yaml with all defaults.

    \b
    Examples:
      autoguard init
      autoguard init --output my_project/config.yaml
    """
    from autoguard import AutoGuardConfig
    cfg = AutoGuardConfig()
    cfg.to_yaml(output)
    console.print(f"[bold green]✓ Config created:[/bold green] [cyan]{output}[/cyan]")
    console.print(f"  Edit it, then pass with:  autoguard train data.csv -t label -c {output}")


# ─────────────────────────────────────────────────────────────────────
# GETBEST
# ─────────────────────────────────────────────────────────────────────

@cli.command("getbest")
@click.argument("data", type=click.Path(exists=True))
@click.option("--target", "-t", required=True, help="Target column name")
@click.option(
    "--problem-type", default=None,
    type=click.Choice(["classification", "regression"]),
    help="Override auto-detected problem type",
)
@click.option(
    "--output", "-o", default=None,
    help="Save recommendations to this JSON path",
)
@click.option(
    "--sample-frac", default=0.15, show_default=True,
    type=click.FloatRange(0.01, 1.0),
    help="Fraction of data used for benchmarking (0.01-1.0)",
)
@click.option(
    "--n-trials", default=5, show_default=True,
    type=click.IntRange(1, 100),
    help="Optuna trials per model (higher = more accurate, slower)",
)
@click.option(
    "--models", default=None,
    help="Comma-separated list of models to evaluate (default: all registered)",
)
@click.option(
    "--top", default=12, show_default=True,
    type=click.IntRange(1, 50),
    help="Number of models to show in the output table",
)
def cmd_getbest(
    data: str,
    target: str,
    problem_type: Optional[str],
    output: Optional[str],
    sample_frac: float,
    n_trials: int,
    models: Optional[str],
    top: int,
) -> None:
    """
    Recommend the best model(s) for your dataset BEFORE training.

    Runs a fast benchmark on a sample of DATA using all registered models
    (with lightweight Optuna HPO), then combines benchmark scores with
    dataset heuristics to surface the top algorithm choices.

    No model is saved — this is purely advisory. Use `autoguard train`
    once you know which model(s) to use.

    \\b
    Examples:
      autoguard getbest data.csv --target label
      autoguard getbest data.csv --target price --problem-type regression
      autoguard getbest data.csv --target label --sample-frac 0.3 --n-trials 10
      autoguard getbest data.csv --target label --output recs.json
      autoguard getbest data.csv --target label --models random_forest,xgboost,lightgbm
    """
    import json
    from autoguard.automl.getbest import GetBestEngine
    from autoguard.core.config import AutoMLConfig

    df = _load_csv(data)

    console.print(
        f"\n[bold green]AutoGuard[/bold green] v0.2.2 [bold cyan]GetBest[/bold cyan] — "
        f"[bold]{data}[/bold]  "
        f"({df.shape[0]:,} rows × {df.shape[1]} cols)"
    )
    console.rule("[bold]Model Recommendation[/bold]")

    # Parse optional models list
    model_list: Optional[list[str]] = None
    if models:
        model_list = [m.strip() for m in models.split(",") if m.strip()]

    cfg = AutoMLConfig(
        getbest_sample_frac=sample_frac,
        getbest_n_trials=n_trials,
    )
    engine = GetBestEngine(config=cfg)

    try:
        results = engine.run(
            df=df,
            target=target,
            problem_type=problem_type,
            sample_frac=sample_frac,
            n_trials=n_trials,
            models=model_list,
        )
    except Exception as e:
        console.print(f"[bold red]GetBest failed:[/bold red] {e}")
        import sys
        sys.exit(1)

    try:
        engine.print_recommendations(results, top_n=top)
    except Exception as e:
        # Rendering glitch (e.g. terminal encoding) — fall back to plain text
        console.print(f"\n[bold]Top recommendations:[/bold]")
        for r in results[:top]:
            if not (r["combined_score"] != r["combined_score"]):  # not NaN
                tag = "[1]" if r["rank"] == 1 else "[*]" if r["recommended"] else "   "
                console.print(
                    f"  {r['rank']:>2}. {tag} {r['model']:<25} "
                    f"cv={r['cv_score']:.4f}  combined={r['combined_score']:.4f}"
                )

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Make results JSON-serializable
        clean = []
        for r in results:
            cr = dict(r)
            for k, v in cr.items():
                if isinstance(v, float) and (v != v):  # NaN check
                    cr[k] = None
            clean.append(cr)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "dataset": data,
                    "target": target,
                    "problem_type": results[0]["metric"].replace("f1_weighted", "classification")
                        if results else "unknown",
                    "sample_frac": sample_frac,
                    "n_trials": n_trials,
                    "recommendations": clean,
                },
                f, indent=2,
            )
        console.print(f"[bold green]✓ Recommendations saved:[/bold green] [cyan]{output}[/cyan]")


if __name__ == "__main__":
    cli()
