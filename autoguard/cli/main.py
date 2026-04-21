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
from typing import Optional

import click
from rich.console import Console

console = Console()

# ─────────────────────────────────────────────────────────────────────
# ROOT GROUP
# ─────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.1.0", prog_name="autoguard")
def cli() -> None:
    """
    \b
    ╔══════════════════════════════════════════╗
    ║  AutoGuard ML  v0.1.0                   ║
    ║  AutoML + Diagnosis + Drift Detection   ║
    ╚══════════════════════════════════════════╝

    Run `autoguard COMMAND --help` for details on any command.
    """


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> "pd.DataFrame":
    import pandas as pd
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        sys.exit(1)
    return pd.read_csv(p)


def _load_ag(model_path: str) -> "AutoGuard":
    from autoguard import AutoGuard
    return AutoGuard.load(model_path)


def _make_ag(target: str, config: Optional[str]) -> "AutoGuard":
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


if __name__ == "__main__":
    cli()
