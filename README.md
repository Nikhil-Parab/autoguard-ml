# 🛡️ AutoGuard ML

> **AutoML + Dataset Diagnosis + Drift Detection — all in one package.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![pip install](https://img.shields.io/badge/pip%20install-autoguard--ml-orange?style=flat-square)](https://pypi.org/project/autoguard-ml/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](https://github.com/autoguard/autoguard-ml/pulls)

---

Most ML tools split responsibilities across multiple libraries.
You use one tool for training, another for monitoring, another for explainability.

**AutoGuard ML puts everything in one clean pipeline:**

```python
from autoguard import AutoGuard

ag = AutoGuard(target="churn")
ag.diagnose(df)            # catch data problems before they ruin your model
df_clean = ag.auto_fix(df) # auto-clean the dataset
ag.fit(df_clean)           # AutoML picks and tunes the best model
ag.explain()               # SHAP feature importance
ag.report()                # HTML report you can share
ag.monitor(new_df)         # detect drift in production
```

That's the whole pipeline. One object. Seven methods.

---

## Table of Contents

- [What it does](#what-it-does)
- [Install](#install)
- [5-Minute Quickstart](#5-minute-quickstart)
- [Feature Guide](#feature-guide)
  - [1. Dataset Doctor](#1-dataset-doctor)
  - [2. Auto-Fix](#2-auto-fix-data-cleaning)
  - [3. AutoML Engine](#3-automl-engine)
  - [4. Explainability](#4-explainability-shap)
  - [5. Report Generator](#5-report-generator)
  - [6. Drift Monitor](#6-drift-monitor)
  - [7. Save and Load](#7-save--load)
  - [8. Predict](#8-predict)
- [CLI Reference](#cli-reference)
- [Config System](#config-system-yaml)
- [Plugin System](#plugin-system-custom-models)
- [REST API](#rest-api)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [FAQ](#faq)
- [Contributing](#contributing)

---

## What it does

| Module | What it catches or solves |
|---|---|
| 🩺 **Dataset Doctor** | Missing values, class imbalance, outliers, feature correlation, data leakage, skewed distributions |
| 🧹 **Auto-Fix** | Fills missing values, encodes categoricals, caps outliers, normalizes skewed columns |
| ⚙️ **AutoML Engine** | Tries RandomForest, XGBoost, LightGBM, LogisticRegression, Ridge — tunes HPO with Optuna, picks winner via CV |
| 🔍 **Explainability** | SHAP global importance and per-prediction local explanations |
| 📊 **Report Generator** | Self-contained dark-theme HTML report with leaderboard, risk score, and issue list |
| 📡 **Drift Monitor** | KS test + PSI for numeric, Chi-squared + PSI for categorical, severity scores, alert logging |
| 🌐 **REST API** | FastAPI server with /predict, /predict/proba, /monitor, /model/info |
| 🖥️ **CLI** | autoguard train, diagnose, fix, monitor, explain, serve, report |

---

## Install

**Basic install (training + monitoring):**

```bash
pip install autoguard-ml
```

**With REST API support:**

```bash
pip install autoguard-ml[api]
```

**Everything:**

```bash
pip install autoguard-ml[all]
```

**From source:**

```bash
git clone https://github.com/autoguard/autoguard-ml
cd autoguard-ml
pip install -e ".[dev]"
```

> **Requirements:** Python 3.10+, numpy, pandas, scikit-learn, xgboost, lightgbm, optuna, shap, scipy, matplotlib, seaborn, rich, click

---

## 5-Minute Quickstart

### Get some data

```python
import pandas as pd
df = pd.read_csv("your_data.csv")
```

No data yet? Generate sample data:

```bash
python examples/generate_sample_data.py
# creates: examples/data/train.csv and new_data.csv
```

### Run the full pipeline

```python
from autoguard import AutoGuard

# Initialize with your target column
ag = AutoGuard(target="churn")

# Step 1: Diagnose data quality
ag.diagnose(df)
```

```
╭─────────────────── Dataset Doctor ──────────────────────╮
│ Rows: 5,000  Cols: 15  Target: churn                    │
│ Risk Score: 58.0 / 100  (HIGH)                          │
╰─────────────────────────────────────────────────────────╯
  HIGH     ⚠ 35% missing values in column 'age'
  HIGH     ⚠ Severe class imbalance: 'churn' = 6.2%
  CRITICAL ⚠ Possible leakage in 'customer_id' (r=0.99)
  MEDIUM   ⚠ Highly correlated: income ↔ salary (0.97)
```

```python
# Step 2: Auto-clean
df_clean = ag.auto_fix(df)

# Step 3: Train — AutoML picks the best model
ag.fit(df_clean)
```

```
  🏆 AutoML Leaderboard
  Rank  Model                CV Score
  ⭐ 1  XGBoost              0.92341
     2  RandomForest         0.90187
     3  LightGBM             0.89923
     4  LogisticRegression   0.83410

  ✓ Best model: XGBoost
```

```python
# Step 4: Explain with SHAP
ag.explain()

# Step 5: Generate HTML report
ag.report()

# Step 6: Save
ag.save("model.pkl")

# Step 7: Monitor production data for drift
ag.monitor(new_df)
```

```
╭──────────────── Drift Monitor ────────────────╮
│ Severity: 72.3/100  (HIGH)  |  3 features     │
╰────────────────────────────────────────────────╯
  monthly_fee  PSI=0.31  HIGH
  city         PSI=0.29  MODERATE
  usage_score  PSI=0.14  MODERATE
```

---

## Feature Guide

### 1. Dataset Doctor

Runs before you train. Catches problems that would silently destroy model accuracy.

```python
report = ag.diagnose(df)

# Summary
report["risk_score"]     # 58.0  (0 to 100)
report["risk_level"]     # "high"  (low / medium / high / critical)
report["issues"]         # list of all warnings with severity

# Individual check results
report["missing_values"]["columns_with_missing"]   # {col: missing_ratio}
report["missing_values"]["high_missing_columns"]   # columns above threshold

report["class_imbalance"]["is_imbalanced"]         # True or False
report["class_imbalance"]["class_distribution"]    # {class: ratio}

report["correlations"]["high_correlation_pairs"]   # [{a, b, correlation}]

report["leakage_risk"]["suspicious_columns"]       # [{column, corr_with_target}]

report["outliers"]["flagged_columns"]              # {col: {n_outliers, pct}}

report["skewness"]["skewed_columns"]               # {col: skew_value}

report["constant_columns"]["constant_columns"]     # [col, ...]
```

**Generated plots** saved to `autoguard_output/plots/`:

| File | What it shows |
|---|---|
| `missing_values.png` | Bar chart of missing ratios per column |
| `correlation_matrix.png` | Heatmap of feature correlations |
| `target_distribution.png` | Class balance or value histogram |
| `skewness.png` | Absolute skewness by feature |

**Disable plots:**

```python
from autoguard import AutoGuard, AutoGuardConfig

cfg = AutoGuardConfig()
cfg.data.generate_plots = False
ag = AutoGuard(target="label", config=cfg)
```

---

### 2. Auto-Fix (Data Cleaning)

Automatically preprocesses raw data ready for training.

```python
df_clean = ag.auto_fix(df)
```

**What it does, in order:**

| Step | What happens |
|---|---|
| Drop constants | Removes columns with only one unique value |
| Fill missing numeric | Fills with column median |
| Fill missing categorical | Fills with column mode |
| Cap outliers | Clips values beyond IQR x 3 |
| Log-transform skewed | Applies log1p to highly skewed numeric columns |
| Encode categoricals | Label encode (<=15 unique), one-hot (<=50 unique), or frequency encode |

The target column is **never modified**.

```python
# Apply the same fitted cleaner to new production data
df_new_clean = ag.auto_fix(new_df)
```

---

### 3. AutoML Engine

Tries multiple models, tunes hyperparameters with Optuna, cross-validates, picks the winner.

```python
ag.fit(df_clean)

# Force problem type if needed
ag.fit(df_clean, problem_type="classification")
ag.fit(df_clean, problem_type="regression")

# After fitting
ag.best_model_name   # "xgboost"
ag.best_model        # the fitted sklearn-compatible estimator
ag.leaderboard       # pd.DataFrame with all model scores
ag.problem_type      # "classification" or "regression"
ag.feature_cols      # list of feature column names used
```

**Default models tried:**

| Model | Classification | Regression |
|---|---|---|
| RandomForest | Yes | Yes |
| XGBoost | Yes | Yes |
| LightGBM | Yes | Yes |
| LogisticRegression | Yes | No |
| Ridge | Yes (RidgeClassifier) | Yes |

**Default scoring metrics:**

| Problem | Metric |
|---|---|
| Classification | f1_weighted |
| Regression | neg_root_mean_squared_error |

Change via config:

```yaml
automl:
  scoring_classification: roc_auc
  scoring_regression: neg_mean_absolute_error
```

**Class imbalance** — SMOTE is applied automatically. Disable: `cfg.automl.handle_imbalance = False`

---

### 4. Explainability (SHAP)

```python
# Uses validation slice from training automatically
ag.explain()

# Custom data and specific sample
ag.explain(X=df.drop(columns=["label"]).head(300), sample_index=5)
```

**Explainer auto-selection:**

| Model type | SHAP explainer |
|---|---|
| RandomForest, XGBoost, LightGBM | TreeExplainer (fast) |
| LogisticRegression, Ridge, Lasso | LinearExplainer |
| Everything else | KernelExplainer (slower, model-agnostic) |

**Output files** saved to `autoguard_output/explain/`:

| File | What it shows |
|---|---|
| `shap_global_importance.png` | Bar chart: mean absolute SHAP per feature |
| `shap_summary_plot.png` | Beeswarm: feature value vs impact |
| `shap_local_0.png` | Waterfall for one specific prediction |

**Get importance as a Series:**

```python
importance = ag._explainer.get_feature_importance(X)
print(importance.head(10))
# monthly_fee    0.1823
# tenure_months  0.1541
# usage_score    0.1203
```

---

### 5. Report Generator

```python
# HTML report (default, opens in browser)
ag.report()
ag.report(output_path="results/my_report.html")

# JSON format
ag.report(format="json", output_path="report.json")

# The method also returns the report dict
data = ag.report()
data["best_model"]    # "xgboost"
data["leaderboard"]   # list of dicts
data["diagnosis"]     # full diagnosis report
```

The HTML report is **self-contained** — no internet required, no external CSS. Open it in any browser and share with teammates or stakeholders.

---

### 6. Drift Monitor

Detects when production data drifts away from the training distribution.

```python
report = ag.monitor(new_df)

report["overall_drift_severity"]  # 72.3 (0 to 100)
report["drift_level"]             # "high" (none / low / moderate / high / critical)
report["n_features_drifted"]      # 3
report["drifted_features"]        # ["monthly_fee", "city", "usage_score"]

# Per-feature detail
f = report["features"]["monthly_fee"]
f["psi"]         # 0.31  (Population Stability Index)
f["ks_pvalue"]   # 0.0003
f["drifted"]     # True
f["severity"]    # "high"
f["ref_mean"]    # 65.2  (training mean)
f["cur_mean"]    # 95.8  (current mean)
```

**PSI severity thresholds:**

| PSI Value | Meaning | Recommended action |
|---|---|---|
| Less than 0.10 | No significant drift | All good |
| 0.10 to 0.20 | Moderate drift | Investigate |
| Greater than 0.20 | Severe drift | Consider retraining |

**Continuous / streaming monitoring:**

```python
for batch in incoming_data_stream:
    result = ag.monitor(batch, save_report=True)

    if result["overall_drift_severity"] > 50:
        print("High drift — retraining needed")
        trigger_retraining_pipeline()
```

---

### 7. Save and Load

```python
# Save the entire fitted AutoGuard instance
ag.save("model.pkl")
ag.save("models/production_v2.pkl")

# Load it back anywhere — comes back fully fitted
ag2 = AutoGuard.load("model.pkl")
preds  = ag2.predict(new_df)
ag2.monitor(stream_df)
ag2.explain()
```

---

### 8. Predict

```python
# Classification
preds = ag.predict(X)          # array of class labels
proba = ag.predict_proba(X)    # array of shape (n_samples, n_classes)

# Regression
preds = ag.predict(X)          # array of continuous values
```

---

## CLI Reference

### train

```bash
autoguard train data.csv --target churn
autoguard train data.csv --target churn --output my_model.pkl
autoguard train data.csv --target churn --config config.yaml
autoguard train data.csv --target price --problem-type regression
autoguard train data.csv --target churn --no-diagnose --no-fix
autoguard train data.csv --target churn --report
```

| Flag | Default | Description |
|---|---|---|
| --target / -t | required | Target column name |
| --output / -o | model.pkl | Where to save the model |
| --config / -c | none | YAML config file path |
| --problem-type | auto | Force classification or regression |
| --no-diagnose | off | Skip the diagnosis step |
| --no-fix | off | Skip auto-cleaning |
| --report | off | Generate HTML report after training |

### diagnose

```bash
autoguard diagnose data.csv --target churn
autoguard diagnose data.csv --target churn --output diag.json
autoguard diagnose data.csv --target churn --no-plots
```

### fix

```bash
autoguard fix data.csv --target churn
autoguard fix data.csv --target churn --output clean.csv
```

### monitor

```bash
autoguard monitor new_data.csv --model model.pkl
autoguard monitor new_data.csv --model model.pkl --output drift_report.json
```

### explain

```bash
autoguard explain --model model.pkl --data data.csv
autoguard explain --model model.pkl --data data.csv --sample-index 5
```

### report

```bash
autoguard report --model model.pkl
autoguard report --model model.pkl --output results/report.html
autoguard report --model model.pkl --format json
```

### serve

```bash
autoguard serve model.pkl
autoguard serve model.pkl --host 127.0.0.1 --port 9000
```

### init

```bash
autoguard init                         # creates autoguard_config.yaml
autoguard init --output project/config.yaml
```

---

## Config System (YAML)

Drive everything from a YAML file instead of passing arguments in code.

```yaml
# autoguard_config.yaml

target: churn
verbose: true
output_dir: autoguard_output

data:
  missing_threshold: 0.30          # flag columns with >30% missing
  correlation_threshold: 0.95      # flag feature pairs above this
  outlier_method: iqr              # iqr or zscore
  outlier_threshold: 3.0
  skewness_threshold: 1.0
  imbalance_ratio_threshold: 0.10
  leakage_correlation_threshold: 0.98
  generate_plots: true

automl:
  models:
    - random_forest
    - xgboost
    - lightgbm
    - logistic_regression
  n_trials: 50            # Optuna trials per model — more = better, slower
  cv_folds: 5
  scoring_classification: f1_weighted
  scoring_regression: neg_root_mean_squared_error
  timeout_per_model: 120  # seconds per model
  handle_imbalance: true

explain:
  max_samples: 300
  plot_top_n_features: 15

drift:
  ks_pvalue_threshold: 0.05
  psi_threshold_warning: 0.10
  psi_threshold_alert: 0.20
  alert_email: null
```

**Use in Python:**

```python
ag = AutoGuard(config_path="autoguard_config.yaml")
ag.fit(df)
```

**Use in CLI:**

```bash
autoguard train data.csv --config autoguard_config.yaml
```

**Programmatic config:**

```python
from autoguard import AutoGuard, AutoGuardConfig
from autoguard.core.config import AutoMLConfig, DriftConfig

cfg = AutoGuardConfig(
    automl=AutoMLConfig(
        n_trials=100,
        models=["xgboost", "lightgbm"],
        scoring_classification="roc_auc",
    ),
    drift=DriftConfig(
        psi_threshold_alert=0.15,
    ),
)

ag = AutoGuard(target="label", config=cfg)
```

---

## Plugin System (Custom Models)

Add any sklearn-compatible model to the AutoML search:

```python
from autoguard.automl.registry import ModelRegistry

@ModelRegistry.register("extra_trees")
def build_extra_trees(trial, problem_type):
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    params = {
        "n_estimators": trial.suggest_int("et_n", 50, 300),
        "max_depth":    trial.suggest_int("et_depth", 3, 15),
        "random_state": 42,
        "n_jobs": -1,
    }
    if problem_type == "classification":
        return ExtraTreesClassifier(**params)
    return ExtraTreesRegressor(**params)


# Now include it in the model search
from autoguard import AutoGuard, AutoGuardConfig
from autoguard.core.config import AutoMLConfig

cfg = AutoGuardConfig(
    automl=AutoMLConfig(models=["xgboost", "random_forest", "extra_trees"])
)
ag = AutoGuard(target="label", config=cfg)
ag.fit(df)
```

**See all registered models:**

```python
from autoguard.automl.registry import ModelRegistry
print(ModelRegistry.available())
# ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression', 'ridge', 'extra_trees']
```

---

## REST API

### Start the server

```bash
pip install autoguard-ml[api]

# Train and save a model first
autoguard train data.csv --target churn --output model.pkl

# Start the API
autoguard serve model.pkl
# Swagger UI: http://localhost:8000/docs
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | /health | Server and model status |
| GET | /model/info | Model name, leaderboard, feature list |
| GET | /diagnosis | Last dataset diagnosis report |
| POST | /predict | Batch inference — returns class labels |
| POST | /predict/proba | Probabilistic inference — returns probabilities |
| POST | /monitor | Drift detection on incoming data batch |

### Example requests

**Health check:**

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "best_model": "xgboost",
  "problem_type": "classification",
  "version": "0.1.0"
}
```

**Predict:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"age": 35, "tenure_months": 12, "monthly_fee": 75, "city": "NYC"},
      {"age": 60, "tenure_months": 2,  "monthly_fee": 120, "city": "LA"}
    ]
  }'
```

```json
{
  "predictions": [0, 1],
  "model_name": "xgboost",
  "problem_type": "classification",
  "n_samples": 2
}
```

**Predict with probabilities:**

```bash
curl -X POST http://localhost:8000/predict/proba \
  -H "Content-Type: application/json" \
  -d '{"data": [{"age": 35, "tenure_months": 12, "monthly_fee": 75}]}'
```

```json
{
  "probabilities": [[0.82, 0.18]],
  "classes": ["0", "1"],
  "model_name": "xgboost",
  "n_samples": 1
}
```

**Drift detection:**

```bash
curl -X POST http://localhost:8000/monitor \
  -H "Content-Type: application/json" \
  -d '{"data": [{"age": 65, "tenure_months": 3, "monthly_fee": 110}]}'
```

```json
{
  "overall_drift_severity": 72.3,
  "drift_level": "high",
  "n_features_drifted": 3,
  "drifted_features": ["age", "monthly_fee", "tenure_months"],
  "timestamp": "2024-11-15T10:23:41Z"
}
```

**Python client:**

```python
import requests

BASE = "http://localhost:8000"

# Predict
r = requests.post(f"{BASE}/predict", json={"data": X_new.to_dict(orient="records")})
predictions = r.json()["predictions"]

# Monitor for drift
r = requests.post(f"{BASE}/monitor", json={"data": stream_batch.to_dict(orient="records")})
print(r.json()["drift_level"])
```

---

## Project Structure

```
autoguard-ml/
│
├── autoguard/                     main package
│   ├── __init__.py                public API: AutoGuard, AutoGuardConfig
│   │
│   ├── core/
│   │   ├── guard.py               AutoGuard class — the main entry point
│   │   ├── config.py              YAML-driven configuration dataclasses
│   │   ├── exceptions.py          custom exception hierarchy
│   │   └── logging.py             Rich console + JSON file logging
│   │
│   ├── data/
│   │   ├── doctor.py              DatasetDoctor: 7 quality checks + plots
│   │   └── cleaner.py             AutoCleaner: fill, encode, normalise
│   │
│   ├── automl/
│   │   ├── engine.py              AutoMLEngine: HPO + CV + leaderboard
│   │   └── registry.py            ModelRegistry: plugin system + built-ins
│   │
│   ├── explain/
│   │   └── shap_explainer.py      ShapExplainer: Tree/Linear/Kernel + plots
│   │
│   ├── drift/
│   │   └── detector.py            DriftDetector: KS + PSI + Chi2 + alerts
│   │
│   ├── api/
│   │   └── server.py              FastAPI REST server
│   │
│   ├── cli/
│   │   └── main.py                Click CLI: 7 commands
│   │
│   └── utils/
│       └── report.py              HTMLReportGenerator
│
├── tests/
│   ├── unit/
│   │   ├── test_doctor.py         DatasetDoctor tests
│   │   ├── test_cleaner.py        AutoCleaner tests
│   │   ├── test_drift.py          DriftDetector tests
│   │   └── test_config.py         config tests
│   └── integration/
│       └── test_pipeline.py       full end-to-end pipeline tests
│
├── examples/
│   ├── quickstart.py              complete working demo script
│   ├── generate_sample_data.py    creates train.csv and new_data.csv
│   └── config_example.yaml        annotated config file
│
├── pyproject.toml                 pip packaging config
├── README.md                      this file
└── LICENSE                        MIT
```

---

## Running Tests

```bash
pip install autoguard-ml[dev]

# All tests
pytest

# Unit tests only — fast, no ML training
pytest tests/unit/

# Integration tests — trains real models, takes 30-60 seconds
pytest tests/integration/

# Verbose with coverage report
pytest -v --cov=autoguard --cov-report=term-missing

# One specific file
pytest tests/unit/test_drift.py -v
```

---

## FAQ

**Do I have to call auto_fix before fit?**

No. `fit` runs its own internal preprocessing. `auto_fix` is optional — use it if you want to inspect or save the cleaned data before training.

**Can I use my own pre-trained model?**

Yes. Assign it directly:

```python
ag = AutoGuard(target="label")
ag.best_model = my_sklearn_model
ag.best_model_name = "my_model"
ag._problem_type = "classification"
ag._feature_cols = X.columns.tolist()
ag._train_df_raw = train_df
ag._is_fitted = True
```

**How do I get better model accuracy?**

Increase `n_trials` in config (default is 30):

```python
cfg.automl.n_trials = 100   # more trials = better model, slower
```

**What if all models fail during AutoML?**

An `AutoMLError` is raised with logs showing what went wrong. Common causes: too few samples for cross-validation, all-null columns, or mismatched data types. Run `diagnose` first to catch these.

**Does it work for regression?**

Yes. Problem type is auto-detected from the target column. Force it explicitly with `ag.fit(df, problem_type="regression")`.

**How do I silence the console output?**

```python
cfg = AutoGuardConfig(verbose=False)
ag = AutoGuard(target="label", config=cfg)
```

**Where do all the output files go?**

Everything goes to `autoguard_output/` by default. Change it:

```python
cfg = AutoGuardConfig(output_dir="my_project/outputs")
```

**How do I trigger alerts when drift is detected?**

Check the returned dict and add your own logic:

```python
result = ag.monitor(batch)
if result["overall_drift_severity"] > 50:
    send_slack_alert(f"Drift severity: {result['overall_drift_severity']:.1f}")
    trigger_retraining_job()
```

---

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Add tests for new functionality
4. Run `pytest` and ensure everything passes
5. Submit a pull request

**Development setup:**

```bash
git clone https://github.com/autoguard/autoguard-ml
cd autoguard-ml
pip install -e ".[dev]"
pytest
```

---

## License

MIT © AutoGuard Contributors. See [LICENSE](LICENSE).

---

## Acknowledgements

Built on top of: scikit-learn · XGBoost · LightGBM · Optuna · SHAP · FastAPI · Rich · Click · scipy · matplotlib · seaborn
