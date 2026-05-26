# 🛡️ AutoGuard ML

> **AutoML + Dataset Diagnosis + Drift Detection — all in one package.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/autoguard-ml?style=flat-square&color=orange)](https://pypi.org/project/autoguard-ml/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](https://github.com/Nikhil-Parab/autoguard-ml/pulls)

---

Most ML tools split responsibilities across multiple libraries.
You use one tool for training, another for monitoring, another for explainability.

**AutoGuard ML puts everything in one clean pipeline:**

```python
from autoguard import AutoGuard

ag = AutoGuard(target="churn")
results = ag.get_best(df)          # NEW: find the best model BEFORE training
ag.diagnose(df)                    # catch data problems before they ruin your model
df_clean = ag.auto_fix(df)        # auto-clean the dataset
ag.fit(df_clean)                   # AutoML picks and tunes the best model
ag.explain()                       # SHAP feature importance
ag.report()                        # HTML report you can share
ag.monitor(new_df)                 # detect drift in production
```

That's the whole pipeline. One object. Eight methods.

---

## Table of Contents

- [What it does](#what-it-does)
- [Install](#install)
- [5-Minute Quickstart](#5-minute-quickstart)
- [Feature Guide](#feature-guide)
  - [0. GetBest — Pre-Training Recommender](#0-getbest--pre-training-model-recommender-new)
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
| 🔍 **GetBest** | **NEW** — Recommends the best algorithm for your dataset before training. Fast benchmark + heuristics. |
| 🩺 **Dataset Doctor** | Missing values, class imbalance, outliers, feature correlation, data leakage, skewed distributions |
| 🧹 **Auto-Fix** | Fills missing values, encodes categoricals, caps outliers, normalizes skewed columns |
| ⚙️ **AutoML Engine** | Tries **12 models** with Optuna HPO + k-fold CV, picks the winner |
| 🔍 **Explainability** | SHAP global importance and per-prediction local explanations |
| 📊 **Report Generator** | Self-contained dark-theme HTML report with leaderboard, risk score, and issue list |
| 📡 **Drift Monitor** | KS test + PSI for numeric, Chi-squared + PSI for categorical, severity scores, alert logging |
| 🌐 **REST API** | FastAPI server with /predict, /predict/proba, /monitor, /model/info |
| 🖥️ **CLI** | `autoguard getbest`, `train`, `diagnose`, `fix`, `monitor`, `explain`, `serve`, `report` |

---

## Install

**Basic install (training + monitoring):**

```bash
pip install autoguard-ml
```

**With CatBoost support:**

```bash
pip install autoguard-ml[extras]
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
git clone https://github.com/Nikhil-Parab/autoguard-ml
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

### Find the best model first (optional but recommended)

```bash
autoguard getbest data.csv --target churn
```

```
AutoGuard - Model Recommendation Report
 Rank  Model                   CV Score  Heuristic  Combined  Time
  [1]  lightgbm                 0.99257    +0.070   1.06257   163s
  [*]  random_forest            0.97053    +0.050   1.02053    44s
  [*]  extra_trees              0.96807    +0.040   1.00807     8s
       gradient_boosting        0.99037    +0.000   0.99037    81s
       ...

BEST: lightgbm — Fast gradient boosting, great for large datasets
Top recommendations: #1 lightgbm  #2 random_forest  #3 extra_trees
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
Dataset Doctor
  Rows: 5,000  Cols: 15  Target: churn
  Risk Score: 58.0 / 100  (HIGH)
  HIGH     - 35% missing values in column 'age'
  HIGH     - Severe class imbalance: 'churn' = 6.2%
  CRITICAL - Possible leakage in 'customer_id' (r=0.99)
  MEDIUM   - Highly correlated: income <-> salary (0.97)
```

```python
# Step 2: Auto-clean
df_clean = ag.auto_fix(df)

# Step 3: Train — AutoML picks the best model from 12 algorithms
ag.fit(df_clean)
```

```
  AutoML Leaderboard
  Rank  Model                CV Score
  1     lightgbm             0.99257
  2     random_forest        0.97053
  3     extra_trees          0.96807
  4     gradient_boosting    0.99037
  ...

  Best model: lightgbm
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

---

## Feature Guide

### 0. GetBest — Pre-Training Model Recommender *(NEW)*

Run **before** `train` to find which algorithm fits your dataset best — without committing to a full training run.

Works by benchmarking all 12 registered models on a **sample** of your data (default 15%) with lightweight Optuna HPO, then combining the CV score with dataset-aware **heuristics** (dataset size, feature count, class imbalance, categorical cardinality, sparsity).

**CLI:**

```bash
# Basic — auto-detects classification/regression
autoguard getbest data.csv --target label

# Regression
autoguard getbest data.csv --target price --problem-type regression

# Use more data and more trials for better accuracy
autoguard getbest data.csv --target label --sample-frac 0.3 --n-trials 10

# Only benchmark specific models
autoguard getbest data.csv --target label --models random_forest,xgboost,lightgbm

# Save recommendations as JSON
autoguard getbest data.csv --target label --output recs.json
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--target / -t` | required | Target column name |
| `--problem-type` | auto-detect | `classification` or `regression` |
| `--sample-frac` | 0.15 | Fraction of data used for benchmark (0.01–1.0) |
| `--n-trials` | 5 | Optuna trials per model |
| `--models` | all | Comma-separated model subset |
| `--top` | 12 | Rows to show in the output table |
| `--output / -o` | none | Save ranked recommendations to JSON |

**Python API:**

```python
from autoguard import AutoGuard

ag = AutoGuard(target="label")
results = ag.get_best(df)

# results is a ranked list of dicts
print(results[0]["model"])          # "lightgbm"
print(results[0]["combined_score"]) # 1.063

# Save the report
results = ag.get_best(df, save_report="recs.json")

# Or use the engine directly
from autoguard.automl.getbest import GetBestEngine
engine = GetBestEngine()
results = engine.run(df, target="label", sample_frac=0.2, n_trials=10)
engine.print_recommendations(results)
```

**Dataset heuristics applied:**

| Dataset characteristic | Models boosted |
|---|---|
| Large (>50k rows) | LightGBM, XGBoost, RandomForest, ExtraTrees |
| Small (<1k rows) | SVM, KNN, LogisticRegression |
| High cardinality categoricals | CatBoost, LightGBM |
| Many features (>100) | RandomForest, XGBoost, ExtraTrees |
| Sparse data (many zeros) | Lasso |
| Class imbalance | XGBoost, RandomForest |

---

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

---

### 3. AutoML Engine

Tries **12 models**, tunes hyperparameters with Optuna, cross-validates, picks the winner.

```python
ag.fit(df_clean)

# Force problem type if needed
ag.fit(df_clean, problem_type="classification")
ag.fit(df_clean, problem_type="regression")

# After fitting
ag.best_model_name   # "lightgbm"
ag.best_model        # the fitted sklearn-compatible estimator
ag.leaderboard       # pd.DataFrame with all model scores
ag.problem_type      # "classification" or "regression"
ag.feature_cols      # list of feature column names used
```

**All 12 models:**

| Model | Type | Best for |
|---|---|---|
| `random_forest` | Ensemble | Robust default, handles noise |
| `xgboost` | Gradient Boosting | Tabular champion, large datasets |
| `lightgbm` | Gradient Boosting | Fast, great for large data |
| `gradient_boosting` | Gradient Boosting | High accuracy, sklearn GBM |
| `extra_trees` | Ensemble | Fast, robust to noise |
| `catboost` | Gradient Boosting | Native categorical support (optional: `pip install catboost`) |
| `logistic_regression` | Linear | Fast linear baseline |
| `ridge` | Linear | Regularized linear |
| `lasso` | Linear | L1 sparsity, feature selection |
| `svm` | Kernel | Small/medium datasets |
| `knn` | Instance-based | Small datasets |
| `decision_tree` | Tree | Interpretable single-tree baseline |

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
  models:
    - xgboost
    - lightgbm
    - random_forest
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
| RandomForest, XGBoost, LightGBM, ExtraTrees, GradientBoosting | TreeExplainer (fast) |
| LogisticRegression, Ridge, Lasso | LinearExplainer |
| SVM, KNN, DecisionTree, CatBoost | KernelExplainer (slower, model-agnostic) |

**Output files** saved to `autoguard_output/explain/`:

| File | What it shows |
|---|---|
| `shap_global_importance.png` | Bar chart: mean absolute SHAP per feature |
| `shap_summary_plot.png` | Beeswarm: feature value vs impact |
| `shap_local_0.png` | Waterfall for one specific prediction |

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
data["best_model"]    # "lightgbm"
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
| < 0.10 | No significant drift | All good |
| 0.10 – 0.20 | Moderate drift | Investigate |
| > 0.20 | Severe drift | Consider retraining |

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

### getbest *(NEW)*

Find the best model for your dataset before training:

```bash
autoguard getbest data.csv --target churn
autoguard getbest data.csv --target price --problem-type regression
autoguard getbest data.csv --target churn --sample-frac 0.3 --n-trials 10
autoguard getbest data.csv --target churn --output recs.json
autoguard getbest data.csv --target churn --models random_forest,xgboost,lightgbm
```

| Flag | Default | Description |
|---|---|---|
| --target / -t | required | Target column name |
| --problem-type | auto | Force classification or regression |
| --sample-frac | 0.15 | Fraction of data for benchmark (0.01–1.0) |
| --n-trials | 5 | Optuna trials per model |
| --models | all | Comma-separated model subset |
| --top | 12 | Number of rows in output table |
| --output / -o | none | Save ranked results to JSON |

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
    - gradient_boosting
    - extra_trees
    - logistic_regression
  n_trials: 50            # Optuna trials per model — more = better, slower
  cv_folds: 5
  scoring_classification: f1_weighted
  scoring_regression: neg_root_mean_squared_error
  timeout_per_model: 120  # seconds per model
  handle_imbalance: true
  getbest_sample_frac: 0.15   # fraction used by getbest
  getbest_n_trials: 5         # trials per model in getbest

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
        models=["xgboost", "lightgbm", "random_forest"],
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

@ModelRegistry.register("my_model")
def build_my_model(trial, problem_type):
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    params = {
        "hidden_layer_sizes": trial.suggest_categorical("mlp_layers", [(64,), (128,), (64, 32)]),
        "alpha":              trial.suggest_float("mlp_alpha", 1e-5, 1e-1, log=True),
        "max_iter":           500,
        "random_state":       42,
    }
    if problem_type == "classification":
        return MLPClassifier(**params)
    return MLPRegressor(**params)

# Now include it in the model search
from autoguard import AutoGuard, AutoGuardConfig
from autoguard.core.config import AutoMLConfig

cfg = AutoGuardConfig(
    automl=AutoMLConfig(models=["xgboost", "lightgbm", "my_model"])
)
ag = AutoGuard(target="label", config=cfg)
ag.fit(df)
```

**See all registered models:**

```python
from autoguard.automl.registry import ModelRegistry
print(ModelRegistry.available())
# ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression', 'ridge',
#  'gradient_boosting', 'extra_trees', 'svm', 'knn', 'decision_tree', 'lasso', 'catboost']
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
  "best_model": "lightgbm",
  "problem_type": "classification",
  "version": "0.2.0"
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
  "model_name": "lightgbm",
  "problem_type": "classification",
  "n_samples": 2
}
```

---

## Project Structure

```
autoguard-ml/
|
+-- autoguard/                     main package
|   +-- __init__.py                public API: AutoGuard, AutoGuardConfig
|   |
|   +-- core/
|   |   +-- guard.py               AutoGuard class — the main entry point
|   |   +-- config.py              YAML-driven configuration dataclasses
|   |   +-- exceptions.py          custom exception hierarchy
|   |   +-- logging.py             Rich console + JSON file logging
|   |
|   +-- data/
|   |   +-- doctor.py              DatasetDoctor: 7 quality checks + plots
|   |   +-- cleaner.py             AutoCleaner: fill, encode, normalise
|   |
|   +-- automl/
|   |   +-- engine.py              AutoMLEngine: HPO + CV + leaderboard
|   |   +-- registry.py            ModelRegistry: 12 built-in models + plugin system
|   |   +-- getbest.py             GetBestEngine: fast pre-training recommender (NEW)
|   |
|   +-- explain/
|   |   +-- shap_explainer.py      ShapExplainer: Tree/Linear/Kernel + plots
|   |
|   +-- drift/
|   |   +-- detector.py            DriftDetector: KS + PSI + Chi2 + alerts
|   |
|   +-- api/
|   |   +-- server.py              FastAPI REST server
|   |
|   +-- cli/
|   |   +-- main.py                Click CLI: 8 commands
|   |
|   +-- utils/
|       +-- report.py              HTMLReportGenerator

+-- tests/
+-- examples/
+-- pyproject.toml
+-- README.md
+-- LICENSE
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
```

---

## FAQ

**Should I run `getbest` before `train`?**

It's optional but recommended for new datasets. `getbest` takes a few minutes and tells you which model family will likely work best, so you can narrow the `train` search with `--config` for faster full training.

**Do I have to call `auto_fix` before `fit`?**

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

Yes. Problem type is auto-detected from the target column. Force it explicitly with `ag.fit(df, problem_type="regression")` or `autoguard getbest data.csv --target price --problem-type regression`.

**How do I use CatBoost?**

```bash
pip install autoguard-ml[extras]   # installs catboost
```

Then add it to your model list in config:

```yaml
automl:
  models: [xgboost, lightgbm, catboost]
```

**How do I silence the console output?**

```python
cfg = AutoGuardConfig(verbose=False)
ag = AutoGuard(target="label", config=cfg)
```

**How do I trigger alerts when drift is detected?**

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
git clone https://github.com/Nikhil-Parab/autoguard-ml
cd autoguard-ml
pip install -e ".[dev]"
pytest
```

---

## Changelog

### v0.2.0
- **NEW**: `autoguard getbest` CLI command — pre-training model recommender
- **NEW**: `ag.get_best(df)` Python API method
- **NEW**: `GetBestEngine` with dataset heuristics + fast benchmark
- **EXPANDED**: Model registry 5 → 12 models (added `gradient_boosting`, `extra_trees`, `svm`, `knn`, `decision_tree`, `lasso`, `catboost`)
- **FIX**: XGBoost/LightGBM GPU auto-detection with graceful CPU fallback
- **FIX**: Windows terminal Unicode compatibility

### v0.1.2
- Fix author email, minor metadata updates

### v0.1.0
- Initial release: AutoML, Dataset Doctor, Drift Detection, SHAP explainability, REST API

---

## License

MIT © Nikhil Parab. See [LICENSE](LICENSE).

---

## Acknowledgements

Built on top of: scikit-learn · XGBoost · LightGBM · CatBoost · Optuna · SHAP · FastAPI · Rich · Click · scipy · matplotlib · seaborn
