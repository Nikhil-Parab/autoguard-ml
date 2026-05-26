# AutoGuard ML — Command Reference

Click any command to copy it. Replace placeholders in `CAPS` with your values.

---

## 1 — Setup

**Create virtual environment**
```powershell
python -m venv venv
```

**Activate virtual environment (run every new terminal)**
```powershell
venv\Scripts\activate
```

**Install AutoGuard ML**
```powershell
pip install autoguard-ml
```

**Install with CatBoost support (optional)**
```powershell
pip install autoguard-ml[extras]
```

**Install API server (optional)**
```powershell
pip install autoguard-ml[api]
```

**Verify installation works**
```powershell
python -c "from autoguard import AutoGuard; print('AutoGuard OK')"
```

**Check version**
```powershell
autoguard --version
```

**Generate sample datasets (creates examples\data\)**
```powershell
python examples\generate_sample_data.py
```

**Run the full demo end-to-end**
```powershell
python examples\quickstart.py
```

---

## 2 — Using Your Own Dataset

**Step 0 — Find the best model BEFORE training (new in v0.2.0)**
```powershell
autoguard getbest YOUR_FILE.csv --target YOUR_COLUMN
```
Benchmarks all 12 models on a sample of your data and recommends the best one.
No model is saved — this is purely advisory. Takes a few minutes.

**Step 1 — Diagnose data quality issues**
```powershell
autoguard diagnose YOUR_FILE.csv --target YOUR_COLUMN
```
Shows risk score 0-100, detects missing values, imbalance, leakage, outliers

**Step 2 — Auto-clean the dataset**
```powershell
autoguard fix YOUR_FILE.csv --target YOUR_COLUMN --output clean.csv
```
Fills missing values, encodes categories, removes constants, normalises skewed columns

**Step 3 — Train a model (AutoML picks the best one)**
```powershell
autoguard train clean.csv --target YOUR_COLUMN --output model.pkl --report
```
Tries all 12 models (XGBoost, LightGBM, RandomForest, ExtraTrees, SVM, KNN, etc.) — saves best model + HTML report

> Warning: **Large datasets or imbalanced classes?** SMOTE can expand your data 5-10x before training,
> making Random Forest very slow or causing out-of-memory errors on Windows. Use a config file
> (see Section 5) to limit models and set `n_jobs: 1` to avoid subprocess crashes.

**Step 4 — Check drift when you get new data**
```powershell
autoguard monitor NEW_DATA.csv --model model.pkl
```
Compares new data vs training data, alerts if distribution has shifted

**For regression problems (predicting a number, not a category)**
```powershell
autoguard train clean.csv --target PRICE_COLUMN --problem-type regression
```

---

## 3 — All CLI Commands

### getbest — Find the best model before training (NEW in v0.2.0)

**Basic usage — auto-detects problem type**
```powershell
autoguard getbest data.csv --target label
```

**Regression**
```powershell
autoguard getbest data.csv --target price --problem-type regression
```

**Use more data for better accuracy (slower)**
```powershell
autoguard getbest data.csv --target label --sample-frac 0.3 --n-trials 10
```

**Only benchmark specific models**
```powershell
autoguard getbest data.csv --target label --models random_forest,xgboost,lightgbm
```

**Save the ranked recommendations to a JSON file**
```powershell
autoguard getbest data.csv --target label --output recommendations.json
```

| Option | Default | What it does |
|---|---|---|
| `--target / -t` | required | Target column name |
| `--problem-type` | auto | Force `classification` or `regression` |
| `--sample-frac` | 0.15 | % of data to use for the benchmark (0.01-1.0) |
| `--n-trials` | 5 | Optuna trials per model (more = more accurate, slower) |
| `--models` | all 12 | Comma-separated list of specific models to test |
| `--top` | 12 | Number of models to show in the table |
| `--output / -o` | none | Save results as JSON |

---

### train — Full options
```powershell
autoguard train data.csv --target label --output model.pkl --report --no-diagnose
```

> Note: The positional argument is the CSV file — it comes right after `train`, with no `--data` flag.
> `autoguard train --data data.csv` is **invalid** and will error.

### diagnose — Save results to JSON
```powershell
autoguard diagnose data.csv --target label --output diagnosis.json --no-plots
```

### fix — Clean a raw dataset
```powershell
autoguard fix data.csv --target label --output clean.csv
```

### monitor — Detect drift
```powershell
autoguard monitor new_data.csv --model model.pkl --output drift.json
```

### explain — Generate SHAP feature importance charts
```powershell
autoguard explain --model model.pkl --data data.csv --sample-index 0
```
Saves charts to `autoguard_output\explain\` — open the .png files

> Note: `explain` requires a trained model saved to disk. Run `train` first and confirm `model.pkl` exists
> before running this command.

### report — Generate HTML report from saved model
```powershell
autoguard report --model model.pkl --output report.html
```

### init — Create a starter config.yaml
```powershell
autoguard init --output config.yaml
```
Edit the YAML to change models, trials, thresholds — then pass with `--config`

### Train with custom config
```powershell
autoguard train data.csv --target label --config config.yaml
```

**Open the HTML report in browser**
```powershell
start autoguard_output\report.html
```

**Open all output files in Explorer**
```powershell
explorer autoguard_output
```

---

## 4 — REST API (Serve Your Model Over HTTP)

**Start the API server**
```powershell
autoguard serve model.pkl
```
Keep this terminal open. Open browser at **http://localhost:8000/docs**

**Start on a different port**
```powershell
autoguard serve model.pkl --host 127.0.0.1 --port 9000
```

### API Endpoints — test these in your browser at localhost:8000/docs

| Method | Endpoint | What it does |
|---|---|---|
| GET | `/health` | Is the server up? Is a model loaded? |
| GET | `/model/info` | Model name, feature list, leaderboard |
| POST | `/predict` | Send data → get class predictions |
| POST | `/predict/proba` | Send data → get probability scores |
| POST | `/monitor` | Send new data → get drift severity score |

> Note: `/predict`, `/predict/proba`, and `/monitor` all return **422 Unprocessable Entity** if the
> JSON body doesn't exactly match the feature names and types the model was trained on.
> Check `/model/info` first to see the required feature list.

**Example predict request (paste into terminal while server is running)**
```powershell
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"data\": [{\"age\": 35, \"tenure_months\": 12, \"monthly_fee\": 75, \"num_products\": 2, \"city\": \"NYC\", \"plan\": \"standard\", \"usage_score\": 45}]}"
```
Returns: `{"predictions": [0], "model_name": "lightgbm", "n_samples": 1}`

**Health check via curl**
```powershell
curl http://localhost:8000/health
```

---

## 5 — Fixing Slow Training on Large / Imbalanced Datasets (Windows)

If your dataset is large (100k+ rows) or has severe class imbalance, SMOTE resampling can
balloon training data massively, causing Random Forest to hang or crash worker processes.

**Step 1 — Generate a base config**
```powershell
autoguard init --output config.yaml
```

**Step 2 — Edit `config.yaml` with these recommended settings**
```yaml
automl:
  models:
    - xgboost
    - lightgbm
    - gradient_boosting   # good accuracy, more stable than RF with SMOTE
  n_trials: 10            # reduce Optuna search trials (default is 30)
  n_jobs: 1               # disable multiprocessing — prevents subprocess crashes on Windows
```

**Step 3 — Train using the config**
```powershell
autoguard train clean.csv --target YOUR_COLUMN --output model.pkl --report --config config.yaml
```

XGBoost and LightGBM handle large post-SMOTE datasets far more efficiently than Random Forest,
and `n_jobs: 1` avoids the Windows `loky` / `multiprocessing` subprocess errors entirely.

**Tip: Run `getbest` first to confirm which models suit your data before committing to a full train run.**

---

## 6 — All 12 Available Models

Use these names in your config `models:` list or with `--models` in `getbest`:

| Model name | Algorithm | Best for |
|---|---|---|
| `random_forest` | Random Forest | Robust default, handles noise well |
| `xgboost` | XGBoost | Tabular data champion, large datasets |
| `lightgbm` | LightGBM | Very fast, great for large data |
| `gradient_boosting` | sklearn GBM | High accuracy, more stable than RF with SMOTE |
| `extra_trees` | Extra Trees | Fast ensemble, robust to noise |
| `catboost` | CatBoost | Native categorical support (needs `pip install autoguard-ml[extras]`) |
| `logistic_regression` | Logistic Reg | Fast linear baseline for classification |
| `ridge` | Ridge | Regularized linear (both classification + regression) |
| `lasso` | Lasso | L1 linear with feature selection |
| `svm` | SVM | Strong on small/medium datasets |
| `knn` | K-Nearest Neighbors | Simple, effective on small data |
| `decision_tree` | Decision Tree | Interpretable single-tree baseline |

---

## 7 — Output Files — Everything Lands Here Automatically

| File | What it is |
|---|---|
| `autoguard_output\report.html` | Full pipeline report — open in Chrome/Edge |
| `autoguard_output\models\best_model.pkl` | Your trained model (also copied to `--output` path) |
| `autoguard_output\plots\*.png` | Diagnostic charts (missing values, correlation, etc.) |
| `autoguard_output\explain\*.png` | SHAP feature importance charts |
| `autoguard_output\drift_reports\*.json` | Drift detection logs |
| `autoguard_output\autoguard.log` | Full run log |

---

## Quick Tips

- Always activate your venv first: `venv\Scripts\activate`
- Run `autoguard getbest data.csv --target label` before `train` to save time on bad model choices
- The CSV filename is always a **positional argument** — no `--data` flag (e.g. `autoguard train clean.csv`, not `autoguard train --data clean.csv`)
- Run `train` before `explain`, `report`, or `serve` — those commands all require a saved model file
- If training is slow, use `--config` with `n_jobs: 1` and limit `models:` to `xgboost` and `lightgbm`
- If `autoguard` command isn't found, make sure venv is activated and run `pip install -e .` again
- All output files go to `autoguard_output\` — run `explorer autoguard_output` to see them
- The HTML report works in any browser — just double-click it
- After `fix`, re-run `train` on the cleaned file (`clean.csv`), not the original
- CatBoost model requires: `pip install autoguard-ml[extras]`
