"""
create_adult_csv.py
-------------------
Downloads the UCI Adult / Census-Income dataset (~49 K rows, 15 cols).
This is one of the most widely-used real-world classification benchmarks:
  - Income: >50K vs <=50K
  - Mix of numeric + categorical features
  - Real imbalance, outliers, and missing values — perfect for AutoGuard demos.
"""
import sys
import urllib.request
import pathlib
import pandas as pd

TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

OUT = pathlib.Path("adult.csv")

print("Downloading Adult/Census-Income dataset …")
try:
    urllib.request.urlretrieve(TRAIN_URL, "adult_train.data")
    urllib.request.urlretrieve(TEST_URL,  "adult_test.data")
except Exception as e:
    print(f"❌ Download failed: {e}")
    sys.exit(1)

train = pd.read_csv(
    "adult_train.data",
    names=COLS,
    skipinitialspace=True,
    na_values=["?"],
)
test = pd.read_csv(
    "adult_test.data",
    names=COLS,
    skipinitialspace=True,
    na_values=["?"],
    skiprows=1,          # first row is a comment in the test file
)

# Normalise the income label (test file has trailing '.')
test["income"] = test["income"].str.replace(".", "", regex=False)

df = pd.concat([train, test], ignore_index=True)

# Clean up
pathlib.Path("adult_train.data").unlink(missing_ok=True)
pathlib.Path("adult_test.data").unlink(missing_ok=True)

df.to_csv(OUT, index=False)
print(
    f"✅ adult.csv created  —  {df.shape[0]:,} rows × {df.shape[1]} cols\n"
    f"   Target: 'income'  |  Classes: {sorted(df['income'].dropna().unique())}\n"
    f"   Missing values:   {df.isna().sum().sum():,} cells\n"
    f"   File size:        {OUT.stat().st_size / 1024:.0f} KB"
)
