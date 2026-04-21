"""
examples/generate_sample_data.py
==================================

Generates sample CSV files for demo and testing:
  - train.csv       : classification dataset (churn prediction)
  - new_data.csv    : shifted data for drift demo
  - regression.csv  : house price regression dataset

Run:
    python examples/generate_sample_data.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def make_churn_dataset(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Binary classification — customer churn prediction."""
    rng = np.random.default_rng(seed)

    age         = rng.integers(18, 75, n).astype(float)
    tenure      = rng.integers(0, 72, n).astype(float)
    monthly_fee = rng.normal(65, 20, n).clip(10, 150)
    num_products = rng.choice([1, 2, 3, 4], n, p=[0.5, 0.3, 0.15, 0.05])
    city        = rng.choice(["NYC", "LA", "Chicago", "Houston", "Phoenix"], n)
    plan        = rng.choice(["basic", "standard", "premium"], n, p=[0.4, 0.4, 0.2])
    usage_score = rng.beta(2, 5, n) * 100

    # Inject missing
    age[rng.choice(n, 80, replace=False)] = np.nan
    monthly_fee[rng.choice(n, 50, replace=False)] = np.nan

    # Churn logic (correlated with tenure + fee)
    churn_prob = (
        0.3
        - 0.004 * tenure
        + 0.002 * monthly_fee
        - 0.05 * (plan == "premium")
        + rng.normal(0, 0.1, n)
    ).clip(0.02, 0.95)
    churn = (rng.uniform(0, 1, n) < churn_prob).astype(int)

    return pd.DataFrame({
        "customer_id":   range(1, n + 1),   # ID column — leakage risk!
        "age":           age,
        "tenure_months": tenure,
        "monthly_fee":   monthly_fee,
        "num_products":  num_products,
        "city":          city,
        "plan":          plan,
        "usage_score":   usage_score,
        "churn":         churn,
    })


def make_drifted_data(train_df: pd.DataFrame, n: int = 400) -> pd.DataFrame:
    """Simulates production data with distribution shift."""
    rng = np.random.default_rng(999)

    # Shift: older customers, higher fees
    age         = rng.integers(45, 80, n).astype(float)   # shifted UP
    tenure      = rng.integers(0, 30, n).astype(float)    # shorter tenures
    monthly_fee = rng.normal(95, 15, n).clip(50, 150)     # shifted UP
    num_products = rng.choice([1, 2], n, p=[0.8, 0.2])
    city        = rng.choice(["NYC", "Miami", "Seattle"], n)  # different cities
    plan        = rng.choice(["basic", "standard", "premium"], n, p=[0.6, 0.3, 0.1])
    usage_score = rng.beta(5, 2, n) * 100  # distribution flipped

    return pd.DataFrame({
        "age":           age,
        "tenure_months": tenure,
        "monthly_fee":   monthly_fee,
        "num_products":  num_products,
        "city":          city,
        "plan":          plan,
        "usage_score":   usage_score,
    })


def make_house_price_dataset(n: int = 800, seed: int = 7) -> pd.DataFrame:
    """Regression dataset — house price prediction."""
    rng = np.random.default_rng(seed)

    sqft       = rng.integers(500, 5000, n).astype(float)
    bedrooms   = rng.choice([1, 2, 3, 4, 5, 6], n, p=[0.05, 0.2, 0.4, 0.25, 0.08, 0.02])
    bathrooms  = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], n)
    age_years  = rng.integers(0, 100, n).astype(float)
    garage     = rng.choice([0, 1, 2], n, p=[0.3, 0.5, 0.2])
    neighborhood = rng.choice(["suburb", "urban", "rural"], n, p=[0.5, 0.35, 0.15])
    has_pool   = rng.choice([0, 1], n, p=[0.8, 0.2])

    # Inject skewness + missing
    sqft_skewed = np.exp(rng.normal(7, 0.5, n))  # log-normal
    sqft_skewed[rng.choice(n, 40, replace=False)] = np.nan

    price = (
        150 * sqft
        + 10000 * bedrooms
        + 8000 * bathrooms
        - 500 * age_years
        + 15000 * garage
        + 20000 * has_pool
        + (neighborhood == "urban") * 30000
        + rng.normal(0, 20000, n)
    ).clip(50000, 2_000_000)

    return pd.DataFrame({
        "sqft":          sqft,
        "sqft_log_norm": sqft_skewed,
        "bedrooms":      bedrooms,
        "bathrooms":     bathrooms,
        "age_years":     age_years,
        "garage_spots":  garage,
        "neighborhood":  neighborhood,
        "has_pool":      has_pool,
        "price":         price.round(0),
    })


def main() -> None:
    out = Path("examples/data")
    out.mkdir(parents=True, exist_ok=True)

    print("Generating sample datasets...")

    train = make_churn_dataset(1500)
    train.to_csv(out / "train.csv", index=False)
    print(f"  ✓ train.csv         → {len(train):,} rows, {len(train.columns)} cols")

    drift = make_drifted_data(train, 400)
    drift.to_csv(out / "new_data.csv", index=False)
    print(f"  ✓ new_data.csv      → {len(drift):,} rows (drifted distribution)")

    house = make_house_price_dataset(800)
    house.to_csv(out / "house_prices.csv", index=False)
    print(f"  ✓ house_prices.csv  → {len(house):,} rows, {len(house.columns)} cols")

    print(f"\nAll datasets saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
