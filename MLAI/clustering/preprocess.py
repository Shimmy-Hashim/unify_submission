"""
preprocess.py — Cleans and engineers features from the SCF 2022 dataset.

Targets young adults (21–35) and derives 5 financial health features used
for clustering. Values are approximated to SGD (×0.85 USD→SGD conversion)
and normalised with MinMaxScaler.

Reference: Metzler, Zhou & Grace (2021), Quantitative Finance and Economics,
5(3). Cited for the three-tier segmentation precedent only. That paper's own
inputs are self-reported survey responses (mixed categorical/ordinal — savings,
for instance, is answered as one of six bands), not computed continuous ratios,
so it is NOT a source for any of the feature definitions below.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

RAW_PATH = "MLAI/clustering/data/raw/scf2022.csv"
PROCESSED_DIR = "MLAI/clustering/data/processed"
FEATURES_PATH = os.path.join(PROCESSED_DIR, "features.csv")
SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler.pkl")
FEATURES_FULL_PATH = os.path.join(PROCESSED_DIR, "features_full.csv")
SCALER_FULL_PATH = os.path.join(PROCESSED_DIR, "scaler_full.pkl")

# Approximate USD → SGD conversion factor (as of project setup, Apr 2025)
USD_TO_SGD = 0.85

REQUIRED_COLS = ["INCOME", "SAVING", "DEBT", "NETWORTH", "LIQ", "INSTALL", "AGE"]

FEATURE_COLS = [
    "savings_rate",
    "emergency_fund_ratio",
    "debt_to_income",
    "expense_to_income",
    "healthy_debt_ratio",
]


def _engineer_and_save(df, features_path, scaler_path):
    """Shared feature engineering, normalisation, and save logic."""
    # --- Drop rows with missing or zero values ---
    df = df.dropna(subset=REQUIRED_COLS)
    df = df[(df[REQUIRED_COLS] != 0).all(axis=1)]
    print(f"  After dropping nulls/zeros: {df.shape}")

    # --- SGD normalisation for monetary columns ---
    monetary_cols = ["INCOME", "SAVING", "DEBT", "NETWORTH", "LIQ"]
    df[monetary_cols] = df[monetary_cols] * USD_TO_SGD

    # --- Feature engineering ---
    # Savings rate: fraction of income saved each year
    df["savings_rate"] = (df["SAVING"] / df["INCOME"]).clip(0, 1)

    # Emergency fund ratio: months of expenses covered by liquid assets
    # (benchmark: 3–6 months recommended by MAS financial literacy guidelines)
    df["emergency_fund_ratio"] = (df["LIQ"] / (df["INCOME"] / 12)).clip(0, 12)

    # Debt-to-income ratio: total debt relative to annual income
    df["debt_to_income"] = (df["DEBT"] / df["INCOME"]).clip(0, 5)

    # Expense-to-income ratio: complement of savings rate
    df["expense_to_income"] = 1 - df["savings_rate"]

    # Healthy debt ratio: instalment (structured/productive) debt as share of total debt
    # Instalment debt (e.g. mortgage, car loan) is considered healthier than revolving debt
    df["healthy_debt_ratio"] = np.where(
        df["DEBT"] > 0,
        (df["INSTALL"] / df["DEBT"]).clip(0, 1),
        0.0,
    )

    features = df[FEATURE_COLS].copy()

    # --- Normalise to [0, 1] ---
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    features = pd.DataFrame(features_scaled, columns=FEATURE_COLS, index=features.index)

    # --- Save ---
    features.to_csv(features_path, index=False)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved features to   : {features_path}")
    print(f"Saved scaler to     : {scaler_path}")
    print(f"\nOutput shape: {features.shape}")
    print("\nFeature summary stats:")
    print(features.describe().to_string())
    print("\nNull count per feature:")
    print(features.isnull().sum().to_string())
    assert features.isnull().sum().sum() == 0, "Unexpected nulls in output!"
    print("\nNo nulls confirmed. Preprocessing complete.")


def preprocess():
    """Age-filtered pipeline: young adults (21–35) → features.csv."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"Loading raw data from {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)
    print(f"  Raw shape: {df.shape}")

    df = df[df["AGE"].between(21, 35)].copy()
    print(f"  After age filter (21–35): {df.shape}")

    df = df[REQUIRED_COLS].copy()
    _engineer_and_save(df, FEATURES_PATH, SCALER_PATH)


def preprocess_full():
    """Full-dataset pipeline: all ages, no age filter → features_full.csv."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"Loading raw data from {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)
    print(f"  Raw shape: {df.shape}")

    df = df[REQUIRED_COLS].copy()
    _engineer_and_save(df, FEATURES_FULL_PATH, SCALER_FULL_PATH)


if __name__ == "__main__":
    if "--full" in sys.argv:
        preprocess_full()
    else:
        preprocess()
