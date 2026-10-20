"""
benchmarks.py — Stage 1: Derives absolute financial health thresholds from the full SCF 2022
dataset (all ages, 22,975 rows). These thresholds define what "At Risk", "Building", and
"Stable" mean in population terms and are used by validate_clusters.py.
"""

import os
import json
import pandas as pd
import numpy as np

RAW_PATH = "MLAI/clustering/data/raw/scf2022.csv"
RESULTS_DIR = "MLAI/clustering/results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "benchmarks.json")

USD_TO_SGD = 0.85

MONETARY_COLS = ["INCOME", "SAVING", "DEBT", "LIQ", "INSTALL"]

FEATURES = [
    "savings_rate",
    "emergency_fund_ratio",
    "debt_to_income",
    "expense_to_income",
    "healthy_debt_ratio",
]

SG_BENCHMARKS = {
    "savings_rate": 0.20,
    "emergency_fund_ratio": 3.0,
    "debt_to_income": 0.36,
    "expense_to_income": None,
    "healthy_debt_ratio": None,
}


def compute_benchmarks():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading raw data from {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)
    print(f"  Raw shape: {df.shape}")

    # Drop rows where INCOME is null or zero (primary denominator — avoids division by zero)
    df = df.dropna(subset=["INCOME"])
    df = df[df["INCOME"] != 0].copy()
    print(f"  After dropping null/zero INCOME: {df.shape}")

    # Apply SGD scaling to monetary columns
    for col in MONETARY_COLS:
        df[col] = df[col] * USD_TO_SGD

    # Feature engineering (same definitions as preprocess.py)
    df["savings_rate"] = (df["SAVING"] / df["INCOME"]).clip(0, 1)
    df["emergency_fund_ratio"] = (df["LIQ"] / (df["INCOME"] / 12)).clip(0, 12)
    df["debt_to_income"] = (df["DEBT"] / df["INCOME"]).clip(0, 5)
    df["expense_to_income"] = 1 - df["savings_rate"]
    df["healthy_debt_ratio"] = np.where(
        df["DEBT"] > 0,
        (df["INSTALL"] / df["DEBT"]).clip(0, 1),
        0.0,
    )

    n_rows = len(df)
    print(f"\nComputing percentile thresholds on {n_rows:,} rows ...\n")

    thresholds = {}
    for feature in FEATURES:
        p33 = float(df[feature].quantile(0.33))
        p66 = float(df[feature].quantile(0.66))
        sg = SG_BENCHMARKS[feature]
        thresholds[feature] = {
            "p33": round(p33, 4),
            "p66": round(p66, 4),
            "sg_benchmark": sg,
        }

    # Print readable summary table
    header = f"{'Feature':<25} {'P33':>8} {'P66':>8} {'SG Benchmark':>14}"
    print(header)
    print("-" * len(header))
    for feature in FEATURES:
        sg = thresholds[feature]["sg_benchmark"]
        sg_str = f"{sg:.2f}" if sg is not None else "-"
        print(
            f"{feature:<25} {thresholds[feature]['p33']:>8.4f} {thresholds[feature]['p66']:>8.4f} {sg_str:>14}"
        )

    # Print comparison against SG benchmarks
    print("\nSG Benchmark comparison:")
    for feature in FEATURES:
        sg = SG_BENCHMARKS[feature]
        if sg is None:
            continue
        p33 = thresholds[feature]["p33"]
        p66 = thresholds[feature]["p66"]
        print(f"  {feature}: p33={p33:.4f}, p66={p66:.4f}, SG={sg}")
        if p33 <= sg <= p66:
            print("    -> SG benchmark falls within middle tier (p33-p66)")
        elif sg < p33:
            print("    -> SG benchmark is BELOW p33 (population generally exceeds SG target)")
        else:
            print("    -> SG benchmark is ABOVE p66 (most of population below SG target)")

    # Build output
    output = {
        "source": f"SCF 2022 full dataset ({n_rows} rows)",
        "sgd_scaling_factor": USD_TO_SGD,
        "thresholds": thresholds,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved benchmarks to: {OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    compute_benchmarks()
