"""Hopkins statistic on the current synthetic dataset (new script, October 2026).

Why this exists: the brief quotes Hopkins H = 0.81, but the only saved value is
results/benchmark_results.json (H = 0.8118, 2026-05-30, 7000 users, the OLD 10-feature
set, RobustScaler), computed on a dataset that has since been regenerated. This script
recomputes it, in three steps:

  1. CONTROL: re-run the original method on synthetic_users_pre_ablation.csv with the
     original 10 features, RobustScaler fitted on all rows, sampling_ratio=0.1, seed 42.
     If that does not reproduce 0.8118 the script stops: the method then differs from
     the original and the later numbers would not be comparable.
  2. CURRENT, Config A: the 11 continuous benchmark features, RobustScaler on all rows.
  3. CURRENT, production: the 10 production clustering features (Config A minus
     net_worth_to_income).
  Each reported at the original seed (42) and over 20 other seeds (1-20).

hopkins_statistic() below is copied VERBATIM from
results/archive/comprehensive_10feature_5algo/cross_domain_benchmark.py (the archived
script is not imported, run or modified). Reads the two CSVs, never writes them.
Refuses to overwrite its output.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "processed")
CURRENT_CSV = os.path.join(DATA_DIR, "synthetic_users.csv")
PRE_ABLATION_CSV = os.path.join(DATA_DIR, "synthetic_users_pre_ablation.csv")
OUT_PATH = os.path.join(HERE, "results", "hopkins_2026-10-10.json")

ORIGINAL_VALUE = 0.8118        # results/benchmark_results.json, hopkins_statistic.value
ORIGINAL_SAMPLING_RATIO = 0.1  # cross_domain_benchmark.py main(): sampling_ratio=0.1
ORIGINAL_SEED = 42             # cross_domain_benchmark.py: RANDOM_STATE = 42
OTHER_SEEDS = list(range(1, 21))

# The old 10 features, verbatim from cross_domain_benchmark.py CLUSTERING_FEATURES.
OLD_10 = [
    "savings_rate", "emergency_fund_months", "dti", "net_worth_to_income",
    "investment_rate", "portfolio_concentration", "risk_age_deviation",
    "life_coverage_ratio", "ci_coverage_ratio", "hospitalisation_coverage",
]
# Config A, verbatim from comprehensive_benchmark_11algo.py FEATURES_11.
CONFIG_A_11 = [
    "savings_rate", "emergency_fund_months", "dti", "net_worth_to_income",
    "investment_rate", "portfolio_concentration", "risk_age_deviation",
    "life_coverage_ratio", "ci_coverage_ratio", "healthy_debt_ratio",
    "insurance_spend_ratio",
]
PRODUCTION_10 = [f for f in CONFIG_A_11 if f != "net_worth_to_income"]


# --- COPIED VERBATIM from cross_domain_benchmark.py (hopkins_statistic) ---
def hopkins_statistic(X, sampling_ratio=0.1, random_state=42):
    """
    Interpretation for this formulation (H = u_dist / (u_dist + w_dist)):
      H close to 1   -> strong clustering tendency (genuine cluster structure)
      H close to 0.5 -> random (no clustering tendency)
      H close to 0   -> regularly / uniformly spaced points (anti-clustered),
                        NOT genuine clusters
    Values well above 0.7 indicate clustering tendency for this dataset.
    """
    n = X.shape[0]
    m = int(n * sampling_ratio)
    rng = np.random.RandomState(random_state)

    idx = rng.choice(n, m, replace=False)
    sample = X[idx]

    mins, maxs = X.min(axis=0), X.max(axis=0)
    random_pts = rng.uniform(mins, maxs, size=(m, X.shape[1]))

    nn = NearestNeighbors(n_neighbors=2).fit(X)

    w_dist, _ = nn.kneighbors(sample)
    w_dist = w_dist[:, 1]  # exclude self

    u_dist, _ = nn.kneighbors(random_pts)
    u_dist = u_dist[:, 0]

    H = u_dist.sum() / (u_dist.sum() + w_dist.sum())
    return H


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hopkins_for(df, features, seed):
    X = RobustScaler().fit_transform(df[features].values)
    return float(hopkins_statistic(X, sampling_ratio=ORIGINAL_SAMPLING_RATIO, random_state=seed))


def summarise(df, features):
    at_orig = hopkins_for(df, features, ORIGINAL_SEED)
    vals = [hopkins_for(df, features, s) for s in OTHER_SEEDS]
    return {
        "n_rows": int(len(df)),
        "features": features,
        "sample_size": int(len(df) * ORIGINAL_SAMPLING_RATIO),
        "value_at_original_seed_42": round(at_orig, 6),
        "other_seeds": OTHER_SEEDS,
        "values_other_seeds": [round(v, 6) for v in vals],
        "mean_over_20_seeds": round(float(np.mean(vals)), 6),
        "min_over_20_seeds": round(float(np.min(vals)), 6),
        "max_over_20_seeds": round(float(np.max(vals)), 6),
    }


def main():
    if os.path.exists(OUT_PATH):
        sys.exit(f"Refusing to overwrite {OUT_PATH}")
    cur_hash_before = sha256(CURRENT_CSV)
    pre_hash = sha256(PRE_ABLATION_CSV)

    # 1. CONTROL
    pre = pd.read_csv(PRE_ABLATION_CSV)
    assert len(pre) == 7000, len(pre)
    control = summarise(pre, OLD_10)
    ctl = control["value_at_original_seed_42"]
    print(f"CONTROL (pre-ablation CSV, old 10 features, seed 42): H = {ctl:.4f}  "
          f"(original saved value {ORIGINAL_VALUE})")
    if round(ctl, 4) != ORIGINAL_VALUE:
        print("CONTROL DID NOT REPRODUCE the original value. Stopping before the "
              "current-data runs: the method differs from the original.")
        sys.exit(2)

    # 2/3. CURRENT DATA
    cur = pd.read_csv(CURRENT_CSV)
    assert len(cur) == 7000, len(cur)
    cfg_a = summarise(cur, CONFIG_A_11)
    prod = summarise(cur, PRODUCTION_10)
    print(f"CURRENT Config A (11 features): H(seed 42) = {cfg_a['value_at_original_seed_42']:.4f}  "
          f"mean/min/max over 20 seeds = {cfg_a['mean_over_20_seeds']:.4f} / "
          f"{cfg_a['min_over_20_seeds']:.4f} / {cfg_a['max_over_20_seeds']:.4f}")
    print(f"CURRENT production (10 features): H(seed 42) = {prod['value_at_original_seed_42']:.4f}  "
          f"mean/min/max over 20 seeds = {prod['mean_over_20_seeds']:.4f} / "
          f"{prod['min_over_20_seeds']:.4f} / {prod['max_over_20_seeds']:.4f}")

    cur_hash_after = sha256(CURRENT_CSV)
    assert cur_hash_before == cur_hash_after, "synthetic_users.csv changed during the run"
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    out = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": "Hopkins statistic on current data; control reproduces the saved 0.8118",
        "method": {
            "function": "hopkins_statistic, copied verbatim from "
                        "results/archive/comprehensive_10feature_5algo/cross_domain_benchmark.py",
            "sampling_ratio": ORIGINAL_SAMPLING_RATIO,
            "original_seed": ORIGINAL_SEED,
            "scaler": "RobustScaler fitted on all 7000 rows",
            "nearest_neighbours": "sklearn NearestNeighbors(n_neighbors=2), euclidean",
        },
        "inputs": {
            "synthetic_users.csv_sha256": cur_hash_after,
            "synthetic_users.csv_sha256_before_run": cur_hash_before,
            "synthetic_users_pre_ablation.csv_sha256": pre_hash,
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                        "git_head": commit},
        "original_saved_value": {"value": ORIGINAL_VALUE,
                                 "file": "MLAI/clustering/results/benchmark_results.json",
                                 "run_timestamp": "2026-05-30T13:11:48.615215+00:00"},
        "control_pre_ablation_old_10_features": control,
        "control_reproduces_original": True,
        "current_config_a_11_features": cfg_a,
        "current_production_10_features": prod,
    }
    with open(OUT_PATH, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
