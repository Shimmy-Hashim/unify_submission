"""
investment_shipped_tier_assignments.py — per-user Investment tiers for the SHIPPED model
(Unify, CCDS25-1170, October 2026).

WHY THIS EXISTS
  results/domain_split/domain_split_tier_assignments.csv was written by
  domain_split_clustering.py in July 2026, BEFORE Investment was recalibrated. Its
  `tier_investment` column holds five tiers from the superseded B|agglomerative|5 and must
  not be used for Investment. The shipped model is A|fuzzy_cmeans|2 (benchmark variant V2:
  risk_age_deviation kept as a clustering feature, dropped from the benchmark set).

  This script writes the per-user column the old file cannot provide, so that
  backend/scripts/verify_tier_assignment.py can compare Investment user for user, as it
  already does for Banking and Insurance.

INDEPENDENCE
  Research-stage chain only: the V2 RobustScaler on INV_FEATS_FULL fitted on the whole
  population, run_kbased("fuzzy_cmeans", 2, X_band) per age band with SEED 42, the V2
  benchmark set from install_bench(False), and assign_tiers() for the health-rank relabel.
  It imports nothing from backend/ and reads nothing under backend/ml/artifacts/ — the
  tier_assignment.py replay is exactly what this file is meant to be checked AGAINST.

  Only Investment is fitted. Banking and Insurance are not refitted, and no existing result
  file is read or modified.

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/investment_shipped_tier_assignments.py
    ... --out PATH        write somewhere else (used to check the output is deterministic)
"""

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domain_split_clustering as ds  # noqa: E402
from domain_split_clustering import DATA_PATH, OUT_DIR, assign_tiers  # noqa: E402
from comprehensive_benchmark_11algo import run_kbased, SEED  # noqa: E402
from risk_age_diagnostic import INV_FEATS_FULL, install_bench  # noqa: E402

KEY = "A|fuzzy_cmeans|2"
OUT_PATH = os.path.join(OUT_DIR, "domain_split_investment_shipped_tiers.csv")
COLUMNS = ["user_id", "age_band", "profile_type", "tier_investment", "coarse_investment"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    with open(DATA_PATH, "rb") as fh:
        dataset_sha = hashlib.sha256(fh.read()).hexdigest()
    df = pd.read_csv(DATA_PATH)

    import scipy
    import sklearn
    import skfuzzy
    print("=" * 100)
    print(f"INVESTMENT SHIPPED-MODEL TIER ASSIGNMENT — model {KEY}, benchmark variant V2")
    print(f"  dataset sha256 {dataset_sha}  ({len(df)} users)")
    print(f"  libraries      numpy {np.__version__}  pandas {pd.__version__}  "
          f"scipy {scipy.__version__}  scikit-learn {sklearn.__version__}  "
          f"scikit-fuzzy {skfuzzy.__version__}")
    print(f"  seed           {SEED}")
    print(f"  run (UTC)      {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"  clustering features: {INV_FEATS_FULL}")
    print("=" * 100)

    # The same construction as domain_split_clustering.run_spec_grid: one RobustScaler
    # fitted on the whole population, applied band by band.
    scaler = RobustScaler().fit(df[INV_FEATS_FULL])
    band_labels = {}
    for b in range(1, 6):
        sub = df[df["age_band"] == b]
        band_labels[b] = run_kbased("fuzzy_cmeans", 2, scaler.transform(sub[INV_FEATS_FULL]))

    install_bench(False)   # V2: investment_rate, portfolio_concentration, has_investments
    print(f"  benchmark set (V2): {ds.BENCH_VARS['investment']}")
    rank, kfound, coarse = assign_tiers(df, KEY, {KEY: band_labels}, "investment")
    assert kfound == 2, f"expected 2 tiers, found {kfound}"

    out = df[["user_id", "age_band", "profile_type"]].copy()
    out["tier_investment"] = rank.astype(int).to_numpy()
    out["coarse_investment"] = coarse.to_numpy()
    out = out[COLUMNS]
    out.to_csv(args.out, index=False, lineterminator="\n")

    shares = out["tier_investment"].value_counts(normalize=True).sort_index()
    print(f"\n  tier shares (%): {[round(100 * float(s), 1) for s in shares]}")
    print("  per band, tier counts:")
    for b in range(1, 6):
        s = out[out["age_band"] == b]["tier_investment"].value_counts().sort_index()
        print(f"    band {b}: " + "  ".join(f"tier {t}={int(n)}" for t, n in s.items()))
    print(f"\n  wrote {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
