"""
fit_production_scaler.py — ⚠️ DEPRECATED (July 2026). Fits the OLD single-model production
RobustScaler.

Extracted July 2026 from the now-archived cross_domain_benchmark.py, which was the only
script that fitted backend/ml/artifacts/fitted_scaler.pkl. This is the sole surviving
capability to regenerate that scaler. Nothing imports this module at run time — it is a
deliberate, standalone regeneration tool.

DEPRECATION. The single combined scaler this produces belongs to the superseded
one-model-across-all-features architecture. Layer 2 now runs three independent domain
clusterings, each with its OWN RobustScaler fitted on that domain's own feature set — and
Insurance additionally carries a StandardScaler on age_midpoint, since it is an Approach B
(pooled, age-as-a-feature) model. Those live scalers are:

    backend/ml/artifacts/domain_split/{banking,investment,insurance}_scaler.pkl
    backend/ml/artifacts/domain_split/insurance_age_scaler.pkl

built by MLAI/clustering/build_production_artifacts.py, which fits them the same way the
validation run did. The scaler METHOD choice (RobustScaler, median/IQR — robust to the 10%
deliberate outliers in the synthetic data) is unchanged and still locked; only the
"one scaler over 10 combined features" packaging is superseded.

This module is retained, not deleted: the committed fitted_scaler.pkl would otherwise have
no producing script, and the report cites this as its regeneration path. It fits on the
10-feature set (benchmark_core.CLUSTERING_FEATURES), NOT the 11-feature Layer 2 redesign
and NOT the domain split — that mismatch is now permanent and expected, since the
11-feature re-fit it was waiting for was overtaken by the domain-split decision.

Run (deliberate; OVERWRITES a historical artifact — use build_production_artifacts.py
instead unless you specifically want the old one back):
    python MLAI/clustering/fit_production_scaler.py
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_core import REPO_ROOT, CLUSTERING_FEATURES  # noqa: E402 single source of truth

DATA_PATH = os.path.join(REPO_ROOT, "MLAI/clustering/data/processed/synthetic_users.csv")
SCALER_PATH = os.path.join(REPO_ROOT, "backend/ml/artifacts/fitted_scaler.pkl")


def fit_and_export(data_path: str | None = None, scaler_path: str | None = None):
    """Fit the RobustScaler on the 10 production features and dump it to scaler_path.

    Both paths fall back to the module constants, resolved HERE rather than bound as
    argument defaults. Binding them in the signature captures the values at import time,
    so reassigning the module global has no effect and the call silently writes to the
    real production artifact — the exact failure that hit this project's July 2026 audit
    via benchmark_core.export_ideal_centroids().
    """
    data_path = DATA_PATH if data_path is None else data_path
    scaler_path = SCALER_PATH if scaler_path is None else scaler_path

    df = pd.read_csv(data_path)
    missing = [f for f in CLUSTERING_FEATURES if f not in df.columns]
    if missing:
        raise SystemExit(f"Dataset missing production features: {missing}")
    X = df[CLUSTERING_FEATURES].values
    scaler = RobustScaler().fit(X)
    joblib.dump(scaler, scaler_path)
    print(f"Fitted RobustScaler on {len(df)} rows x {len(CLUSTERING_FEATURES)} features "
          f"-> {scaler_path}")
    return scaler


if __name__ == "__main__":
    fit_and_export()
