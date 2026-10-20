"""
test_users.py — Validates all 6 clustering models against 5 synthetic users with known
financial profiles. Re-fits each model on the full SCF dataset (features_full.csv), derives
tier labels by scoring cluster centroids against population benchmarks, then predicts a tier
for each test user and compares against expected labels.
"""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

FEATURES_FULL_PATH = "MLAI/clustering/data/processed/features_full.csv"
SCALER_FULL_PATH = "MLAI/clustering/data/processed/scaler_full.pkl"
BENCHMARKS_PATH = "MLAI/clustering/results/benchmarks.json"
OUTPUT_PATH = "MLAI/clustering/results/test_user_predictions.json"

N_CLUSTERS = 3
TIERS = ["High Performer", "Stable", "At Risk"]
ABBREV = {"High Performer": "HP", "Stable": "S", "At Risk": "AR", "Unclassified": "UC"}

FEATURE_COLS = [
    "savings_rate",
    "emergency_fund_ratio",
    "debt_to_income",
    "expense_to_income",
    "healthy_debt_ratio",
]

# ---------------------------------------------------------------------------
# Test users — raw SGD values, no scaling needed
# ---------------------------------------------------------------------------

TEST_USERS = [
    {
        "name": "Alice — Clear High Performer",
        "expected_tier": "High Performer",
        "INCOME": 120000,
        "SAVING": 36000,
        "LIQ": 60000,
        "DEBT": 180000,
        "INSTALL": 160000,
    },
    {
        "name": "Bob — Clear At Risk",
        "expected_tier": "At Risk",
        "INCOME": 30000,
        "SAVING": 600,
        "LIQ": 1500,
        "DEBT": 45000,
        "INSTALL": 5000,
    },
    {
        "name": "Charlie — Borderline Stable",
        "expected_tier": "Stable",
        "INCOME": 60000,
        "SAVING": 7200,
        "LIQ": 15000,
        "DEBT": 20000,
        "INSTALL": 15000,
    },
    {
        "name": "Diana — Borderline At Risk",
        "expected_tier": "At Risk",
        "INCOME": 42000,
        "SAVING": 2100,
        "LIQ": 5000,
        "DEBT": 38000,
        "INSTALL": 8000,
    },
    {
        "name": "Ethan — Edge Case: High Income, Poor Habits",
        "expected_tier": "Stable",
        "INCOME": 150000,
        "SAVING": 9000,
        "LIQ": 25000,
        "DEBT": 60000,
        "INSTALL": 40000,
    },
]

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(user):
    income = user["INCOME"]
    savings_rate = float(np.clip(user["SAVING"] / income, 0, 1))
    emergency_fund_ratio = float(np.clip(user["LIQ"] / (income / 12), 0, 12))
    debt_to_income = float(np.clip(user["DEBT"] / income, 0, 5))
    expense_to_income = 1.0 - savings_rate
    healthy_debt_ratio = (
        float(np.clip(user["INSTALL"] / user["DEBT"], 0, 1))
        if user["DEBT"] > 0
        else 0.0
    )
    return {
        "savings_rate": savings_rate,
        "emergency_fund_ratio": emergency_fund_ratio,
        "debt_to_income": debt_to_income,
        "expense_to_income": expense_to_income,
        "healthy_debt_ratio": healthy_debt_ratio,
    }


# ---------------------------------------------------------------------------
# Tier derivation — score cluster centroids against population benchmarks
# ---------------------------------------------------------------------------

def score_centroid(centroid, thresholds):
    score = 0.0
    if centroid["savings_rate"] >= thresholds["savings_rate"]["p66"]:
        score += 1
    elif centroid["savings_rate"] >= thresholds["savings_rate"]["p33"]:
        score += 0.5
    if centroid["emergency_fund_ratio"] >= thresholds["emergency_fund_ratio"]["p66"]:
        score += 1
    elif centroid["emergency_fund_ratio"] >= thresholds["emergency_fund_ratio"]["p33"]:
        score += 0.5
    if centroid["debt_to_income"] <= thresholds["debt_to_income"]["p33"]:
        score += 1
    elif centroid["debt_to_income"] <= thresholds["debt_to_income"]["p66"]:
        score += 0.5
    if centroid["expense_to_income"] <= thresholds["expense_to_income"]["p33"]:
        score += 1
    elif centroid["expense_to_income"] <= thresholds["expense_to_income"]["p66"]:
        score += 0.5
    if centroid["healthy_debt_ratio"] >= thresholds["healthy_debt_ratio"]["p66"]:
        score += 1
    elif centroid["healthy_debt_ratio"] >= thresholds["healthy_debt_ratio"]["p33"]:
        score += 0.5
    return score


def derive_tier_map(labels, X_norm, scaler, thresholds):
    """
    For each cluster label, compute the mean centroid in original scale,
    score it, and assign tiers by rank (ties broken by savings_rate).
    Returns {cluster_id: tier_string}.
    """
    # np.unique already returns sorted values; -1 is DBSCAN's noise label.
    unique_labels = [lab for lab in np.unique(labels) if lab >= 0]
    scored = []
    for label in unique_labels:
        members = X_norm[labels == label]
        centroid_norm = members.mean(axis=0).reshape(1, -1)
        centroid_orig = scaler.inverse_transform(centroid_norm)[0]
        c = {feat: float(centroid_orig[j]) for j, feat in enumerate(FEATURE_COLS)}
        scored.append((label, c, score_centroid(c, thresholds)))

    ranked = sorted(scored, key=lambda x: (x[2], x[1]["savings_rate"]), reverse=True)
    return {label: TIERS[rank] for rank, (label, _, _) in enumerate(ranked)}


# ---------------------------------------------------------------------------
# K-Medoids helper (mirrors benchmark.py)
# ---------------------------------------------------------------------------

def _get_kmedoids():
    try:
        from sklearn_extra.cluster import KMedoids
        return KMedoids(n_clusters=N_CLUSTERS, random_state=42)
    except ImportError:
        class KMedoidsFallback:
            def __init__(self, n_clusters=3, random_state=42):
                self.n_clusters = n_clusters
                self.random_state = random_state
                self.labels_ = None
                self.cluster_centers_ = None

            def fit_predict(self, X):
                rng = np.random.default_rng(self.random_state)
                n = len(X)
                medoid_idx = rng.choice(n, self.n_clusters, replace=False)
                for _ in range(100):
                    dists = np.array([np.linalg.norm(X - X[m], axis=1) for m in medoid_idx])
                    labels = np.argmin(dists, axis=0)
                    new_medoids = []
                    for k in range(self.n_clusters):
                        members = np.where(labels == k)[0]
                        if len(members) == 0:
                            new_medoids.append(medoid_idx[k])
                            continue
                        intra = np.sum(
                            np.linalg.norm(X[members][:, None] - X[members][None, :], axis=2),
                            axis=1,
                        )
                        new_medoids.append(members[np.argmin(intra)])
                    if np.array_equal(sorted(new_medoids), sorted(medoid_idx)):
                        break
                    medoid_idx = np.array(new_medoids)
                self.labels_ = labels
                self.cluster_centers_ = X[medoid_idx]
                return labels

        return KMedoidsFallback(n_clusters=N_CLUSTERS, random_state=42)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_nearest_centroid(X_train, train_labels, x_user):
    """Assign test point to the cluster whose mean centroid is nearest."""
    unique = [lab for lab in np.unique(train_labels) if lab >= 0]
    centroids = np.array([X_train[train_labels == lab].mean(axis=0) for lab in unique])
    dists = np.linalg.norm(centroids - x_user, axis=1)
    return int(unique[np.argmin(dists)])


def predict_dbscan(X_train, train_labels, x_user, eps=0.3):
    """
    Assign test point to the label of the nearest non-noise training point
    if it lies within eps; otherwise return -1 (Unclassified).
    """
    mask = train_labels >= 0
    X_core = X_train[mask]
    labels_core = train_labels[mask]
    if len(X_core) == 0:
        return -1
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_core)
    dist, idx = nn.kneighbors(x_user.reshape(1, -1))
    if dist[0][0] <= eps:
        return int(labels_core[idx[0][0]])
    return -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    # --- Load data ---
    print("Loading features_full.csv ...")
    X = pd.read_csv(FEATURES_FULL_PATH)[FEATURE_COLS].values

    print("Loading scaler_full.pkl ...")
    with open(SCALER_FULL_PATH, "rb") as f:
        scaler = pickle.load(f)

    print("Loading benchmarks.json ...")
    with open(BENCHMARKS_PATH) as f:
        benchmarks = json.load(f)
    thresholds = benchmarks["thresholds"]

    # --- Engineer and normalise test user features ---
    user_features_raw = [engineer_features(u) for u in TEST_USERS]
    X_users_raw = np.array([[f[col] for col in FEATURE_COLS] for f in user_features_raw])
    X_users = scaler.transform(X_users_raw)

    # --- Fit all 6 models ---
    print("Fitting models on features_full.csv ...")

    # 1. KMeans
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    km_labels = km.fit_predict(X)

    # 2. KMedoids
    kmed = _get_kmedoids()
    kmed_labels = kmed.fit_predict(X)

    # 3. Fuzzy C-Means — cmeans returns (centers, membership, ...)
    fcm_cntr, fcm_membership, *_ = fuzz.cluster.cmeans(
        X.T, c=N_CLUSTERS, m=2, error=0.005, maxiter=1000, seed=42
    )
    fcm_labels = np.argmax(fcm_membership, axis=0)  # hard assignment per training point

    # 4. GMM
    gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=42)
    gmm.fit(X)
    gmm_labels = gmm.predict(X)

    # 5. DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X)

    # 6. Agglomerative
    agg = AgglomerativeClustering(n_clusters=N_CLUSTERS)
    agg_labels = agg.fit_predict(X)

    # --- Derive tier maps ---
    tier_maps = {
        "KMeans":        derive_tier_map(km_labels,      X, scaler, thresholds),
        "KMedoids":      derive_tier_map(kmed_labels,    X, scaler, thresholds),
        "FuzzyCMeans":   derive_tier_map(fcm_labels,     X, scaler, thresholds),
        "GMM":           derive_tier_map(gmm_labels,     X, scaler, thresholds),
        "DBSCAN":        derive_tier_map(dbscan_labels,  X, scaler, thresholds),
        "Agglomerative": derive_tier_map(agg_labels,     X, scaler, thresholds),
    }

    # --- Predict for each test user ---
    model_order = ["KMeans", "KMedoids", "FuzzyCMeans", "GMM", "DBSCAN", "Agglomerative"]
    all_labels_map = {
        "KMeans":        km_labels,
        "KMedoids":      kmed_labels,
        "FuzzyCMeans":   fcm_labels,
        "GMM":           gmm_labels,
        "DBSCAN":        dbscan_labels,
        "Agglomerative": agg_labels,
    }

    predictions = []

    for i, user in enumerate(TEST_USERS):
        x = X_users[i]
        preds = {}
        for model_name in model_order:
            train_labels = all_labels_map[model_name]
            tier_map = tier_maps[model_name]

            if model_name == "KMeans":
                cluster = int(km.predict(x.reshape(1, -1))[0])
            elif model_name == "GMM":
                cluster = int(gmm.predict(x.reshape(1, -1))[0])
            elif model_name == "FuzzyCMeans":
                _, u_pred, *_ = fuzz.cluster.cmeans_predict(
                    x.reshape(-1, 1), fcm_cntr, m=2, error=0.005, maxiter=1000
                )
                cluster = int(np.argmax(u_pred[:, 0]))
            elif model_name == "DBSCAN":
                cluster = predict_dbscan(X, train_labels, x, eps=0.3)
            else:
                cluster = predict_nearest_centroid(X, train_labels, x)

            if cluster == -1:
                preds[model_name] = "Unclassified"
            else:
                preds[model_name] = tier_map.get(cluster, "Unknown")

        correct = {m: (preds[m] == user["expected_tier"]) for m in model_order}
        predictions.append({
            "name": user["name"],
            "expected_tier": user["expected_tier"],
            "features": {k: round(v, 4) for k, v in user_features_raw[i].items()},
            "predictions": preds,
            "correct": correct,
        })

    # --- Print results table ---
    sep = "=" * 90
    col_w = 6
    headers = ["KM", "KMed", "FCM", "GMM", "DBSC", "Agglo"]
    print(f"\n{sep}")
    print("Test User Prediction Results")
    print(sep)
    print(f"{'User':<34} {'Expected':<16}" + "".join(f"{h:>{col_w}}" for h in headers))
    print(sep)
    for p in predictions:
        abbrevs = "".join(
            f"{ABBREV.get(p['predictions'][m], '?'):>{col_w}}" for m in model_order
        )
        print(f"{p['name'][:33]:<34} {p['expected_tier']:<16}{abbrevs}")
    print(sep)

    # Accuracy per model
    acc_parts = []
    accuracy_summary = {}
    for m, h in zip(model_order, headers):
        correct_count = sum(1 for p in predictions if p["correct"][m])
        total = len(predictions)
        acc_parts.append(f"{h}:{correct_count}/{total}")
        accuracy_summary[m] = f"{correct_count}/{total}"
    print("Accuracy (vs expected):  " + "  ".join(acc_parts))
    print(sep)
    print("Legend: HP=High Performer, S=Stable, AR=At Risk, UC=Unclassified")

    # --- Print feature summary ---
    print("\nUser Features (pre-normalisation):")
    feat_header = (
        f"{'User':<34} {'savings_rate':>12} {'emerg_fund':>11}"
        f" {'debt_to_inc':>11} {'expense_to_inc':>14} {'healthy_debt':>12}"
    )
    print(feat_header)
    print("-" * len(feat_header))
    for p in predictions:
        f = p["features"]
        print(
            f"{p['name'][:33]:<34}"
            f" {f['savings_rate']:>12.4f}"
            f" {f['emergency_fund_ratio']:>11.4f}"
            f" {f['debt_to_income']:>11.4f}"
            f" {f['expense_to_income']:>14.4f}"
            f" {f['healthy_debt_ratio']:>12.4f}"
        )

    # --- Save JSON ---
    output = {"test_users": predictions, "accuracy_summary": accuracy_summary}
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
