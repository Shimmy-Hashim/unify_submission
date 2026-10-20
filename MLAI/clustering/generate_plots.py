"""
generate_plots.py — Regenerate all exploratory/analysis plots from the CURRENT analysis.

Clean-slate redesign (July 2026). The old exploratory plots (produced by the archived
cross_domain_benchmark.py on the old 10-feature data) were moved to
results/archive/comprehensive_10feature_5algo/plots/; this script regenerates their
equivalents from the current 11-continuous-feature dataset and the 12-algorithm
benchmark results.

Separate from comprehensive_benchmark_11algo.py on purpose: plotting is fast and should
be re-runnable without re-running the ~5-minute benchmark sweep. The k-selection and
silhouette plots read the already-computed results (comprehensive_results.json,
algorithm_metrics.csv) rather than recomputing — as intended.

Output (flat in results/plots/, alongside the two canonical benchmark plots
k_comparison.png and algorithm_heatmap.png — a flat layout is cleaner than the old
eda/ + k_search/ subdirs for this handful of files):
    correlation_heatmap.png   — 11-feature correlation (normal-profile users)
    feature_distributions.png — histograms of all 11 features (corrected distributions)
    dendrogram.png            — Ward-linkage hierarchical dendrogram (sampled)
    k_selection.png           — selection score vs k, both approaches (elbow equivalent)
    silhouette_approachA.png  — silhouette vs k per algorithm, Approach A
    silhouette_approachB.png  — silhouette vs k per algorithm, Approach B

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/generate_plots.py
"""

import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend; must be selected before pyplot is imported
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import RobustScaler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(REPO_ROOT, "MLAI/clustering/data/processed/synthetic_users.csv")
RESULTS_DIR = os.path.join(REPO_ROOT, "MLAI/clustering/results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
RESULTS_JSON = os.path.join(RESULTS_DIR, "comprehensive_results.json")
METRICS_CSV = os.path.join(RESULTS_DIR, "algorithm_metrics.csv")

# The current 11 continuous Layer 2 features (mirrors scoring.py
# LAYER2_CLUSTERING_FEATURES and comprehensive_benchmark_11algo.FEATURES_11).
FEATURES_11 = [
    "savings_rate", "emergency_fund_months", "dti", "net_worth_to_income",
    "investment_rate", "portfolio_concentration", "risk_age_deviation",
    "life_coverage_ratio", "ci_coverage_ratio", "healthy_debt_ratio",
    "insurance_spend_ratio",
]
SEED = 42
K_RANGE = [2, 3, 4, 5, 6]


def _save(fig, name):
    """Write one figure into PLOTS_DIR at the shared dpi, then close it."""
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def plot_correlation_heatmap(df):
    """11-feature Pearson correlation on normal-profile users (cleaner signal, same
    convention as the old EDA plot)."""
    normal = df[df["profile_type"] == "normal"]
    corr = normal[FEATURES_11].corr().values
    n = len(FEATURES_11)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(FEATURES_11, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(FEATURES_11, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr[i, j]) > 0.55 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature correlation — 11 continuous features, normal-profile users "
                 f"(n={len(normal)})", fontsize=12)
    fig.tight_layout()
    _save(fig, "correlation_heatmap.png")
    return corr


def plot_feature_distributions(df):
    """Histograms of all 11 features (should visibly show the corrected shapes:
    investment_rate zero-spike from the hurdle model, revised net_worth, PC taper)."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    for ax, feat in zip(axes.ravel(), FEATURES_11):
        vals = df[feat].values
        ax.hist(vals, bins=40, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(vals.mean(), color="crimson", linestyle="--", linewidth=1.4,
                   label=f"mean={vals.mean():.3f}")
        ax.set_title(feat, fontsize=11)
        ax.legend(fontsize=8)
    axes.ravel()[-1].axis("off")  # 12th cell unused (11 features)
    fig.suptitle(f"Feature distributions — current 11 continuous features (n={len(df)})",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "feature_distributions.png")


def plot_dendrogram(df):
    """Ward-linkage dendrogram on a reproducible sample (a 7000-leaf dendrogram is
    unreadable; Ward matches the Agglomerative config in the slate. BIRCH builds a
    CF-tree, not a comparable dendrogram, so one Ward dendrogram represents the
    hierarchical family)."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=2000, replace=False)
    X = RobustScaler().fit_transform(df[FEATURES_11].values)[idx]
    Z = linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=(16, 7))
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title("Ward-linkage hierarchical dendrogram (2000-user sample, 11 features, "
                 "RobustScaler; top 30 merges)", fontsize=12)
    ax.set_xlabel("cluster (or sample count in parentheses)")
    ax.set_ylabel("Ward distance")
    fig.tight_layout()
    _save(fig, "dendrogram.png")


def plot_k_selection(results):
    """Selection score vs k, both approaches — the new methodology's k-selection curve
    (the elbow equivalent), read directly from comprehensive_results.json."""
    sel_a = results["approach_a"]["selection_scores_by_k"]
    sel_b = results["approach_b"]["selection_scores_by_k"]
    ya = [sel_a[str(k)] for k in K_RANGE]
    yb = [sel_b[str(k)] for k in K_RANGE]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(K_RANGE, ya, "o-", color="#4C72B0", label="Approach A (age-stratified)")
    ax.plot(K_RANGE, yb, "s-", color="#DD8452", label="Approach B (single model)")
    ka = max(sel_a, key=sel_a.get)
    kb = max(sel_b, key=sel_b.get)
    ax.axvline(int(ka), color="#4C72B0", ls=":", alpha=0.6)
    ax.axvline(int(kb), color="#DD8452", ls=":", alpha=0.6)
    ax.set_xticks(K_RANGE)
    ax.set_xlabel("k")
    ax.set_ylabel("selection score (mean of 4 norm. metrics)")
    ax.set_title(f"k selection — selection score vs k (A peaks k={ka}, B peaks k={kb})", fontsize=12)
    ax.legend()
    fig.tight_layout()
    _save(fig, "k_selection.png")


def plot_silhouette_by_approach(metrics):
    """Silhouette vs k per algorithm, one figure per approach — from algorithm_metrics.csv
    (mirrors the old silhouette_approachA/B plots, now covering all 9 k-based algos)."""
    for approach in ("A", "B"):
        sub = metrics[metrics["approach"] == approach]
        fig, ax = plt.subplots(figsize=(11, 7))
        for algo in sorted(sub["algorithm"].unique()):
            a = sub[sub["algorithm"] == algo].sort_values("k")
            ax.plot(a["k"], a["silhouette"], "o-", label=algo, alpha=0.85)
        ax.set_xticks(K_RANGE)
        ax.set_xlabel("k")
        ax.set_ylabel("silhouette")
        ax.set_title(f"Silhouette vs k by algorithm — Approach {approach}", fontsize=12)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _save(fig, f"silhouette_approach{approach}.png")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    with open(RESULTS_JSON) as f:
        results = json.load(f)
    metrics = pd.read_csv(METRICS_CSV)

    print("Regenerating plots from current data + 12-algorithm results...")
    corr = plot_correlation_heatmap(df)
    plot_feature_distributions(df)
    plot_dendrogram(df)
    plot_k_selection(results)
    plot_silhouette_by_approach(metrics)

    # Sanity print: confirm the correlation heatmap shows the audit-expected relationships.
    idx = {f: i for i, f in enumerate(FEATURES_11)}

    def rho(a, b):
        return round(float(corr[idx[a], idx[b]]), 3)

    print("\nCorrelation sanity (normal-profile users):")
    print(f"  net_worth ~ savings_rate:      {rho('net_worth_to_income','savings_rate')} (expect +)")
    print(f"  net_worth ~ investment_rate:   {rho('net_worth_to_income','investment_rate')} (expect +)")
    print(f"  net_worth ~ dti:               {rho('net_worth_to_income','dti')} (expect -)")
    print(f"  life_cov ~ ci_cov:             {rho('life_coverage_ratio','ci_coverage_ratio')} (expect +)")
    print(f"  life_cov ~ insurance_spend:    {rho('life_coverage_ratio','insurance_spend_ratio')} (expect +)")
    print("Done.")


if __name__ == "__main__":
    main()
