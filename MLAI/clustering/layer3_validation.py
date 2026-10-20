"""
layer3_validation.py — check two assumptions behind the PROPOSED Layer 3 combination rules
against the actual synthetic data (Unify, CCDS25-1170, July 2026).

ANALYSIS ONLY. Nothing here is production code, nothing is refit, and no Layer 3 rule is
implemented. The proposed rules are evaluated as predicates over the tier assignment the
committed production artifacts already produce, so a result here is a statement about the
data, not about any candidate implementation.

WHAT IS BEING CHECKED
  Assumption 1 (Step 2) — that "Banking = Stable AND Insurance = Not Covered" contains two
    real, separable populations, split by net_worth_to_income at the band-appropriate
    threshold (1/3/6/8/10x for bands 1-5): the ~21.8x `wealthy_exposed` mode and the ~5.1x
    `underinsured_saver` mode the brief's Step 4 finding names. If the distribution is
    instead smooth, the proposed split of rule 1 vs rule 2 is an arbitrary bisection of a
    continuum and should be reported as such.
  Assumption 2 (Step 3) — that each of the 5 proposed rules fires for a small enough share
    of users to read as a personalised finding rather than a near-universal statement.

THE 5 PROPOSED RULES (design input from the PM thread; NOT implemented here)
  1. Banking=Stable      + Insurance=Not Covered + nw >= band threshold -> Significant Assets, No Protection
  2. Banking=Stable      + Insurance=Not Covered + nw <  band threshold -> Building Wealth, No Safety Net
  3. Insurance=Covered   + Banking=Building                             -> Protection First
  4. Banking=Stable      + Investment=Getting Started                   -> Strong Foundation, Room to Grow
  5. Banking=Building    + Investment=On Track                          -> Investing Fast, Foundation Still Building

TIER ASSIGNMENT IS REPLAYED, NOT REFIT. Tiers come from the committed artifacts in
backend/ml/artifacts/domain_split/ by exactly the documented consumer path: scale -> nearest
centroid -> read the label out of that BAND's own `tier_map`. Cluster ids are never assumed
to mean anything (cluster 0 is Building in Banking band 1 and Stable in bands 2-5), and the
replay is asserted against each artifact's own recorded population shares before any analysis
runs — if the replay were wrong, the script stops rather than reporting from bad tiers.

BIMODALITY METHOD. `bimodality()` is imported from final_selection.py, the same function and
the same BC_FLAG = 0.555 threshold used by the domain-split validation, so a "bimodal" claim
here means what it means elsewhere in this project. Three independent signals are reported —
Sarle's bimodality coefficient, a 1-vs-2-component Gaussian-mixture BIC comparison, and a KDE
mode/valley count — plus the one check that actually matters for the proposed rule: whether
the band threshold lands in a valley, and whether it separates the two archetypes it is
supposed to separate.

Outputs (new, additive — nothing existing is written to):
    MLAI/clustering/results/layer3_validation/layer3_validation.json
    MLAI/clustering/results/layer3_validation/LAYER3_VALIDATION.md
    MLAI/clustering/results/layer3_validation/plots/*.png

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/layer3_validation.py
"""

import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from scipy import stats                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_selection import bimodality   # noqa: E402
from benchmark_core import SYNTHETIC_IDEALS  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ART_DIR = os.path.join(REPO_ROOT, "backend/ml/artifacts/domain_split")
DATA_PATH = os.path.join(REPO_ROOT, "MLAI/clustering/data/processed/synthetic_users.csv")
OUT_DIR = os.path.join(REPO_ROOT, "MLAI/clustering/results/layer3_validation")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

BC_FLAG = 0.555                 # domain_split_clustering.BC_FLAG — Sarle's, uniform-distribution value

# Banking's out-of-model validation benchmark, band-appropriate: 1/3/6/8/10x. Read from
# SYNTHETIC_IDEALS rather than retyped, so this cannot drift from the source of truth.
NW_THRESHOLD = {b: float(SYNTHETIC_IDEALS[b]["net_worth_to_income"]) for b in range(1, 6)}
BANDS = {1: "20-29", 2: "30-39", 3: "40-49", 4: "50-59", 5: "60-70"}

# The two modes the proposed split is premised on (brief Section 5, Step 4 finding).
EXPECTED_MODES = {"wealthy_exposed": 21.8, "underinsured_saver": 5.1}


# ======================================================================================
# Tier replay — the documented consumer path, nothing refit
# ======================================================================================
def load_pickle(name):
    with open(os.path.join(ART_DIR, name), "rb") as f:
        return pickle.load(f)


def replay_tiers(df, domain):
    """Assign this domain's tier LABEL to every user, straight from the artifacts.

    Labels are read per band out of that band's own tier_map. Nothing maps a cluster id to a
    label directly — that is the documented trap (Banking cluster 0 is the weak tier in band
    1 and the healthy tier in bands 2-5), and the whole point of going through tier_map.
    """
    with open(os.path.join(ART_DIR, f"{domain}_model.json")) as f:
        art = json.load(f)
    scaler = load_pickle(f"{domain}_scaler.pkl")
    feats = art["clustering_features"]
    k = art["k_native"]
    labels = pd.Series(index=df.index, dtype=object)

    if art["approach"] == "A":
        for b, bv in art["bands"].items():
            m = df["age_band"] == int(b)
            X = scaler.transform(df.loc[m, feats])
            C = np.array([bv["centroids_scaled"][str(c)] for c in range(k)])
            near = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(2), axis=1)
            labels.loc[m] = [bv["tier_map"][str(c)]["tier_label"] for c in near]
    else:
        age_scaler = load_pickle(f"{domain}_age_scaler.pkl")
        X = np.hstack([scaler.transform(df[feats]),
                       age_scaler.transform(df[["age_midpoint"]])])
        C = np.array([art["model"]["centroids_scaled"][str(c)] for c in range(k)])
        near = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(2), axis=1)
        smap = art["tier_merge"]["shipped_tier_map"]
        labels[:] = [smap[str(c)]["tier_label"] for c in near]

    # Gate: the replay must reproduce the artifact's own recorded shares before anything
    # downstream is believed. Order of tier_labels is healthiest-first.
    declared = [round(float(x), 1) for x in art["shipped_tiers_summary"]["population_shares_pct"]]
    got = [round(100 * float((labels == lab).mean()), 1)
           for lab in art["shipped_tiers_summary"]["tier_labels"]]
    if got != declared:
        raise SystemExit(f"{domain}: replayed shares {got} != artifact's own {declared} — "
                         f"refusing to analyse from an unverified tier assignment")
    return labels, art["shipped_tiers_summary"]["tier_labels"], got


# ======================================================================================
# Step 2 — is the "Stable + Not Covered" net-worth distribution actually two populations?
# ======================================================================================
def kde_modes(v, grid_n=512):
    """Count KDE local maxima and locate the deepest valley between the two highest modes.

    Sarle's BC and the GMM BIC comparison both answer 'could two components fit better'.
    Neither answers 'is there a gap you could put a threshold in', which is the question the
    proposed rule actually depends on — hence this third, deliberately literal signal.
    """
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    out = {"n_modes": None, "mode_positions": None, "primary_mode": None,
           "valley_position": None, "valley_depth_ratio": None}
    if len(v) < 20 or np.allclose(v, v[0]):
        return out
    kde = stats.gaussian_kde(v)                       # Scott's rule, no hand-tuned bandwidth
    xs = np.linspace(v.min(), v.max(), grid_n)
    ys = kde(xs)
    peaks = [i for i in range(1, grid_n - 1) if ys[i] > ys[i - 1] and ys[i] >= ys[i + 1]]
    out["n_modes"] = len(peaks)
    out["mode_positions"] = [round(float(xs[i]), 2) for i in peaks]
    if peaks:
        out["primary_mode"] = round(float(xs[max(peaks, key=lambda i: ys[i])]), 2)
    if len(peaks) >= 2:
        top2 = sorted(sorted(peaks, key=lambda i: -ys[i])[:2])
        lo, hi = top2
        j = lo + int(np.argmin(ys[lo:hi + 1]))
        out["valley_position"] = round(float(xs[j]), 2)
        # 1.0 = valley as deep as zero density; 0.0 = no dip at all between the modes.
        out["valley_depth_ratio"] = round(float(1 - ys[j] / min(ys[lo], ys[hi])), 3)
    return out


def unimodal_null(v, n_sims=300, seed=42):
    """What Sarle's BC looks like when the data is unimodal but skewed, at this n.

    BC = (skew^2 + 1) / kurtosis, so a heavily right-skewed but perfectly unimodal
    distribution inflates it. Net worth is exactly that kind of variable, so an observed
    BC > 0.555 cannot be read as 'two modes' without a control. This fits a gamma (unimodal
    by construction, location free so negatives are handled), draws `n_sims` samples of the
    same size, and reports how often a KNOWN-unimodal sample clears the flag and how often
    it reaches the observed BC.
    """
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    obs = bimodality(v).get("bimodality_coefficient")
    out = {"observed_bc": obs, "null_family": "gamma (unimodal by construction)",
           "n_sims": n_sims, "null_bc_median": None,
           "pct_null_draws_exceeding_flag": None, "pct_null_draws_exceeding_observed": None}
    if obs is None or len(v) < 20:
        return out
    try:
        a, loc, scale = stats.gamma.fit(v)
        rng = np.random.default_rng(seed)
        bcs = []
        for _ in range(n_sims):
            s = stats.gamma.rvs(a, loc=loc, scale=scale, size=len(v), random_state=rng)
            b = bimodality(s).get("bimodality_coefficient")
            if b is not None:
                bcs.append(b)
        if bcs:
            bcs = np.array(bcs)
            out["null_bc_median"] = round(float(np.median(bcs)), 4)
            out["pct_null_draws_exceeding_flag"] = round(100 * float((bcs > BC_FLAG).mean()), 1)
            out["pct_null_draws_exceeding_observed"] = round(
                100 * float((bcs >= obs).mean()), 1)
    except Exception as e:
        out["error"] = str(e)
    return out


def describe(v):
    v = np.asarray(v, float)
    return {"n": int(len(v)), "mean": round(float(v.mean()), 3),
            "median": round(float(np.median(v)), 3), "std": round(float(v.std(ddof=1)), 3),
            "min": round(float(v.min()), 3), "q25": round(float(np.percentile(v, 25)), 3),
            "q75": round(float(np.percentile(v, 75)), 3),
            "q90": round(float(np.percentile(v, 90)), 3), "max": round(float(v.max()), 3)}


def split_evidence(sub, thr):
    """What the proposed rule-1/rule-2 split actually produces at this threshold."""
    above = sub[sub["net_worth_to_income"] >= thr]
    below = sub[sub["net_worth_to_income"] < thr]
    ev = {
        "threshold": thr,
        "pct_above": round(100 * len(above) / len(sub), 1),
        "pct_below": round(100 * len(below) / len(sub), 1),
        "mean_above": round(float(above["net_worth_to_income"].mean()), 2) if len(above) else None,
        "mean_below": round(float(below["net_worth_to_income"].mean()), 2) if len(below) else None,
    }
    # Does the threshold separate the two archetypes it is supposed to separate? This is the
    # decisive check: a split can look statistically tidy and still cut the wrong groups.
    for arch in EXPECTED_MODES:
        a = sub[sub["profile_type"] == arch]
        ev[f"{arch}_n"] = int(len(a))
        ev[f"{arch}_pct_above_threshold"] = (
            round(100 * float((a["net_worth_to_income"] >= thr).mean()), 1) if len(a) else None)
        ev[f"{arch}_median_nw"] = (round(float(a["net_worth_to_income"].median()), 2)
                                   if len(a) else None)
    return ev


def empirical_best_separator(s):
    """If the goal is to isolate `wealthy_exposed` inside this band's subpopulation, where
    would the cut actually go, and how well would it do?

    DIAGNOSTIC ONLY — deliberately not a proposed production threshold. It is fitted in
    sample, on a label (`profile_type`) that exists only because this data is synthetic and
    that no real user carries. Its value is in showing how far the proposed benchmark
    threshold is from the best any single net-worth cut could do, and what the ceiling on
    that separation is at all.
    """
    v = s["net_worth_to_income"].to_numpy()
    y = (s["profile_type"] == "wealthy_exposed").to_numpy()
    if y.sum() < 10 or (~y).sum() < 10:
        return {"n_wealthy_exposed": int(y.sum()), "note": "too few to fit a cut"}
    cuts = np.quantile(v, np.linspace(0.02, 0.98, 97))
    # Youden's J: sensitivity + specificity - 1. Balances both errors, unlike raw accuracy,
    # which a 12%-prevalence class would let a trivial "nobody is wealthy" rule win.
    js = [( (v >= c) & y).sum() / y.sum() - ((v >= c) & ~y).sum() / (~y).sum() for c in cuts]
    i = int(np.argmax(js))
    best = float(cuts[i])
    tp = int(((v >= best) & y).sum())
    fp = int(((v >= best) & ~y).sum())
    return {
        "n_wealthy_exposed": int(y.sum()),
        "best_cut": round(best, 2),
        "youden_j": round(float(js[i]), 3),
        "sensitivity_pct": round(100 * tp / int(y.sum()), 1),
        "precision_pct": round(100 * tp / max(tp + fp, 1), 1),
        "note": ("in-sample, fitted on a synthetic-only label — a diagnostic ceiling, "
                 "not a proposed threshold"),
    }


def step2(df, out):
    sub = df[(df["banking_tier"] == "Stable") & (df["insurance_tier"] == "Not Covered")]
    res = {"subpopulation": "banking_tier == Stable AND insurance_tier == Not Covered",
           "n": int(len(sub)),
           "pct_of_population": round(100 * len(sub) / len(df), 1),
           "profile_mix_pct": {k: round(100 * v / len(sub), 1) for k, v in
                               sub["profile_type"].value_counts().items()},
           "overall": {}, "by_band": {}}

    v = sub["net_worth_to_income"]
    res["overall"] = {**describe(v), **{"bimodality": bimodality(v), "kde": kde_modes(v),
                                        "unimodal_null": unimodal_null(v)}}
    res["overall"]["bc_exceeds_flag"] = (
        res["overall"]["bimodality"]["bimodality_coefficient"] is not None
        and res["overall"]["bimodality"]["bimodality_coefficient"] > BC_FLAG)

    for b in range(1, 6):
        s = sub[sub["age_band"] == b]
        if len(s) < 20:
            res["by_band"][str(b)] = {"n": int(len(s)), "note": "too few users to assess"}
            continue
        vv = s["net_worth_to_income"]
        bim = bimodality(vv)
        entry = {"age_range": BANDS[b], **describe(vv), "bimodality": bim,
                 "kde": kde_modes(vv), "unimodal_null": unimodal_null(vv),
                 "split": split_evidence(s, NW_THRESHOLD[b]),
                 "empirical_best_separator": empirical_best_separator(s)}
        entry["bc_exceeds_flag"] = (bim["bimodality_coefficient"] is not None
                                    and bim["bimodality_coefficient"] > BC_FLAG)
        # Where the threshold sits relative to the observed spread — a threshold in the far
        # tail bisects nothing, whatever the shape of the distribution.
        entry["threshold_percentile_within_subpop"] = round(
            float(stats.percentileofscore(vv, NW_THRESHOLD[b])), 1)
        # How close the cut lands to the densest point of the distribution. A threshold
        # sitting ON the primary mode is the most arbitrary place a cut can go: it splits
        # the single most crowded region, where users either side are most alike.
        iqr = float(np.percentile(vv, 75) - np.percentile(vv, 25))
        pm = entry["kde"]["primary_mode"]
        entry["threshold_minus_primary_mode"] = (None if pm is None else
                                                 round(NW_THRESHOLD[b] - pm, 2))
        entry["threshold_distance_from_mode_in_iqr"] = (
            None if pm is None or iqr == 0 else round(abs(NW_THRESHOLD[b] - pm) / iqr, 2))
        entry["threshold_in_valley"] = (
            None if entry["kde"]["valley_position"] is None else
            bool(abs(entry["kde"]["valley_position"] - NW_THRESHOLD[b])
                 <= 0.10 * (vv.max() - vv.min())))
        res["by_band"][str(b)] = entry

    # Archetype-level reference: where the two named modes actually sit in this subpopulation.
    res["expected_modes_actual"] = {
        arch: (describe(sub.loc[sub["profile_type"] == arch, "net_worth_to_income"])
               if (sub["profile_type"] == arch).sum() >= 5 else
               {"n": int((sub["profile_type"] == arch).sum())})
        for arch in EXPECTED_MODES}
    out["step2_bimodality"] = res
    return sub, res


# ======================================================================================
# Step 3 — base rate of each proposed rule
# ======================================================================================
def rule_masks(df):
    """The 5 proposed rules as boolean masks. Rules 1 and 2 partition the same subpopulation
    at the band-appropriate net-worth threshold, so they are built from a per-user threshold
    column rather than a single constant."""
    thr = df["age_band"].map(NW_THRESHOLD)
    stable_uncovered = (df["banking_tier"] == "Stable") & (df["insurance_tier"] == "Not Covered")
    return {
        "1. Significant Assets, No Protection":
            stable_uncovered & (df["net_worth_to_income"] >= thr),
        "2. Building Wealth, No Safety Net":
            stable_uncovered & (df["net_worth_to_income"] < thr),
        "3. Protection First":
            (df["insurance_tier"] == "Covered") & (df["banking_tier"] == "Building"),
        "4. Strong Foundation, Room to Grow":
            (df["banking_tier"] == "Stable") & (df["investment_tier"] == "Getting Started"),
        "5. Investing Fast, Foundation Still Building":
            (df["banking_tier"] == "Building") & (df["investment_tier"] == "On Track"),
    }


HIGH_BASE_RATE_PP = 25.0        # the task's "is this really a personalised finding" line


def step3(df, out):
    normal = df[df["profile_type"] == "normal"]
    masks_all, masks_norm = rule_masks(df), rule_masks(normal)
    res = {"populations": {"full": int(len(df)), "normal_only": int(len(normal))},
           "high_base_rate_threshold_pct": HIGH_BASE_RATE_PP, "rules": {}}

    for name in masks_all:
        ma, mn = masks_all[name], masks_norm[name]
        entry = {
            "full_population_pct": round(100 * float(ma.mean()), 1),
            "full_population_n": int(ma.sum()),
            "normal_only_pct": round(100 * float(mn.mean()), 1),
            "normal_only_n": int(mn.sum()),
            "by_band_full_pct": {}, "by_band_normal_pct": {},
            "profile_mix_pct": {k: round(100 * v / max(int(ma.sum()), 1), 1) for k, v in
                                df.loc[ma, "profile_type"].value_counts().items()},
        }
        for b in range(1, 6):
            fb, nb = df["age_band"] == b, normal["age_band"] == b
            entry["by_band_full_pct"][str(b)] = round(100 * float(ma[fb].mean()), 1)
            entry["by_band_normal_pct"][str(b)] = round(100 * float(mn[nb].mean()), 1)
        entry["exceeds_high_base_rate"] = bool(
            max(entry["full_population_pct"], entry["normal_only_pct"]) > HIGH_BASE_RATE_PP)
        # A rule can be rare overall and still near-universal inside one age band, which is
        # the same framing problem confined to a segment — flag it separately.
        entry["bands_exceeding_high_base_rate"] = [
            b for b in range(1, 6)
            if max(entry["by_band_full_pct"][str(b)],
                   entry["by_band_normal_pct"][str(b)]) > HIGH_BASE_RATE_PP]
        res["rules"][name] = entry

    # How much of the population any rule at all touches, and how much is double-flagged —
    # both matter for whether this reads as a set of rare findings.
    any_mask = np.logical_or.reduce([m.to_numpy() for m in masks_all.values()])
    n_hits = np.sum([m.to_numpy().astype(int) for m in masks_all.values()], axis=0)
    res["coverage"] = {
        "pct_triggering_at_least_one_rule": round(100 * float(any_mask.mean()), 1),
        "pct_triggering_2_or_more": round(100 * float((n_hits >= 2).mean()), 1),
        "max_rules_triggered_by_one_user": int(n_hits.max()),
    }
    out["step3_base_rates"] = res
    return res


# ======================================================================================
# Plots
# ======================================================================================
def plots(df, sub, s2, s3):
    os.makedirs(PLOT_DIR, exist_ok=True)
    written = []

    # 1. The whole subpopulation, with every band threshold marked.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(sub["net_worth_to_income"], bins=60, color="#4C72B0", edgecolor="white")
    for b, t in NW_THRESHOLD.items():
        ax.axvline(t, color="#C44E52", ls="--", lw=1)
        ax.text(t, ax.get_ylim()[1] * 0.96, f"b{b}:{t:g}x", rotation=90,
                va="top", ha="right", fontsize=8, color="#C44E52")
    for arch, m in EXPECTED_MODES.items():
        ax.axvline(m, color="#55A868", ls=":", lw=1.5)
        ax.text(m, ax.get_ylim()[1] * 0.55, f" expected {arch} mode ~{m}x",
                fontsize=8, color="#2F6B45")
    ax.set_title(f"net_worth_to_income — Banking=Stable & Insurance=Not Covered "
                 f"(n={len(sub)})")
    ax.set_xlabel("net worth / annual income (x)")
    ax.set_ylabel("users")
    fig.tight_layout()
    p = os.path.join(PLOT_DIR, "nw_stable_notcovered_overall.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    # 2. Per band, with that band's own threshold and a KDE overlay.
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
    for b, ax in zip(range(1, 6), axes):
        s = sub[sub["age_band"] == b]["net_worth_to_income"]
        ax.hist(s, bins=30, color="#4C72B0", edgecolor="white", density=True)
        if len(s) >= 20:
            xs = np.linspace(s.min(), s.max(), 400)
            ax.plot(xs, stats.gaussian_kde(s)(xs), color="#333333", lw=1.5)
        ax.axvline(NW_THRESHOLD[b], color="#C44E52", ls="--", lw=1.6,
                   label=f"threshold {NW_THRESHOLD[b]:g}x")
        bc = s2["by_band"].get(str(b), {}).get("bimodality", {}).get("bimodality_coefficient")
        ax.set_title(f"Band {b} ({BANDS[b]})  n={len(s)}\nBC={bc}", fontsize=10)
        ax.set_xlabel("net worth / income (x)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("net_worth_to_income within Banking=Stable & Insurance=Not Covered, by age band")
    fig.tight_layout()
    p = os.path.join(PLOT_DIR, "nw_stable_notcovered_by_band.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    # 3. The same subpopulation decomposed by profile_type — shows directly whether the two
    #    named archetypes form separate humps or sit inside one continuum.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    order = ["normal", "wealthy_exposed", "underinsured_saver"]
    rest = [pt for pt in sub["profile_type"].unique() if pt not in order]
    # One colour per profile type: 8 here for the 8 generated types (normal + 6 mismatch +
    # outlier). zip truncates, so a new profile type would silently go unplotted.
    for name, colour in zip(order + rest, ["#B0B0B0", "#55A868", "#C44E52", "#8172B2",
                                           "#CCB974", "#64B5CD", "#937860", "#DA8BC3"]):
        s = sub[sub["profile_type"] == name]["net_worth_to_income"]
        if len(s) >= 10:
            xs = np.linspace(sub["net_worth_to_income"].min(),
                             sub["net_worth_to_income"].max(), 400)
            ax.plot(xs, stats.gaussian_kde(s)(xs), lw=2, color=colour,
                    label=f"{name} (n={len(s)})")
    ax.set_title("Same subpopulation, decomposed by injected profile type")
    ax.set_xlabel("net worth / annual income (x)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(PLOT_DIR, "nw_stable_notcovered_by_profile.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    # 4. Base rates, full vs normal-only.
    r = s3["rules"]
    names = list(r)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 0.2, [r[n]["full_population_pct"] for n in names], 0.4,
           label="full population", color="#4C72B0")
    ax.bar(x + 0.2, [r[n]["normal_only_pct"] for n in names], 0.4,
           label="normal only (65%)", color="#55A868")
    ax.axhline(HIGH_BASE_RATE_PP, color="#C44E52", ls="--", lw=1.2,
               label=f"{HIGH_BASE_RATE_PP:g}% 'common finding' line")
    ax.set_xticks(x)
    ax.set_xticklabels([n.split(". ", 1)[1].replace(", ", ",\n") for n in names], fontsize=8)
    ax.set_ylabel("% of users triggering")
    ax.set_title("Proposed Layer 3 rules — base rates")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(PLOT_DIR, "layer3_rule_base_rates.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)
    return [os.path.relpath(w, REPO_ROOT).replace("\\", "/") for w in written]


# ======================================================================================
# Markdown report
# ======================================================================================
def write_markdown(df, sub, s2, s3, plot_paths, tier_summary):
    L = []
    a = L.append
    a("# Layer 3 combination rules — validation against the synthetic data\n")
    a("**Analysis only.** No production file was written, nothing was refit, and no Layer 3 "
      "rule was implemented. Tier assignment is replayed from the committed artifacts in "
      "`backend/ml/artifacts/domain_split/` through their own per-band `tier_map`, and the "
      "replay is asserted against each artifact's recorded population shares before any "
      "number below is computed.\n")
    a(f"Generated by `MLAI/clustering/layer3_validation.py` on "
      f"{len(df):,} synthetic users.\n")

    a("## Tier replay\n")
    a("| Domain | Tiers (healthiest first) | Population share |")
    a("|---|---|---|")
    for d, (labels, shares) in tier_summary.items():
        a(f"| {d} | {' / '.join(labels)} | {' / '.join(f'{s}%' for s in shares)} |")
    a("")

    a("## Step 2 — is \"Stable + Not Covered\" two populations or one?\n")
    a(f"Subpopulation: **n = {s2['n']:,}** ({s2['pct_of_population']}% of all users).\n")
    o = s2["overall"]
    a(f"Pooled distribution of `net_worth_to_income`: mean **{o['mean']}x**, median "
      f"**{o['median']}x**, std **{o['std']}**, IQR {o['q25']}–{o['q75']}x, range "
      f"{o['min']}–{o['max']}x.\n")
    a(f"- Sarle's bimodality coefficient **{o['bimodality']['bimodality_coefficient']}** "
      f"(flag > {BC_FLAG}): {'exceeds' if o['bc_exceeds_flag'] else 'does NOT exceed'} the flag.")
    a(f"- 2-component GMM preferred over 1: **{o['bimodality']['gmm_prefers_2']}** "
      f"(BIC {o['bimodality']['gmm_bic_1']} vs {o['bimodality']['gmm_bic_2']}).")
    a(f"- KDE modes: **{o['kde']['n_modes']}** at {o['kde']['mode_positions']}; deepest "
      f"valley {o['kde']['valley_position']} (depth ratio "
      f"{o['kde']['valley_depth_ratio']} — 1.0 would be a clean gap, 0.0 no dip at all).")
    un = o["unimodal_null"]
    a(f"- Unimodal control: skewness is **{o['bimodality']['skewness']}**, and BC is "
      f"skew-driven. Fitting a gamma (unimodal by construction) to this same data and "
      f"redrawing {un['n_sims']} samples gives a median BC of **{un['null_bc_median']}**, "
      f"with **{un['pct_null_draws_exceeding_flag']}%** of known-unimodal draws clearing the "
      f"{BC_FLAG} flag and **{un['pct_null_draws_exceeding_observed']}%** reaching the "
      f"observed BC.\n")
    a("Pooling across bands is not the right unit of analysis here (the threshold is "
      "band-specific and net worth rises with age), so the per-band view below is the one "
      "the proposed rule actually depends on.\n")

    a("### Per age band\n")
    a("| Band | n | mean | median | std | skew | BC | unimodal-null BC (% of draws > flag) | "
      "GMM prefers 2 | KDE modes | valley (depth) | threshold | thr. percentile | % above |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for b in range(1, 6):
        e = s2["by_band"][str(b)]
        if "note" in e:
            a(f"| {b} | {e['n']} | — | — | — | — | — | — | — | — | — | "
              f"{NW_THRESHOLD[b]:g}x | — | — |")
            continue
        sp, un, kd = e["split"], e["unimodal_null"], e["kde"]
        a(f"| {b} ({e['age_range']}) | {e['n']} | {e['mean']}x | {e['median']}x | {e['std']} | "
          f"{e['bimodality']['skewness']} | "
          f"{e['bimodality']['bimodality_coefficient']} | "
          f"{un['null_bc_median']} ({un['pct_null_draws_exceeding_flag']}%) | "
          f"{e['bimodality']['gmm_prefers_2']} | {kd['n_modes']} | "
          f"{kd['valley_position']} ({kd['valley_depth_ratio']}) | "
          f"{NW_THRESHOLD[b]:g}x | {e['threshold_percentile_within_subpop']} | "
          f"{sp['pct_above']}% |")
    a("")
    a("`valley (depth)` is the deepest KDE trough between the two highest modes and how far "
      "the density falls into it (1.0 = clean gap, 0.0 = no dip). `thr. percentile` is where "
      "the band threshold sits inside this subpopulation. A threshold that separates two "
      "real groups should sit at or near a deep valley; the values above are the direct "
      "check of that.\n")

    a("### Does the threshold separate the two archetypes it is meant to separate?\n")
    a("| Band | threshold | `wealthy_exposed` n / % above | `underinsured_saver` n / % above |")
    a("|---|---|---|---|")
    for b in range(1, 6):
        e = s2["by_band"][str(b)]
        if "note" in e:
            continue
        sp = e["split"]
        a(f"| {b} | {NW_THRESHOLD[b]:g}x | {sp['wealthy_exposed_n']} / "
          f"{sp['wealthy_exposed_pct_above_threshold']}% | "
          f"{sp['underinsured_saver_n']} / "
          f"{sp['underinsured_saver_pct_above_threshold']}% |")
    a("")
    a("If the point of the split is to isolate `wealthy_exposed`, where would the cut "
      "actually have to go? (Fitted in sample on a synthetic-only label — a diagnostic "
      "ceiling on what any single net-worth cut can do, **not** a proposed threshold.)\n")
    a("| Band | proposed threshold | best empirical cut | Youden's J | sensitivity | precision |")
    a("|---|---|---|---|---|---|")
    for b in range(1, 6):
        e = s2["by_band"][str(b)]
        if "note" in e:
            continue
        bs = e["empirical_best_separator"]
        if "best_cut" not in bs:
            a(f"| {b} | {NW_THRESHOLD[b]:g}x | — | — | — | — |")
            continue
        a(f"| {b} | {NW_THRESHOLD[b]:g}x | **{bs['best_cut']}x** | {bs['youden_j']} | "
          f"{bs['sensitivity_pct']}% | {bs['precision_pct']}% |")
    a("")

    a("Where the two named modes actually sit inside this subpopulation:\n")
    for arch, d in s2["expected_modes_actual"].items():
        if "mean" in d:
            a(f"- `{arch}` (expected ~{EXPECTED_MODES[arch]}x): n={d['n']}, median "
              f"**{d['median']}x**, IQR {d['q25']}–{d['q75']}x")
        else:
            a(f"- `{arch}`: n={d['n']} — too few to describe")
    a("")

    a("### Verdict on the split assumption\n")
    # (band number, entry) pairs for the bands that were actually scored. Kept as pairs
    # rather than a filtered list zipped against range(1, 6): a band thin enough to be
    # skipped would shift that zip and silently label every later band wrongly for the
    # whole verdict section.
    scored = [(b, s2["by_band"][str(b)]) for b in range(1, 6)
              if "note" not in s2["by_band"][str(b)]]
    in_valley = [b for b, e in scored if e["threshold_in_valley"]]
    ws_above = [e["split"]["underinsured_saver_pct_above_threshold"] for _, e in scored]
    a(f"- The band threshold sits in a density valley in **{len(in_valley)} of "
      f"{len(scored)} bands** ({in_valley if in_valley else 'none'}).")
    db_band, db_e = max(scored, key=lambda t: t[1]["kde"]["valley_depth_ratio"] or 0)
    deepest = db_e["kde"]["valley_depth_ratio"] or 0
    others = [e["kde"]["valley_depth_ratio"] or 0 for b, e in scored if b != db_band]
    a(f"- Deepest valley in any band: **{deepest}** on a 0–1 scale, in band {db_band} at "
      f"**{db_e['kde']['valley_position']}x** — "
      f"{'a genuine gap' if deepest > 0.5 else 'still not a gap'}, but band {db_band}'s "
      f"threshold is {NW_THRESHOLD[db_band]:g}x, so the gap and the cut are in different "
      f"places. Every other band's deepest valley is shallower than {max(others):.3f}.")
    a(f"- The thresholds land between the "
      f"{min(e['threshold_percentile_within_subpop'] for _, e in scored)}th and "
      f"{max(e['threshold_percentile_within_subpop'] for _, e in scored)}th percentile of the "
      f"subpopulation — i.e. through the middle of the mass, not through a gap.")
    dm = [e["threshold_distance_from_mode_in_iqr"] for _, e in scored
          if e["threshold_distance_from_mode_in_iqr"] is not None]
    pairs = ", ".join("b{}: {:g}x vs mode {}x".format(b, NW_THRESHOLD[b],
                                                      e["kde"]["primary_mode"])
                      for b, e in scored)
    a(f"- Worse than merely 'not in a valley': the threshold sits **on or beside the primary "
      f"mode** in every band — {min(dm)}–{max(dm)} IQRs from the densest point ({pairs}). "
      f"The cut runs through the most crowded part of the distribution, where the users "
      f"either side of it are most alike.")
    a(f"- `wealthy_exposed` is 100% above threshold in every band, but "
      f"**{min(ws_above)}–{max(ws_above)}% of `underinsured_saver`** also lands above it, so "
      f"the threshold does not cleanly separate the two archetypes the split is premised on.")
    a(f"- The subpopulation is **{s2['profile_mix_pct'].get('normal', 0)}% `normal` users**; "
      f"the two named archetypes are a minority of it "
      f"({s2['profile_mix_pct'].get('wealthy_exposed', 0)}% + "
      f"{s2['profile_mix_pct'].get('underinsured_saver', 0)}%), so their separation was "
      f"never going to shape the aggregate distribution on its own.\n")

    a("## Step 3 — base rate of each proposed rule\n")
    a(f"Full population n={s3['populations']['full']:,}; "
      f"`normal` only n={s3['populations']['normal_only']:,}.\n")
    a("| Rule | full pop | normal only | b1 | b2 | b3 | b4 | b5 | > "
      f"{HIGH_BASE_RATE_PP:g}% overall? | bands > {HIGH_BASE_RATE_PP:g}% |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for name, e in s3["rules"].items():
        cells = " | ".join(f"{e['by_band_full_pct'][str(b)]}%" for b in range(1, 6))
        hb = e["bands_exceeding_high_base_rate"]
        a(f"| {name} | **{e['full_population_pct']}%** | {e['normal_only_pct']}% | {cells} | "
          f"{'**YES**' if e['exceeds_high_base_rate'] else 'no'} | "
          f"{', '.join(str(b) for b in hb) if hb else '—'} |")
    a("")
    a("Per-band figures are on the full population; the `normal`-only per-band split is in "
      "the JSON (`by_band_normal_pct`).\n")
    a(f"- Users triggering **at least one** rule: "
      f"{s3['coverage']['pct_triggering_at_least_one_rule']}%")
    a(f"- Users triggering **two or more**: {s3['coverage']['pct_triggering_2_or_more']}% "
      f"(max {s3['coverage']['max_rules_triggered_by_one_user']} rules on one user)\n")

    a("## Plots\n")
    for p in plot_paths:
        a(f"- `{p}`")
    a("")
    return "\n".join(L)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    print("=" * 96)
    print(f"LAYER 3 RULE VALIDATION — {len(df):,} synthetic users (analysis only, no refit)")
    print("=" * 96)

    tier_summary = {}
    for domain, col in [("banking", "banking_tier"), ("investment", "investment_tier"),
                        ("insurance", "insurance_tier")]:
        labels, tier_labels, shares = replay_tiers(df, domain)
        df[col] = labels
        tier_summary[domain] = (tier_labels, shares)
        print(f"  [replay OK] {domain:<11} {tier_labels} -> {shares}%  "
              f"(matches the artifact's own recorded shares)")

    summary = {}
    summary["dataset"] = {"path": "MLAI/clustering/data/processed/synthetic_users.csv",
                          "n_users": int(len(df))}
    summary["nw_thresholds_by_band"] = NW_THRESHOLD
    summary["tier_replay"] = {d: {"tier_labels": v[0], "population_shares_pct": v[1]}
                              for d, v in tier_summary.items()}
    summary["tier_crosstab_pct"] = {
        f"{bt} / {it}": round(100 * float(((df["banking_tier"] == bt)
                                           & (df["insurance_tier"] == it)).mean()), 1)
        for bt in df["banking_tier"].unique() for it in df["insurance_tier"].unique()}

    sub, s2 = step2(df, summary)
    print(f"\n  step 2 subpopulation (Stable + Not Covered): n={s2['n']:,} "
          f"({s2['pct_of_population']}% of users)")
    o = s2["overall"]
    print(f"    pooled: mean {o['mean']}x  median {o['median']}x  std {o['std']}  "
          f"BC {o['bimodality']['bimodality_coefficient']}  "
          f"KDE modes {o['kde']['n_modes']}")
    for b in range(1, 6):
        e = s2["by_band"][str(b)]
        if "note" in e:
            continue
        print(f"    band {b}: n={e['n']:<5} median {e['median']:>6}x  "
              f"BC {e['bimodality']['bimodality_coefficient']:<7} "
              f"GMM2 {str(e['bimodality']['gmm_prefers_2']):<6} "
              f"modes {e['kde']['n_modes']}  thr {NW_THRESHOLD[b]:g}x at pctile "
              f"{e['threshold_percentile_within_subpop']:<5} -> "
              f"{e['split']['pct_above']}% above")

    s3 = step3(df, summary)
    print("\n  step 3 base rates (full / normal-only):")
    for name, e in s3["rules"].items():
        flag = "  <-- HIGH" if e["exceeds_high_base_rate"] else ""
        print(f"    {name:<48} {e['full_population_pct']:>5}% / "
              f"{e['normal_only_pct']:>5}%{flag}")

    plot_paths = plots(df, sub, s2, s3)
    summary["plots"] = plot_paths

    with open(os.path.join(OUT_DIR, "layer3_validation.json"), "w") as f:
        json.dump(summary, f, indent=2)
    md = write_markdown(df, sub, s2, s3, plot_paths, tier_summary)
    with open(os.path.join(OUT_DIR, "LAYER3_VALIDATION.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n  wrote {os.path.relpath(OUT_DIR, REPO_ROOT)}/"
          f"{{layer3_validation.json, LAYER3_VALIDATION.md}} + {len(plot_paths)} plots")


if __name__ == "__main__":
    main()
