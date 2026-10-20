"""
plot_report_figures.py — two report figures built purely from already-computed results.

Reads (never writes) two current-data result files and draws:
    ablation_silhouette_by_k.png   Config A (11 continuous) vs Config B (14, incl. 3
                                   binaries) silhouette across k=2..6, one panel per
                                   approach — the "dilution, not distortion" finding.
    step5_dual_bar_scatter.png     every size-floor survivor of the 12-algorithm
                                   benchmark at (worst-group min Spearman rho, min lift
                                   over base rate) — Step 5's "the two bars never
                                   co-occur" finding.

No clustering is re-run and synthetic_users.csv is never opened: every plotted number is
read straight out of the result files below. Their sha256 and the dataset's are asserted
unchanged across the run.

Inputs:
    results/ablation_rerun_2026-09-27.json          (ablation on the current dataset)
    results/rerun_2026-09-27/final_selection.json   (12-algorithm selection, current data)
Output:
    results/report_figures_2026-09-27/   (refuses to overwrite an existing file)

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/plot_report_figures.py
"""

import collections
import hashlib
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend; must be selected before pyplot is imported
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "MLAI/clustering/results")
ABLATION_JSON = os.path.join(RESULTS_DIR, "ablation_rerun_2026-09-27.json")
SELECTION_JSON = os.path.join(RESULTS_DIR, "rerun_2026-09-27", "final_selection.json")
OUT_DIR = os.path.join(RESULTS_DIR, "report_figures_2026-09-27")
DATA_PATH = os.path.join(REPO_ROOT, "MLAI/clustering/data/processed/synthetic_users.csv")
DATASET_SHA256 = "843758a99de5d98de485f5cfb9f0a81b4ea0a8741c2266e0eb5ea5883b4ae6d5"

K_RANGE = [2, 3, 4, 5, 6]
# Step 5 gates, as final_selection.py defines them (MONOTONIC_RHO_MIN, MIN_LIFT_PP).
RHO_MIN = 0.5
LIFT_MIN_PP = 10.0

# Palette — validated categorical slots 1-2 plus neutral tokens (dataviz reference
# palette, light mode). Blue/orange pass every check all-pairs (CVD dE 24.7).
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.linewidth": 1,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _style(ax):
    """Recessive frame: hairline solid y-grid, no top/right spines."""
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    if os.path.exists(path):
        sys.exit(f"ABORT: {path} already exists — refusing to overwrite")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# --------------------------------------------------------------------------------------
# Figure 1 — ablation: silhouette by k, Config A vs Config B, one panel per approach
# --------------------------------------------------------------------------------------
def plot_ablation(ablation):
    configs = ablation["configs"]
    series = {}
    for approach in ("approach_A", "approach_B"):
        series[approach] = {
            cfg: [configs[cfg][approach][str(k)]["silhouette_score"] for k in K_RANGE]
            for cfg in ("A_continuous_11", "B_mixed_14")
        }

    # The finding the figure must carry, checked rather than assumed.
    for approach, s in series.items():
        assert all(b < a for a, b in zip(s["A_continuous_11"], s["B_mixed_14"])), approach

    # Shared y so the two panels are directly comparable (small multiples, one scale).
    lo = min(min(v) for s in series.values() for v in s.values())
    hi = max(max(v) for s in series.values() for v in s.values())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    titles = {"approach_A": "Approach A — one model per age band",
              "approach_B": "Approach B — single model, age as a feature"}
    for ax, approach in zip(axes, ("approach_A", "approach_B")):
        a = series[approach]["A_continuous_11"]
        b = series[approach]["B_mixed_14"]
        _style(ax)
        ax.fill_between(K_RANGE, b, a, color=ORANGE, alpha=0.10, linewidth=0)
        ax.plot(K_RANGE, a, color=BLUE, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.plot(K_RANGE, b, color=ORANGE, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_title(titles[approach], loc="left", fontsize=11, color=TEXT)
        ax.set_xticks(K_RANGE)
        ax.set_xlabel("k (number of clusters)")
        ax.set_xlim(1.8, 6.2)
    axes[0].set_ylabel("Silhouette score (higher is better)")
    axes[0].set_ylim(lo - 0.01, hi + 0.015)
    # Report-figure style: legend and a title below the plot; the explanation (per-panel
    # gap percentages, data provenance) belongs in the caption, not on the image.
    handles = [plt.Line2D([], [], color=BLUE, linewidth=2, marker="o",
                          label="Config A — 11 continuous features"),
               plt.Line2D([], [], color=ORANGE, linewidth=2, marker="o",
                          label="Config B — 14 features (adds 3 binary flags)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9.5,
               labelcolor=TEXT_2, bbox_to_anchor=(0.5, 0.075))
    fig.text(0.5, 0.012, "Silhouette score by k, Config A vs Config B, under both "
             "approaches", ha="center", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    _save(fig, "ablation_silhouette_by_k.png")
    return series


# --------------------------------------------------------------------------------------
# Figure 2 — Step 5 dual-bar scatter over all size-floor survivors
# --------------------------------------------------------------------------------------
def plot_step5(selection):
    sweep = selection["step5_survivor_usability_sweep"]
    survivors = selection["step1_summary"]["survivors"]
    points = {}
    for key in survivors:
        s = sweep[key]
        rho, lift = s["worst_group_min_spearman_rho"], s["min_lift_pp"]
        if rho is None or lift is None or len(s["lift"]) != 5:
            sys.exit(f"ABORT: {key} lacks a worst-group rho or all 5 lifts — not plottable "
                     "without estimating")
        orders, means = rho >= RHO_MIN, lift >= LIFT_MIN_PP
        # Must agree with the file's own gate verdicts, or the extraction is wrong.
        assert orders == s["passes_monotonicity_gate"], key
        assert means == s["passes_materiality_gate"], key
        region = ("both" if orders and means else "orders" if orders
                  else "means" if means else "neither")
        points[key] = (rho, lift, region)
    counts = collections.Counter(r for *_, r in points.values())
    assert counts["orders"] == len(selection["gate2a_ordering_passing"])
    assert counts["means"] == len(selection["gate2b_materiality_passing"])
    assert counts["both"] == len(selection["gate2_passing_combinations"])

    fig, ax = plt.subplots(figsize=(10, 6.6))
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=1)
    xmin, xmax, ymin, ymax = -1.12, 1.12, -0.6, 15.5
    # The "both bars pass" region — a faint unlabelled wash; its emptiness is the finding.
    ax.fill_between([RHO_MIN, xmax], LIFT_MIN_PP, ymax, color=BLUE, alpha=0.06, linewidth=0)
    ax.axvline(RHO_MIN, color=TEXT_2, linewidth=1)
    ax.axhline(LIFT_MIN_PP, color=TEXT_2, linewidth=1)

    style = {  # colour + shape (secondary encoding), neither-group recessive
        "neither": dict(color=MUTED, marker="o", s=46, alpha=0.75, zorder=2,
                        label=f"Passes neither bar ({counts['neither']})"),
        "orders": dict(color=BLUE, marker="s", s=70, zorder=3,
                       label=f"Orders sensibly only — ρ ≥ 0.5 ({counts['orders']})"),
        "means": dict(color=ORANGE, marker="D", s=62, zorder=3,
                      label=f"Means something only — lift ≥ 10 pp ({counts['means']})"),
        "both": dict(color=TEXT, marker="*", s=140, zorder=4,
                     label=f"Passes both ({counts['both']})"),
    }
    for region in ("neither", "orders", "means", "both"):
        xs = [p[0] for p in points.values() if p[2] == region]
        ys = [p[1] for p in points.values() if p[2] == region]
        ax.scatter(xs, ys, edgecolors=SURFACE, linewidths=1.5, **style[region])

    # Exact duplicate coordinates are never jittered apart; they are reported with the
    # figure (printed by main()) for the caption rather than annotated on the image.
    dup = collections.Counter((p[0], p[1]) for p in points.values())
    stacked = sorted(((x, y), n) for (x, y), n in dup.items() if n > 1)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Orders sensibly →  worst-group minimum Spearman ρ across the 5 "
                  "external benchmarks")
    ax.set_ylabel("Means something →  minimum lift over base rate (pp)")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=TEXT_2)
    fig.text(0.5, 0.012, "Step 5 dual-bar check: all 41 size-floor survivors, ordering vs "
             "materiality", ha="center", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "step5_dual_bar_scatter.png")
    return counts, stacked, points


def main():
    for path in (ABLATION_JSON, SELECTION_JSON, DATA_PATH):
        if not os.path.isfile(path):
            sys.exit(f"ABORT: missing {path}")
    if sha256(DATA_PATH) != DATASET_SHA256:
        sys.exit("ABORT: synthetic_users.csv is not the expected current dataset")
    before = {p: sha256(p) for p in (ABLATION_JSON, SELECTION_JSON, DATA_PATH)}
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(ABLATION_JSON) as f:
        ablation = json.load(f)
    with open(SELECTION_JSON) as f:
        selection = json.load(f)

    print("Report figures from existing current-data results (nothing re-run)...")
    series = plot_ablation(ablation)
    counts, stacked, points = plot_step5(selection)

    after = {p: sha256(p) for p in before}
    for p in before:
        state = "UNCHANGED" if before[p] == after[p] else "CHANGED !!"
        print(f"  [guard] {state:<10} {os.path.relpath(p, REPO_ROOT)}")
    if before != after:
        sys.exit("ABORT: an input file changed during the run")
    print(f"\nStep 5 regions: neither {counts['neither']}, orders-only {counts['orders']}, "
          f"means-only {counts['means']}, both {counts['both']} "
          f"(total {sum(counts.values())})")
    # Caption material — the detail deliberately kept off the images.
    print("Stacked (exact duplicate) points: " + "; ".join(
        f"x{n} at rho={x:g}, lift={y:g}pp" for (x, y), n in stacked))
    for region in ("orders", "means"):
        print(f"{region}-only: " + "; ".join(
            f"{k} (rho={p[0]:g}, lift={p[1]:g})" for k, p in sorted(points.items())
            if p[2] == region))
    for approach, s in series.items():
        a, b = s["A_continuous_11"], s["B_mixed_14"]
        gaps = [round(x - y, 4) for x, y in zip(a, b)]
        rel = [round(100 * (x - y) / x, 1) for x, y in zip(a, b)]
        print(f"Ablation {approach}: A - B gap by k {gaps}; relative % by k {rel} "
              f"(range {min(rel)}-{max(rel)}%)")


if __name__ == "__main__":
    main()
