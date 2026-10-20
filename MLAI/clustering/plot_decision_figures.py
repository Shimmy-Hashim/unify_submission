"""
plot_decision_figures.py — one plain decision chart per ML stage, built purely from
already-computed, already-verified result files. Sibling of plot_report_figures.py (same
style, same palette, same refuse-to-overwrite and input-hash guards).

No clustering is re-run. Every plotted number is read out of a saved result file; the one
exception is figure 11, which aggregates a SAVED label cache against the dataset's
profile_type column (a read-only count — nothing is fitted). Every figure the brief quotes
is asserted against the file before that figure is drawn; a mismatch skips the figure and
is reported, never adjusted.

Figures (results/decision_figures_2026-09-28/):
    01  binary variance share, actual vs proportional          (ablation)
    02  5% size floor by algorithm, 90-combination grid        (Thread 1, Step 1)
    03  the 41 survivors by macro-silhouette, top 15           (Thread 1, Step 3)
    04  leader A|birch|2: healthier cluster vs base rate       (Thread 1, Step 4A)
    05  leader A|birch|2: share of separation by feature       (Thread 1, Step 4B)
    06  gate-passers with vs without net_worth_to_income       (domain split)
    07  Banking gate-passers ranked                            (domain split)
    08  Insurance gate-passers ranked                          (domain split)
    09  Investment gate-passers ranked (V2 benchmark set)      (domain split)
    10  Banking: why 2 tiers, not 3                            (tier count)
    11  Investment: who fills the third tier                   (tier count)
    12  Insurance: are tiers 1 and 2 different?                (tier count)
    13  archetype placement under the domain split             (Step 4)

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/plot_decision_figures.py [--only 11,12]
(--only builds just the listed figures; every figure still refuses to overwrite.)
"""

import collections
import csv
import hashlib
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend; must be selected before pyplot is imported
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "MLAI/clustering/results")
DS_DIR = os.path.join(RESULTS_DIR, "domain_split")
ABLATION_JSON = os.path.join(RESULTS_DIR, "ablation_rerun_2026-09-27.json")
SELECTION_JSON = os.path.join(RESULTS_DIR, "rerun_2026-09-27", "final_selection.json")
SELECTION_GRID_CSV = os.path.join(RESULTS_DIR, "rerun_2026-09-27", "final_selection_grid.csv")
DS_RESULTS_JSON = os.path.join(DS_DIR, "domain_split_results.json")
DS_GRID_CSV = os.path.join(DS_DIR, "domain_split_grid.csv")
RISK_AGE_FINAL_JSON = os.path.join(DS_DIR, "risk_age_final.json")
DISTINCT_JSON = os.path.join(DS_DIR, "adjacent_tier_distinctness.json")
RISK_AGE_CACHE = os.path.join(DS_DIR, ".risk_age_diagnostic_grids.pkl")
DATA_PATH = os.path.join(REPO_ROOT, "MLAI/clustering/data/processed/synthetic_users.csv")
DATASET_SHA256 = "843758a99de5d98de485f5cfb9f0a81b4ea0a8741c2266e0eb5ea5883b4ae6d5"
OUT_DIR = os.path.join(RESULTS_DIR, "decision_figures_2026-09-28")

DS_ASSIGN_CSV = os.path.join(DS_DIR, "domain_split_tier_assignments.csv")
L3_AUDIT_JSON = os.path.join(RESULTS_DIR, "layer3_coverage_audit", "layer3_coverage_audit.json")
INPUTS = (ABLATION_JSON, SELECTION_JSON, SELECTION_GRID_CSV, DS_RESULTS_JSON, DS_GRID_CSV,
          RISK_AGE_FINAL_JSON, DISTINCT_JSON, RISK_AGE_CACHE, DS_ASSIGN_CSV, L3_AUDIT_JSON,
          DATA_PATH)

# Palette — plot_report_figures.py's tokens, plus one darker step per approach hue for the
# winner/leader (blue step 600 from the same reference ramp; a darker step of the orange).
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
BLUE_DARK = "#184f95"
BLUE_LIGHT = "#86b6ef"
ORANGE = "#eb6834"
ORANGE_DARK = "#a8401a"
ORANGE_LIGHT = "#f4a98a"
AQUA = "#1baf7a"
NEUTRAL_DARK = TEXT_2
NEUTRAL_LIGHT = BASELINE

APPROACH_COLOUR = {"A": BLUE, "B": ORANGE}
APPROACH_WINNER = {"A": BLUE_DARK, "B": ORANGE_DARK}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.linewidth": 1,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

REPORT = []      # (figure, line) — printed at the end for the task report
SKIPPED = {}     # figure -> reason


class Mismatch(Exception):
    pass


def check(fig_id, label, got, expected, tol=0.0):
    ok = (abs(got - expected) <= tol) if isinstance(expected, (int, float)) and \
        not isinstance(expected, bool) else got == expected
    REPORT.append((fig_id, f"check {label}: file={got!r} expected={expected!r} "
                           f"-> {'MATCH' if ok else 'MISMATCH'}"))
    if not ok:
        raise Mismatch(f"{label}: file says {got!r}, expected {expected!r}")


def note(fig_id, text):
    REPORT.append((fig_id, text))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _style(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def _finish(fig, title, legend_handles=None, ncol=2, bottom=0.14):
    """Report-figure layout: legend, then the title, both below the plot."""
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center", ncol=ncol, frameon=False,
                   fontsize=9.5, labelcolor=TEXT_2, bbox_to_anchor=(0.5, 0.06))
    fig.text(0.5, 0.012, title, ha="center", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, bottom, 1, 1))


def _save(fig, fig_id, name):
    path = os.path.join(OUT_DIR, name)
    if os.path.exists(path):
        sys.exit(f"ABORT: {path} already exists — refusing to overwrite")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    note(fig_id, f"saved {os.path.relpath(path, REPO_ROOT)}")


def _patch(colour, label):
    return plt.Rectangle((0, 0), 1, 1, color=colour, label=label)


def _hbar_ranking(ax, keys, values, colours):
    """Horizontal bars, first key at the TOP. 2px surface gap between bars."""
    ys = list(range(len(keys)))[::-1]
    ax.barh(ys, values, color=colours, height=0.72, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(ys)
    ax.set_yticklabels(keys)


# --------------------------------------------------------------------------------------
# 01 — binary variance share
# --------------------------------------------------------------------------------------
def fig01(ablation):
    f = "01"
    sd = ablation["scaler_degeneracy_check"]
    actual, prop = sd["binary_combined_pct_of_scaled_variance"], sd["proportionate_share_pct"]
    note(f, "source: ablation_rerun_2026-09-27.json -> scaler_degeneracy_check."
            "binary_combined_pct_of_scaled_variance / proportionate_share_pct")
    check(f, "binary share actual %", actual, 6.49)
    check(f, "proportional share %", prop, 21.43)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    _style(ax)
    _hbar_ranking(ax, ["Proportional share (3 of 14 features)", "Actual share, 3 binary flags"],
                  [prop, actual], [NEUTRAL_LIGHT, NEUTRAL_DARK])
    ax.set_xlabel("Share of total scaled variance (%)")
    ax.set_xlim(0, 25)
    _finish(fig, "Binary flags' share of scaled variance under RobustScaler: actual vs "
                 "proportional", bottom=0.1)
    note(f, "bars: 2")
    _save(fig, f, "01_binary_variance_share.png")


# --------------------------------------------------------------------------------------
# 02 — size floor by algorithm
# --------------------------------------------------------------------------------------
def fig02(selection):
    f = "02"
    grid = selection["grid"]
    note(f, "source: rerun_2026-09-27/final_selection.json -> grid[*].algorithm, "
            "grid[*].passes_floor (cross-checked against final_selection_grid.csv)")
    with open(SELECTION_GRID_CSV, newline="") as fh:
        csv_rows = {r["combination"]: r["passes_floor"] == "True" for r in csv.DictReader(fh)}
    check(f, "json vs csv passes_floor agree", all(csv_rows[k] == v["passes_floor"]
                                                   for k, v in grid.items()), True)
    passes, fails = collections.Counter(), collections.Counter()
    for v in grid.values():
        (passes if v["passes_floor"] else fails)[v["algorithm"]] += 1
    algos = sorted(set(passes) | set(fails), key=lambda a: (-passes[a], a))
    check(f, "algorithms in grid", len(algos), 9)
    check(f, "combinations per algorithm all 10",
          all(passes[a] + fails[a] == 10 for a in algos), True)
    check(f, "total passing", sum(passes.values()), 41)
    check(f, "total failing", sum(fails.values()), 49)
    check(f, "step1_summary n_survivors", selection["step1_summary"]["n_survivors"], 41)
    note(f, "sort rule: combinations passing the floor, most first; ties alphabetical")
    note(f, "order: " + ", ".join(f"{a} {passes[a]}/{fails[a]}" for a in algos))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    _style(ax)
    ys = list(range(len(algos)))[::-1]
    p = [passes[a] for a in algos]
    ax.barh(ys, p, color=NEUTRAL_DARK, height=0.72, edgecolor=SURFACE, linewidth=2)
    ax.barh(ys, [fails[a] for a in algos], left=p, color=NEUTRAL_LIGHT, height=0.72,
            edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(ys)
    ax.set_yticklabels(algos)
    ax.set_xlim(0, 10)
    ax.set_xlabel("Combinations (5 k values × 2 approaches = 10 per algorithm)")
    _finish(fig, "Step 1: combinations passing the 5% minimum-cluster-size floor, by "
                 "algorithm (41 pass, 49 fail)",
            [_patch(NEUTRAL_DARK, "Passes the 5% floor"),
             _patch(NEUTRAL_LIGHT, "Fails the 5% floor")])
    note(f, f"bars: {len(algos)} (stacked pass/fail)")
    _save(fig, f, "02_size_floor_by_algorithm.png")


# --------------------------------------------------------------------------------------
# 03 — the 41 survivors by macro-silhouette, top 15
# --------------------------------------------------------------------------------------
def fig03(selection):
    f = "03"
    grid = selection["grid"]
    surv = selection["step1_summary"]["survivors"]
    note(f, "source: final_selection.json -> step1_summary.survivors, "
            "grid[*].silhouette_macro; winner from step3_winner")
    check(f, "survivor count", len(surv), 41)
    # Step 3's cross-approach comparison is on macro-silhouette (methodology.step3_ranking:
    # "cross-approach comparison uses macro-silhouette and Davies-Bouldin only"); within an
    # approach the composite rank breaks ties.
    ranks = selection["step3_ranks"]
    order = sorted(surv, key=lambda k: (-grid[k]["silhouette_macro"],
                                        ranks[k.split("|")[0]][k]["composite_rank_macro"], k))
    top = order[:15]
    check(f, "top bar", top[0], "A|birch|2")
    check(f, "top bar macro-silhouette", grid[top[0]]["silhouette_macro"], 0.3385)
    check(f, "step3_winner", selection["step3_winner"], top[0])
    note(f, "sort rule: macro-silhouette descending (Step 3's cross-approach metric), "
            "ties by within-approach composite rank")
    note(f, "top 15: " + ", ".join(f"{k} {grid[k]['silhouette_macro']}" for k in top))
    fig, ax = plt.subplots(figsize=(9, 6.4))
    _style(ax)
    cols = [(APPROACH_WINNER if k == top[0] else APPROACH_COLOUR)[k[0]] for k in top]
    _hbar_ranking(ax, top, [grid[k]["silhouette_macro"] for k in top], cols)
    ax.set_xlabel("Macro-averaged silhouette (higher is better)")
    _finish(fig, "Step 3: the 41 size-floor survivors ranked by macro-silhouette "
                 "(top 15 shown)",
            [_patch(BLUE_DARK, "Leader (A|birch|2)"), _patch(BLUE, "Approach A"),
             _patch(ORANGE, "Approach B")], ncol=3)
    note(f, "bars: 15")
    _save(fig, f, "03_survivors_macro_silhouette.png")


# --------------------------------------------------------------------------------------
# 04 — leader validation
# --------------------------------------------------------------------------------------
def fig04(selection):
    f = "04"
    v = selection["step4a_winner_validation"]
    mono = selection["step4a_monotonicity"]
    base = selection["population_base_rates_pct"]["by_band"]
    grid_bands = selection["grid"]["A|birch|2"]["per_band"]
    note(f, "source: final_selection.json -> step4a_winner_validation.band{b}.clusters[c]."
            "variables.{life,ci}_coverage_ratio.pct_meeting_benchmark, healthier cluster = "
            "step4a_monotonicity.band{b}.overall_health_order_best_to_worst[0]; base = "
            "population_base_rates_pct.by_band")
    exp = {"life_coverage_ratio": ([4.2, 15.8, 29.8, 14.6, 5.4], [3.7, 15.9, 30.2, 14.0, 4.9]),
           "ci_coverage_ratio": ([4.1, 7.7, 12.2, 9.1, 4.6], [3.7, 7.6, 12.5, 8.7, 4.3])}
    series = {}
    for var, (exp_c, exp_b) in exp.items():
        cl, bs = [], []
        for i, b in enumerate(range(1, 6)):
            band = v[f"band{b}"]
            # The validated block must be A|birch|2's own partition.
            check(f, f"band {b} cluster shares = A|birch|2 grid",
                  sorted(c["share_pct"] for c in band["clusters"].values()),
                  sorted(grid_bands[str(b)]["cluster_shares_pct"]))
            best = str(mono[f"band{b}"]["overall_health_order_best_to_worst"][0])
            c = band["clusters"][best]["variables"][var]["pct_meeting_benchmark"]
            check(f, f"{var} band {b} healthier cluster", c, exp_c[i])
            check(f, f"{var} band {b} base rate", base[str(b)][var], exp_b[i])
            cl.append(c)
            bs.append(base[str(b)][var])
        series[var] = (cl, bs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    bands = [1, 2, 3, 4, 5]
    w = 0.38
    for ax, (var, title) in zip(axes, (("life_coverage_ratio", "Life cover ≥ 9× income"),
                                       ("ci_coverage_ratio", "CI cover ≥ 3.9× income"))):
        _style(ax, "y")
        cl, bs = series[var]
        ax.bar([b - w / 2 for b in bands], cl, w, color=BLUE_DARK, edgecolor=SURFACE,
               linewidth=2)
        ax.bar([b + w / 2 for b in bands], bs, w, color=NEUTRAL_LIGHT, edgecolor=SURFACE,
               linewidth=2)
        ax.set_xticks(bands)
        ax.set_xticklabels([f"Band {b}" for b in bands])
        ax.set_title(title, loc="left", fontsize=11, color=TEXT)
    axes[0].set_ylabel("% of users clearing the benchmark")
    _finish(fig, "Step 4A: A|birch|2's healthier cluster vs the band base rate, by age band",
            [_patch(BLUE_DARK, "A|birch|2 healthier cluster"),
             _patch(NEUTRAL_LIGHT, "Band base rate (all users)")])
    note(f, "bars: 20 (2 panels × 5 bands × 2)")
    _save(fig, f, "04_leader_validation_birch.png")


# --------------------------------------------------------------------------------------
# 05 — leader split drivers
# --------------------------------------------------------------------------------------
def fig05(selection):
    f = "05"
    prof = selection["step4b_survivor_profiles"]["A|birch|2"]["by_band"]
    note(f, "source: final_selection.json -> step4b_survivor_profiles['A|birch|2'].by_band"
            "[b].separation.spread_scaled_units (share = spread / sum of spreads)")
    shares = {}
    for b in range(1, 6):
        sp = prof[str(b)]["separation"]["spread_scaled_units"]
        tot = sum(sp.values())
        shares[b] = {k: x / tot for k, x in sp.items()}
        stored = prof[str(b)]["separation"]
        check(f, f"band {b} top feature", stored["top_feature"], "investment_rate")
        # The stored spreads are themselves rounded to 3 dp, so a share recomputed from
        # them can differ from the stored share in the third decimal (band 2: 0.4695 vs
        # 0.47).
        check(f, f"band {b} computed share = stored share",
              round(shares[b]["investment_rate"], 4),
              stored["top_feature_share_of_total_separation"], tol=0.001)
    for b, e in zip(range(1, 6), (50.5, 47.0, 39.9, 42.0, 44.2)):
        check(f, f"band {b} investment_rate %",
              round(100 * prof[str(b)]["separation"]["top_feature_share_of_total_separation"],
                    1), e)
    mean = {k: sum(shares[b][k] for b in shares) / 5 for k in shares[1]}
    named = sorted(mean, key=lambda k: -mean[k])[:3]
    check(f, "largest feature on average", named[0], "investment_rate")
    note(f, f"segments: {named} + 'other {len(mean) - 3} features' (ranked by mean share)")
    colours = [BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    _style(ax)
    ys = [5 - b for b in range(1, 6)]
    left = [0.0] * 5
    handles = []
    for feat, col in zip(named + ["other"], colours + [NEUTRAL_LIGHT]):
        vals = [100 * (shares[b][feat] if feat != "other" else
                       sum(x for k, x in shares[b].items() if k not in named))
                for b in range(1, 6)]
        ax.barh(ys, vals, left=left, color=col, height=0.66, edgecolor=SURFACE, linewidth=2)
        left = [l + x for l, x in zip(left, vals)]
        handles.append(_patch(col, feat if feat != "other"
                              else f"other {len(mean) - 3} features"))
    ax.set_yticks(ys)
    ax.set_yticklabels([f"Band {b}" for b in range(1, 6)])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of total separation between the two clusters (%)")
    _finish(fig, "Step 4B: what separates A|birch|2's two clusters, by age band",
            handles, ncol=4)
    note(f, "bars: 5 (stacked, 4 segments each)")
    _save(fig, f, "05_leader_split_drivers_birch.png")


# --------------------------------------------------------------------------------------
# 06 — with vs without net_worth_to_income
# --------------------------------------------------------------------------------------
def fig06(dsr):
    f = "06"
    sr = dsr["spec_results"]
    note(f, "source: domain_split_results.json -> spec_results[spec].gate_passing_combinations "
            "(Investment: both bars from this same grid, i.e. the ORIGINAL Investment benchmark "
            "set, before the portfolio_concentration and risk_age_deviation fixes)")
    n = {s: len(sr[s]["gate_passing_combinations"]) for s in sr}
    check(f, "Banking without net worth", n["banking_core4"], 9)
    check(f, "Banking with net worth", n["banking_plus_nw"], 4)
    check(f, "Investment with net worth", n["investment_plus_nw"], 0)
    note(f, f"Investment without net worth, same grid: {n['investment_core3']}")
    with open(DS_GRID_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for s in ("banking_core4", "banking_plus_nw", "investment_core3", "investment_plus_nw"):
        c = sum(1 for r in rows if r["spec"] == s and r["passes_floor"] == "True"
                and r["passes_gate2_headroom"] == "True")
        check(f, f"{s} json vs grid csv", c, n[s])
    cats = ["Banking", "Investment (original benchmark set)"]
    without = [n["banking_core4"], n["investment_core3"]]
    with_nw = [n["banking_plus_nw"], n["investment_plus_nw"]]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    _style(ax)
    ys = [1, 0]
    h = 0.36
    ax.barh([y + h / 2 for y in ys], without, h, color=NEUTRAL_DARK, edgecolor=SURFACE,
            linewidth=2)
    ax.barh([y - h / 2 for y in ys], with_nw, h, color=NEUTRAL_LIGHT, edgecolor=SURFACE,
            linewidth=2)
    ax.set_yticks(ys)
    ax.set_yticklabels(cats)
    ax.set_xlabel("Gate-passing combinations (of 80)")
    ax.xaxis.get_major_locator().set_params(integer=True)
    _finish(fig, "Gate-passers without vs with net_worth_to_income_ratio, the same grid",
            [_patch(NEUTRAL_DARK, "Without net_worth_to_income (shipped feature set)"),
             _patch(NEUTRAL_LIGHT, "With net_worth_to_income")], bottom=0.2)
    note(f, "bars: 4")
    _save(fig, f, "06_net_worth_with_without.png")


# --------------------------------------------------------------------------------------
# 07-08 — Banking / Insurance gate-passers
# --------------------------------------------------------------------------------------
def fig_domain(f, dsr, spec, domain, winner, n_expected, fname):
    sr = dsr["spec_results"][spec]
    gp = sr["gate_passing_combinations"]
    grid, ranks = sr["grid"], sr["step3_ranks"]
    note(f, f"source: domain_split_results.json -> spec_results.{spec}."
            "gate_passing_combinations, grid[*].silhouette_macro, "
            "step3_ranks[approach][*].composite_rank_macro")
    check(f, "bar count", len(gp), n_expected)
    # domain_split_clustering.py:650-653 —
    #   winner = max(gate_pass, key=(silhouette_macro, -composite_rank_macro))
    order = sorted(gp, key=lambda k: (-grid[k]["silhouette_macro"],
                                      ranks[k.split("|")[0]][k]["composite_rank_macro"]))
    check(f, "top bar", order[0], winner)
    check(f, "file's own winner", sr["winner"], winner)
    note(f, "sort rule: macro-silhouette descending, ties by within-approach composite rank "
            "(the selection code's own max key, domain_split_clustering.py:652-653)")
    note(f, "order: " + ", ".join(f"{k} {grid[k]['silhouette_macro']}" for k in order))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(order) + 2.4))
    _style(ax)
    cols = [(APPROACH_WINNER if k == winner else APPROACH_COLOUR)[k[0]] for k in order]
    _hbar_ranking(ax, order, [grid[k]["silhouette_macro"] for k in order], cols)
    ax.set_xlabel("Macro-averaged silhouette (higher is better)")
    present = sorted({k[0] for k in order})
    handles = [_patch(APPROACH_WINNER[winner[0]], f"Winner ({winner})")] + \
              [_patch(APPROACH_COLOUR[a], f"Approach {a}") for a in present]
    _finish(fig, f"{domain}: the {len(order)} gate-passing combinations ranked by "
                 "macro-silhouette", handles, ncol=3,
            bottom=0.1 if len(order) > 10 else 0.15)
    note(f, f"bars: {len(order)}")
    return fig, order


def fig07(dsr):
    fig, _ = fig_domain("07", dsr, "banking_core4", "Banking", "A|kmeans|2", 9, "")
    check("07", "winner macro-silhouette",
          dsr["spec_results"]["banking_core4"]["grid"]["A|kmeans|2"]["silhouette_macro"], 0.2588)
    _save(fig, "07", "07_banking_gate_passers.png")


def fig08(dsr):
    fig, _ = fig_domain("08", dsr, "insurance_core3", "Insurance", "B|kmedoids|3", 16, "")
    check("08", "winner macro-silhouette",
          dsr["spec_results"]["insurance_core3"]["grid"]["B|kmedoids|3"]["silhouette_macro"],
          0.3516)
    _save(fig, "08", "08_insurance_gate_passers.png")


# --------------------------------------------------------------------------------------
# 09 — Investment gate-passers, V2 benchmark set
# --------------------------------------------------------------------------------------
def fig09(raf):
    f = "09"
    v2 = raf["variants"]["V2_bench_drop_only"]
    note(f, "source: risk_age_final.json -> variants.V2_bench_drop_only.gate_passers, "
            ".per_band_screen[*].{macro_sil, passes_per_band}, "
            ".passing_gate_and_per_band_floor")
    check(f, "V2 excludes risk_age_deviation from benchmarks", v2["rad_in_benchmarks"], False)
    gp, scr = v2["gate_passers"], v2["per_band_screen"]
    check(f, "bar count", len(gp), 15)
    order = sorted(gp, key=lambda k: (not scr[k]["passes_per_band"], -scr[k]["macro_sil"]))
    passing = [k for k in gp if scr[k]["passes_per_band"]]
    check(f, "per-band-floor passers", passing, ["A|fuzzy_cmeans|2"])
    check(f, "file's passing_gate_and_per_band_floor", v2["passing_gate_and_per_band_floor"],
          ["A|fuzzy_cmeans|2"])
    check(f, "top bar", order[0], "A|fuzzy_cmeans|2")
    check(f, "winner macro-silhouette", scr["A|fuzzy_cmeans|2"]["macro_sil"], 0.2481)
    note(f, "sort rule: passes the per-band normal-population floor first, then "
            "macro-silhouette descending")
    note(f, "order: " + ", ".join(f"{k} {scr[k]['macro_sil']}"
                                  f"{' (per-band floor PASS)' if scr[k]['passes_per_band'] else ''}"
                                  for k in order))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(order) + 2.4))
    _style(ax)
    cols = [(APPROACH_WINNER if k == "A|fuzzy_cmeans|2" else APPROACH_COLOUR)[k[0]]
            for k in order]
    _hbar_ranking(ax, order, [scr[k]["macro_sil"] for k in order], cols)
    ax.set_xlabel("Macro-averaged silhouette (higher is better)")
    _finish(fig, "Investment (V2 benchmark set): the 15 gate-passers, per-band floor passers "
                 "first, then macro-silhouette",
            [_patch(BLUE_DARK, "Winner, the only per-band-floor pass (A|fuzzy_cmeans|2)"),
             _patch(BLUE, "Approach A"), _patch(ORANGE, "Approach B")], ncol=3, bottom=0.1)
    note(f, "bars: 15")
    _save(fig, f, "09_investment_gate_passers.png")


# --------------------------------------------------------------------------------------
# 10 — Banking: why 2 tiers
# --------------------------------------------------------------------------------------
def fig10(dist):
    f = "10"
    combos = {c["key"]: c for c in dist["domains"]["banking"]["combinations"]}
    alt, win = combos["A|fuzzy_cmeans|3"], combos["A|kmeans|2"]
    note(f, "source: adjacent_tier_distinctness.json -> domains.banking.combinations"
            "[key].macro_sil, .min_rho, .distinct.groups.band{b}.tier_means[t]."
            "healthy_debt_ratio, .adjacent_pairs[*].mean_inverting_vars")
    check(f, "winner macro-silhouette", win["macro_sil"], 0.2588)
    check(f, "3-tier alternative macro-silhouette", alt["macro_sil"], 0.1693)
    check(f, "winner min rho", win["min_rho"], 1.0)
    inv_bands = [b for b in range(1, 6) if any(
        "healthy_debt_ratio" in p["mean_inverting_vars"]
        for p in alt["distinct"]["groups"][f"band{b}"]["adjacent_pairs"])]
    check(f, "bands where healthy_debt_ratio inverts (3-tier)", len(inv_bands), 4)
    win_inv = sum(len(p["mean_inverting_vars"]) for b in range(1, 6)
                  for p in win["distinct"]["groups"][f"band{b}"]["adjacent_pairs"])
    check(f, "winner inversions, any variable, any band", win_inv, 0)
    note(f, f"inverting bands (3-tier): {inv_bands}")
    tm = lambda c, b: c["distinct"]["groups"][f"band{b}"]["tier_means"]
    alt_m = {b: [tm(alt, b)[str(t)]["healthy_debt_ratio"] for t in range(3)] for b in range(1, 6)}
    win_m = {b: [tm(win, b)[str(t)]["healthy_debt_ratio"] for t in range(2)] for b in range(1, 6)}
    note(f, "3-tier means T0/T1/T2 by band: " + "; ".join(f"b{b} {alt_m[b]}" for b in alt_m))
    note(f, "winner means T0/T1 by band: " + "; ".join(f"b{b} {win_m[b]}" for b in win_m))

    fig = plt.figure(figsize=(12, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.25, 1.0])
    ax_a = fig.add_subplot(gs[0])
    _style(ax_a)
    _hbar_ranking(ax_a, ["A|kmeans|2\n(2 tiers)", "A|fuzzy_cmeans|3\n(3 tiers)"],
                  [win["macro_sil"], alt["macro_sil"]], [BLUE_DARK, BLUE])
    ax_a.set_xlabel("Macro-silhouette")
    ax_a.set_title("(a) Clustering quality", loc="left", fontsize=11, color=TEXT)
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2], sharey=ax_b)
    tier_cols3 = [BLUE_DARK, BLUE, BLUE_LIGHT]
    bands = list(range(1, 6))
    for ax, means, k, cols, title in (
            (ax_b, alt_m, 3, tier_cols3, "(b) 3-tier alternative A|fuzzy_cmeans|3"),
            (ax_c, win_m, 2, [BLUE_DARK, BLUE], "Winner A|kmeans|2")):
        _style(ax, "y")
        w = 0.8 / k
        for t in range(k):
            ax.bar([b - 0.4 + w * (t + 0.5) for b in bands], [means[b][t] for b in bands], w,
                   color=cols[t], edgecolor=SURFACE, linewidth=2)
        ax.set_xticks(bands)
        ax.set_xticklabels([f"B{b}" for b in bands])
        ax.set_xlabel("Age band")
        ax.set_title(title, loc="left", fontsize=11, color=TEXT)
    ax_b.set_ylabel("Mean healthy_debt_ratio")
    ax_b.set_ylim(0, 1)
    plt.setp(ax_c.get_yticklabels(), visible=False)
    _finish(fig, "Banking: why 2 tiers, not 3 — lower silhouette, and healthy_debt_ratio "
                 "inverts between the top two tiers in 4 of 5 bands",
            [_patch(BLUE_DARK, "Tier 0 (healthiest) / winner"), _patch(BLUE, "Tier 1"),
             _patch(BLUE_LIGHT, "Tier 2 (3-tier model only)")], ncol=3, bottom=0.14)
    note(f, "bars: panel (a) 2; panel (b) 15 (3-tier) + 10 (winner)")
    _save(fig, f, "10_banking_why_two_tiers.png")


# --------------------------------------------------------------------------------------
# 11 — Investment: who fills the third tier
# --------------------------------------------------------------------------------------
SLIVER_ALGOS = ["B|gmm_diag|3", "B|kmeans|3", "B|kmedoids|3", "B|gmm_full|3",
                "B|agglomerative|3"]


def fig11(dist):
    f = "11"
    combos = {c["key"]: c for c in dist["domains"]["investment"]["combinations"]}
    note(f, "source: tier-0 share per combination from adjacent_tier_distinctness.json -> "
            "domains.investment.combinations[key].tier_shares_pct[0]; membership = read-only "
            "count of saved labels (.risk_age_diagnostic_grids.pkl['rad100'], the V2 geometry "
            "adjacent_tier_distinctness.py reads) against synthetic_users.csv profile_type. "
            "No saved result file stores per-profile composition for these five.")
    k3 = sorted(k for k, c in combos.items() if c["k"] == 3)
    note(f, "all k=3 Investment gate-passers and their smallest (top) tier: " + ", ".join(
        f"{k} {combos[k]['tier_shares_pct'][0]}%" for k in k3))
    # "~8.5% sliver ... five different algorithms" (FINAL_TIER_DECISIONS.md:56-57): the five
    # k=3 combinations whose top tier is 8.4-8.8%. B|birch|3 (9.36%) and the two ~26.5% tiers
    # are outside that band.
    sliver = [k for k in k3 if 8.0 <= combos[k]["tier_shares_pct"][0] <= 9.0]
    check(f, "combinations with a ~8.5% top tier", sorted(sliver), sorted(SLIVER_ALGOS))
    with open(RISK_AGE_CACHE, "rb") as fh:
        grid, labels = pickle.load(fh)["rad100"]
    with open(DATA_PATH, newline="") as fh:
        profile = [r["profile_type"] for r in csv.DictReader(fh)]
    n_norm = profile.count("normal")
    rows = {}
    for key in SLIVER_ALGOS:
        lab = labels[key]
        check(f, f"{key} macro-sil cache = file", round(grid[key]["silhouette_macro"], 4),
              combos[key]["macro_sil"])
        target = combos[key]["tier_shares_pct"][0]
        sizes = collections.Counter(int(x) for x in lab)
        match = [c for c, n in sizes.items() if abs(100 * n / len(lab) - target) < 0.01]
        check(f, f"{key} exactly one cluster at the top-tier share {target}%", len(match), 1)
        members = [profile[i] for i, x in enumerate(lab) if int(x) == match[0]]
        cnt = collections.Counter(members)
        # Links the identified cluster back to the saved file: share of ALL normals in it.
        check(f, f"{key} share of normals in tier = file",
              round(100 * cnt["normal"] / n_norm, 2),
              combos[key]["normal_floor"]["pooled_shares_pct"][0], tol=0.01)
        n = len(members)
        rows[key] = {"n": n, "over_invested": cnt["over_invested"],
                     "high_debt_investor": cnt["high_debt_investor"], "normal": cnt["normal"],
                     "other_injected": n - cnt["over_invested"] - cnt["high_debt_investor"]
                     - cnt["normal"], "other_detail": dict(cnt)}
    # Corrected wording (replaces the brief's unsupported "99%+ in most cases"): each
    # figure below was read off this same count in the 2026-09-28 run and is now pinned.
    expected = {  # key: (injected %, over_invested + high_debt_investor %, normal %)
        "B|gmm_full|3": (99.0, 89.1, 1.0), "B|gmm_diag|3": (98.5, 87.8, 1.5),
        "B|kmedoids|3": (97.9, 87.8, 2.1), "B|agglomerative|3": (97.5, 87.4, 2.5),
        "B|kmeans|3": (97.4, 87.3, 2.6)}
    for k, r in rows.items():
        r["pct_injected"] = 100 * (r["n"] - r["normal"]) / r["n"]
        r["pct_oi_hd"] = 100 * (r["over_invested"] + r["high_debt_investor"]) / r["n"]
        r["pct_other"] = 100 * r["other_injected"] / r["n"]
        r["pct_normal"] = 100 * r["normal"] / r["n"]
        inj, oi_hd, nrm = expected[k]
        check(f, f"{k} injected %", round(r["pct_injected"], 1), inj)
        check(f, f"{k} over_invested + high_debt_investor %", round(r["pct_oi_hd"], 1), oi_hd)
        check(f, f"{k} normal %", round(r["pct_normal"], 1), nrm)
        note(f, f"{k}: top tier n={r['n']} = {100 * r['n'] / 7000:.2f}% of all users; "
                f"segments {r['pct_oi_hd']:.2f} / {r['pct_other']:.2f} / "
                f"{r['pct_normal']:.2f}% — {r['other_detail']}")
    shares = [100 * r["n"] / 7000 for r in rows.values()]
    check(f, "top-tier share of all users within 8.46-8.79%",
          all(8.455 <= x <= 8.795 for x in shares), True)
    order = sorted(rows, key=lambda k: (-rows[k]["pct_injected"], k))
    check(f, "sort order (injected share, descending)", order, list(expected))
    note(f, "sort rule: total injected share, descending")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    _style(ax)
    ys = list(range(len(order)))[::-1]
    left = [0.0] * len(order)
    segs = (("pct_oi_hd", ORANGE_DARK, "over_invested + high_debt_investor (injected)"),
            ("pct_other", ORANGE_LIGHT, "other injected profiles"),
            ("pct_normal", NEUTRAL_DARK, "normal users"))
    for field, col, _ in segs:
        vals = [rows[k][field] for k in order]
        ax.barh(ys, vals, left=left, color=col, height=0.66, edgecolor=SURFACE, linewidth=2)
        left = [l + x for l, x in zip(left, vals)]
    ax.set_yticks(ys)
    ax.set_yticklabels(order)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the third tier's members (%)")
    _finish(fig, "Investment: who fills the third tier of the five k=3 alternatives with a "
                 "~8.5% top tier", [_patch(c, l) for _, c, l in segs], ncol=3)
    note(f, "bars: 5 (stacked, 3 segments)")
    _save(fig, f, "11_investment_third_tier_membership.png")


# --------------------------------------------------------------------------------------
# 12 — Insurance: are tiers 1 and 2 different?
# --------------------------------------------------------------------------------------
def fig12(dist):
    f = "12"
    c = {x["key"]: x for x in dist["domains"]["insurance"]["combinations"]}["B|kmedoids|3"]
    pairs = {p["pair"]: p for p in c["distinct"]["groups"]["all"]["adjacent_pairs"]}
    note(f, "source: adjacent_tier_distinctness.json -> domains.insurance.combinations"
            "['B|kmedoids|3'].distinct.groups.all.adjacent_pairs[*].health_by_variable, "
            ".density_inverting_vars; cutoff_iqr")
    t12 = pairs["T1|T2"]
    check(f, "T1|T2 life separation (IQR)", t12["health_by_variable"]["life_coverage_ratio"],
          0.046)
    check(f, "T1|T2 CI separation (IQR)", t12["health_by_variable"]["ci_coverage_ratio"], 0.064)
    check(f, "life coverage inverts on density between T1 and T2",
          "life_coverage_ratio" in t12["density_inverting_vars"], True)
    ac = c["age_confound"]
    check(f, "band x tier cells under 5% of band normals", ac["cells_below_5pct_of_band_normals"],
          7)
    check(f, "of cells", ac["n_cells"], 15)
    note(f, "7 of 15 cells: reported, not charted (a single count)")
    cutoff = dist["cutoff_iqr"]
    cats = [("T0|T1", "Tier 0 vs tier 1\n(Covered vs the next tier)"),
            ("T1|T2", "Tier 1 vs tier 2\n(merged as Not Covered)")]
    note(f, "values: " + "; ".join(
        f"{p}: life {pairs[p]['health_by_variable']['life_coverage_ratio']}, CI "
        f"{pairs[p]['health_by_variable']['ci_coverage_ratio']}" for p, _ in cats))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _style(ax, "y")
    w = 0.36
    for i, (p, _) in enumerate(cats):
        hv = pairs[p]["health_by_variable"]
        ax.bar(i - w / 2, hv["life_coverage_ratio"], w, color=ORANGE_DARK, edgecolor=SURFACE,
               linewidth=2)
        ax.bar(i + w / 2, hv["ci_coverage_ratio"], w, color=ORANGE, edgecolor=SURFACE,
               linewidth=2)
    ax.axhline(cutoff, color=TEXT_2, linewidth=1, linestyle="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([l for _, l in cats])
    ax.set_ylabel("Separation between adjacent tiers (IQR units)")
    _finish(fig, "Insurance B|kmedoids|3: separation between adjacent tiers on the two "
                 "coverage ratios",
            [_patch(ORANGE_DARK, "life_coverage_ratio"), _patch(ORANGE, "ci_coverage_ratio"),
             plt.Line2D([], [], color=TEXT_2, linewidth=1, linestyle="--",
                        label=f"Distinctness cutoff ({cutoff} IQR)")], ncol=3)
    note(f, "bars: 4")
    _save(fig, f, "12_insurance_tier1_vs_tier2.png")


# --------------------------------------------------------------------------------------
# 13 — archetype placement under the domain split (Step 4)
# --------------------------------------------------------------------------------------
PLACEMENT_PROFILES = ("wealthy_exposed", "underinsured_saver", "protection_first", "normal")


def fig13(dsr):
    f = "13"
    step4 = dsr["step4_profile_tier_combinations"]
    note(f, "source (plotted): domain_split_results.json -> step4_profile_tier_combinations"
            "[profile].exact_tier_distribution_pct.{banking,insurance}. Tier index = health "
            "rank, 0 = healthiest. Banking is the shipped 2-tier A|kmeans|2 (tier 0 = Stable). "
            "Insurance is B|kmedoids|3's NATIVE 3-tier rank; shipped Covered = native tier 0, "
            "Not Covered = native tiers 1 + 2 (insurance_model.json tier_merge).")
    note(f, "cross-check 1: domain_split_tier_assignments.csv (tier_banking, tier_insurance — "
            "the same native ranks) joined to synthetic_users.csv profile_type by user_id")
    note(f, "cross-check 2: layer3_coverage_audit.json -> tier_replay (shipped labels) and "
            "step2_coverage[*].per_rule_pct (joint rates: R1 = Stable AND Not Covered, "
            "R2 = Covered AND Building)")
    vals = {}
    for p in PLACEMENT_PROFILES:
        e = step4[p]["exact_tier_distribution_pct"]
        check(f, f"{p} insurance tiers are the native 3", sorted(e["insurance"]), ["0", "1", "2"])
        vals[p] = {"stable": e["banking"]["0"], "covered": e["insurance"]["0"],
                   "n": step4[p]["n"]}
        note(f, f"{p} (n={vals[p]['n']}): Banking Stable {e['banking']['0']}%, Building "
                f"{e['banking']['1']}%; Insurance native {e['insurance']} -> Covered "
                f"{e['insurance']['0']}%, Not Covered "
                f"{round(e['insurance']['1'] + e['insurance']['2'], 1)}%")
    check(f, "wealthy_exposed Banking Stable", vals["wealthy_exposed"]["stable"], 93.2)
    check(f, "wealthy_exposed Insurance Covered", vals["wealthy_exposed"]["covered"], 0.0)
    check(f, "underinsured_saver Banking Stable", vals["underinsured_saver"]["stable"], 96.2)
    check(f, "underinsured_saver Insurance Covered", vals["underinsured_saver"]["covered"], 0.0)
    check(f, "protection_first Insurance Covered", vals["protection_first"]["covered"], 100.0)
    check(f, "protection_first Banking weak (Building)",
          round(100 - vals["protection_first"]["stable"], 1), 87.0)
    check(f, "protection_first Banking Stable", vals["protection_first"]["stable"], 13.0)

    # Cross-check 1: the per-user assignment file reproduces every plotted share.
    with open(DATA_PATH, newline="") as fh:
        profile = {r["user_id"]: r["profile_type"] for r in csv.DictReader(fh)}
    with open(DS_ASSIGN_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    check(f, "assignment rows joined by user_id", sum(r["user_id"] in profile for r in rows),
          7000)
    check(f, "profile_type agrees between the two files",
          sum(profile[r["user_id"]] != r["profile_type"] for r in rows), 0)
    for p in PLACEMENT_PROFILES:
        g = [r for r in rows if profile[r["user_id"]] == p]
        check(f, f"{p} n (assignments)", len(g), vals[p]["n"])
        check(f, f"{p} Stable % (assignments)",
              round(100 * sum(r["tier_banking"] == "0" for r in g) / len(g), 1),
              vals[p]["stable"])
        check(f, f"{p} Covered % (assignments)",
              round(100 * sum(r["tier_insurance"] == "0" for r in g) / len(g), 1),
              vals[p]["covered"])
    # Tier 0 carries the shipped "Stable"/"Covered" label: its population share must equal
    # the shipped tier-0 share the Layer 3 audit replayed through the artifacts' tier_map.
    with open(L3_AUDIT_JSON) as fh:
        audit = json.load(fh)
    rep = audit["tier_replay"]
    check(f, "shipped Banking labels", rep["banking"]["labels"], ["Stable", "Building"])
    check(f, "shipped Insurance labels", rep["insurance"]["labels"], ["Covered", "Not Covered"])
    check(f, "Banking tier 0 share of all users = shipped Stable share",
          round(100 * sum(r["tier_banking"] == "0" for r in rows) / len(rows), 1),
          rep["banking"]["shares_pct"][0])
    check(f, "Insurance native tier 0 share of all users = shipped Covered share",
          round(100 * sum(r["tier_insurance"] == "0" for r in rows) / len(rows), 1),
          rep["insurance"]["shares_pct"][0])
    # Cross-check 2: Layer 3's joint rates are implied exactly by the plotted marginals
    # wherever one marginal is 0% or 100%.
    cov = {c["profile"]: c["per_rule_pct"] for c in audit["step2_coverage"]}
    for p in ("wealthy_exposed", "underinsured_saver"):
        check(f, f"{p} R1 (Stable AND Not Covered) = Stable when Covered is 0%",
              cov[p]["Wealth Without Protection"], vals[p]["stable"])
    check(f, "protection_first R2 (Covered AND Building) = Building when Covered is 100%",
          cov["protection_first"]["Protection First"],
          round(100 - vals["protection_first"]["stable"], 1))

    labels = {"wealthy_exposed": "wealthy_exposed", "underinsured_saver": "underinsured_saver",
              "protection_first": "protection_first", "normal": "normal population"}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    _style(ax)
    ys = list(range(len(PLACEMENT_PROFILES)))[::-1]
    h = 0.36
    ax.barh([y + h / 2 for y in ys], [vals[p]["stable"] for p in PLACEMENT_PROFILES], h,
            color=BLUE, edgecolor=SURFACE, linewidth=2)
    ax.barh([y - h / 2 for y in ys], [vals[p]["covered"] for p in PLACEMENT_PROFILES], h,
            color=ORANGE, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{labels[p]} (n={vals[p]['n']:,})" for p in PLACEMENT_PROFILES])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the profile's users (%)")
    _finish(fig, "Step 4: where the cross-domain archetypes land under the domain split",
            [_patch(BLUE, "Banking: Stable (A|kmeans|2)"),
             _patch(ORANGE, "Insurance: Covered (B|kmedoids|3, merged to 2 tiers)")])
    note(f, "bars: 8 (4 groups × 2)")
    _save(fig, f, "13_archetype_placement_domain_split.png")


def main():
    for path in INPUTS:
        if not os.path.isfile(path):
            sys.exit(f"ABORT: missing {path}")
    if sha256(DATA_PATH) != DATASET_SHA256:
        sys.exit("ABORT: synthetic_users.csv is not the expected current dataset")
    before = {p: sha256(p) for p in INPUTS}
    os.makedirs(OUT_DIR, exist_ok=True)

    load = lambda p: json.load(open(p))
    ablation, selection = load(ABLATION_JSON), load(SELECTION_JSON)
    dsr, raf, dist = load(DS_RESULTS_JSON), load(RISK_AGE_FINAL_JSON), load(DISTINCT_JSON)

    jobs = [("01", fig01, ablation), ("02", fig02, selection), ("03", fig03, selection),
            ("04", fig04, selection), ("05", fig05, selection), ("06", fig06, dsr),
            ("07", fig07, dsr), ("08", fig08, dsr), ("09", fig09, raf),
            ("10", fig10, dist), ("11", fig11, dist), ("12", fig12, dist),
            ("13", fig13, dsr)]
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    for fid, fn, arg in jobs:
        if only and fid not in only:
            continue
        try:
            fn(arg)
        except Mismatch as e:
            plt.close("all")
            SKIPPED[fid] = str(e)

    after = {p: sha256(p) for p in before}
    print("Input guard:")
    for p in before:
        state = "UNCHANGED" if before[p] == after[p] else "CHANGED !!"
        print(f"  {state:<10} {before[p][:16]}  {os.path.relpath(p, REPO_ROOT)}")
    if before != after:
        sys.exit("ABORT: an input file changed during the run")
    last = None
    for fid, line in REPORT:
        if fid != last:
            print(f"\n[{fid}]")
            last = fid
        print(f"  {line}")
    print("\nSKIPPED: " + (json.dumps(SKIPPED, indent=1) if SKIPPED else "none"))


if __name__ == "__main__":
    main()
