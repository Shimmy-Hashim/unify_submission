"""
run_correlation_retest.py — Re-run the canonical 12-algorithm benchmark AND the full
gate-based selection process against a redirected output directory.

WHY THIS EXISTS (July 2026, accumulation<->protection correlation retest):
  The correlation-fix retest needs the benchmark + selection outputs written into
  results/new_results/ while the CANONICAL files in results/ root stay byte-untouched
  and authoritative until a promotion decision is made separately.

  Rather than edit comprehensive_benchmark_11algo.py / final_selection.py to take an
  output-dir flag (touching canonical analysis code for a one-off comparison), this
  runner redirects their module-level output constants before calling main(). The
  METHODOLOGY IS THEREFORE BIT-IDENTICAL to the canonical scripts — same functions, same
  seed, same gates, same thresholds — only the write paths differ. That is exactly the
  property a before/after comparison needs: any difference in the result must come from
  the data, not from the analysis code.

  Belt-and-braces: the canonical files' SHA-256 hashes are recorded before and after the
  run and asserted unchanged, so "canonical untouched" is verified rather than assumed.

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/run_correlation_retest.py --out-dir new_results
"""

import argparse
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_ROOT = os.path.join(REPO_ROOT, "MLAI/clustering/results")
RULE = "=" * 92

# Canonical files that MUST NOT change during this run.
CANONICAL_GUARDED = [
    "comprehensive_results.json", "algorithm_metrics.csv",
    "final_selection.json", "final_selection_grid.csv", "final_selection_context.json",
    "plots/k_comparison.png", "plots/algorithm_heatmap.png",
]


def banner(title):
    print(f"\n{RULE}\n{title}\n{RULE}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot():
    """SHA-256 every guarded canonical file, keyed by its path relative to RESULTS_ROOT.

    A missing file is a hard error here, never a None. main() compares snapshots with
    `before[p] == after[p]`, and None == None is True — so hashing an absent file to None
    would make the integrity check report UNCHANGED for something it never read, and then
    print "All canonical files verified byte-identical" having verified nothing. That
    fires on a typo in CANONICAL_GUARDED, and on the most natural way to point this script
    at a different tree: reassigning RESULTS_ROOT moves the guard's base path along with
    the output path, silently emptying the guard exactly when it matters most. Redirect by
    passing an ABSOLUTE --out-dir instead, which wins the join in main() and leaves the
    guard reading the real results/ tree.
    """
    paths = {p: os.path.join(RESULTS_ROOT, p) for p in CANONICAL_GUARDED}
    missing = sorted(p for p, full in paths.items() if not os.path.isfile(full))
    if missing:
        raise SystemExit(f"ABORT: cannot guard canonical files — not found under "
                         f"{RESULTS_ROOT}: {missing}")
    return {p: sha256(full) for p, full in paths.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True,
                    help="Subdirectory of results/ to write into, e.g. new_results")
    args = ap.parse_args()

    out_dir = os.path.join(RESULTS_ROOT, args.out_dir)
    out_plots = os.path.join(out_dir, "plots")
    os.makedirs(out_plots, exist_ok=True)

    before = snapshot()
    print(RULE)
    print(f"CORRELATION-FIX RETEST — writing all outputs to results/{args.out_dir}/")
    print(f"canonical files guarded: {len(CANONICAL_GUARDED)}")
    print(RULE)

    # Each analysis module is imported inside main(), immediately above the redirect of
    # its own output constants — deliberately, so an import can never be separated from
    # the reassignment that keeps it off the canonical paths.

    # ---------------------------------------------------------------- STEP 3: benchmark
    t0 = time.time()
    banner("[STEP 3] 12-algorithm benchmark on the regenerated data")
    import comprehensive_benchmark_11algo as bench
    bench.RESULTS_DIR = out_dir
    bench.PLOTS_DIR = out_plots
    bench.main()
    print(f"\n[STEP 3] done in {time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- STEP 4: selection
    t1 = time.time()
    banner("[STEP 4] Gate-based selection process on the regenerated data")
    import final_selection as fs
    fs.RESULTS_DIR = out_dir
    fs.OUT_JSON = os.path.join(out_dir, "final_selection.json")
    # Reproduction check validates the re-fit grid against THIS run's benchmark output.
    fs.CANONICAL_JSON = os.path.join(out_dir, "comprehensive_results.json")
    # Fresh label cache — the data changed, so the old cache must never be reused.
    fs.LABEL_CACHE = os.path.join(out_dir, ".final_selection_labels.pkl")
    if os.path.exists(fs.LABEL_CACHE):
        os.remove(fs.LABEL_CACHE)
    fs.main()
    print(f"\n[STEP 4] selection done in {time.time() - t1:.0f}s")

    # Context script (base rates + discrimination lift) against the same new labels.
    banner("[STEP 4b] Base rates + discrimination lift")
    import final_selection_context as ctx
    ctx.LABEL_CACHE = fs.LABEL_CACHE
    ctx.RESULTS_DIR = out_dir
    ctx.OUT_JSON = os.path.join(out_dir, "final_selection_context.json")
    ctx.main()

    # ------------------------------------------------------- canonical integrity check
    after = snapshot()
    banner("CANONICAL INTEGRITY CHECK (results/ root must be unchanged)")
    ok = True
    for p in CANONICAL_GUARDED:
        same = before[p] == after[p]
        ok &= same
        print(f"  {'UNCHANGED' if same else 'MODIFIED !!':<12} {p}")
    if not ok:
        raise SystemExit("ABORT: a canonical file changed during the retest.")
    print("\n  All canonical files verified byte-identical (SHA-256).")
    print(f"\nOutputs written to results/{args.out_dir}/:")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
