"""
verify_production_artifacts.py — check the domain-split production artifacts on their own
terms, WITHOUT refitting anything (Unify, CCDS25-1170, July 2026).

WHY THIS EXISTS SEPARATELY FROM build_production_artifacts.py
  The builder proves the refit reproduces the validated result. That is a different claim
  from "the files that got written are usable and faithful". This script asserts the second
  one: it loads ONLY the committed artifacts (the pickled scalers, the JSON centroids and
  the tier maps), replays tier assignment on the dataset from those files alone, and checks
  the answer matches what the artifacts themselves claim.

  That is the exact path Phase 3 will take at inference time, so a pass here means the
  artifacts are self-sufficient — no clustering library call, no seed, no refit involved.

Checks per domain:
  1. The pickled scaler agrees with the JSON's plain center_/scale_ numbers (so a consumer
     that never unpickles gets the identical transform).
  2. Nearest-centroid assignment from the saved centroids reproduces the artifact's own
     stated population tier shares, exactly.
  3. Tier ordering is genuinely monotonic: each out-of-model benchmark's % clearing
     decreases from tier 0 down, on the artifact's own recorded profile.
  4. Every declared file exists and its SHA-256 matches the manifest.
  5. The shipped tier labels are the locked wording, and every per-band tier_map entry
     carries the label of ITS OWN health rank — the guard against a build that pinned a
     label to a cluster id, which is invalid because cluster 0 is the weak tier in Banking
     band 1 and the healthy tier in bands 2-5.

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/verify_production_artifacts.py
"""

import os
import json
import pickle
import hashlib

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ART_DIR = os.path.join(REPO_ROOT, "backend/ml/artifacts/domain_split")
DATA_PATH = os.path.join(REPO_ROOT, "MLAI/clustering/data/processed/synthetic_users.csv")

# The final user-facing tier wording, locked in the brief (Section 5, July 2026), healthiest
# first. Transcribed here independently of the builder so the check is a real assertion about
# what shipped, not a restatement of whatever the builder happened to write.
LOCKED_TIER_LABELS = {
    "banking": ["Stable", "Building"],
    "investment": ["On Track", "Getting Started"],
    "insurance": ["Covered", "Not Covered"],
}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_json(name):
    with open(os.path.join(ART_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def load_pickle(name):
    with open(os.path.join(ART_DIR, name), "rb") as f:
        return pickle.load(f)


def assign(df, art, scaler, age_scaler):
    """Replay tier assignment from the artifacts alone — the Phase 3 inference path."""
    feats = art["clustering_features"]
    tiers = np.empty(len(df), dtype=int)
    if art["approach"] == "A":
        for b, bv in art["bands"].items():
            m = (df["age_band"] == int(b)).to_numpy()
            X = scaler.transform(df.loc[m, feats])
            C = np.array([bv["centroids_scaled"][str(c)] for c in range(art["k_native"])])
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(2), axis=1)
            tiers[m] = [bv["tier_map"][str(c)]["health_rank"] for c in lab]
    else:
        X = np.hstack([scaler.transform(df[feats]),
                       age_scaler.transform(df[["age_midpoint"]])])
        C = np.array([art["model"]["centroids_scaled"][str(c)]
                      for c in range(art["k_native"])])
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(2), axis=1)
        smap = art["tier_merge"]["shipped_tier_map"]
        tiers = np.array([smap[str(c)]["shipped_tier"] for c in lab])
    return tiers


def shipped_profile(art):
    """The per-tier benchmark profile matching the SHIPPED tiers.

    The last branch is for an Approach B domain that ships its native clusters unmerged.
    No such domain exists today — Insurance is the only B domain and it is merged — so the
    branch is currently unreachable, but it read art["native_tier_benchmark_profile"], and
    the builder writes that key under art["model"]. It would have raised KeyError the first
    time it was ever taken."""
    if art.get("tier_merge", {}).get("applied"):
        return art["tier_merge"]["shipped_tier_benchmark_profile"]
    if art["approach"] == "A":
        return None            # per band; handled separately
    return art["model"]["native_tier_benchmark_profile"]


def main():
    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    manifest = load_json("manifest.json")
    df = pd.read_csv(DATA_PATH)

    print("=" * 96)
    print("DOMAIN-SPLIT PRODUCTION ARTIFACTS — verification (no refit)")
    print("=" * 96)

    print("\nmanifest")
    repro = manifest["reproduction_checks"]
    check("manifest records a fully-passing build", repro["all_passed"],
          f"{repro['passed']}/{repro['total']}")
    check("dataset unchanged since the build",
          manifest["dataset"]["sha256"] == sha256(DATA_PATH),
          manifest["dataset"]["sha256"][:16] + "...")
    for fname, digest in manifest["files"].items():
        p = os.path.join(ART_DIR, fname)
        check(f"{fname} present and unmodified",
              os.path.exists(p) and sha256(p) == digest)

    for domain in manifest["domains"]:
        art = load_json(f"{domain}_model.json")
        scaler = load_pickle(f"{domain}_scaler.pkl")
        age_scaler = (load_pickle(f"{domain}_age_scaler.pkl")
                      if art["approach"] == "B" else None)
        print(f"\n{domain} — {art['combination']}")

        check("pickled scaler matches the JSON's plain center_/scale_",
              np.allclose(scaler.center_, art["scaler"]["center_"])
              and np.allclose(scaler.scale_, art["scaler"]["scale_"]))
        if age_scaler is not None:
            check("pickled age scaler matches the JSON",
                  np.allclose(age_scaler.mean_, art["age_scaler"]["mean_"])
                  and np.allclose(age_scaler.scale_, art["age_scaler"]["scale_"]))

        tiers = assign(df, art, scaler, age_scaler)
        n_tiers = art["shipped_tiers"]
        got = [round(100 * float((tiers == t).mean()), 1) for t in range(n_tiers)]
        want = [round(float(x), 1) for x in
                art["shipped_tiers_summary"]["population_shares_pct"]]
        check("replayed tier shares match the artifact's own", got == want,
              f"{got} vs {want}")
        check("every shipped tier is non-empty and clears the 5% floor",
              all(g >= 5.0 for g in got), str(got))

        # Monotonicity, read off the artifact's own recorded per-tier profile.
        if art["approach"] == "A":
            profiles = [(f"band {b}", bv["tier_benchmark_profile"])
                        for b, bv in art["bands"].items()]
        else:
            profiles = [("pooled", shipped_profile(art))]
        bad = []
        for label, prof in profiles:
            ranks = sorted(prof, key=int)
            for var in prof[ranks[0]]["benchmarks"]:
                seq = [prof[r]["benchmarks"][var]["pct_clearing"] for r in ranks]
                if any(lower > upper + 1e-9 for upper, lower in zip(seq, seq[1:])):
                    bad.append(f"{label}/{var}: {seq}")
        check("% clearing each benchmark decreases from tier 0 downward",
              not bad, "; ".join(bad) if bad else f"{len(profiles)} group(s)")

        check("tier 0 carries this domain's locked healthy-tier label",
              art["shipped_tiers_summary"]["tier_labels"] == LOCKED_TIER_LABELS[domain],
              f"{art['shipped_tiers_summary']['tier_labels']} vs "
              f"{LOCKED_TIER_LABELS[domain]}")

        # The cluster-id trap: a label is only correct if it follows the HEALTH RANK of the
        # cluster in THAT band. A build that pinned labels to cluster ids would still pass
        # every check above, so assert the per-entry label explicitly, per band.
        locked = LOCKED_TIER_LABELS[domain]
        if art["approach"] == "A":
            wrong, by_id = [], {}
            for b, bv in art["bands"].items():
                for cid, e in bv["tier_map"].items():
                    if e["tier_label"] != locked[e["health_rank"]]:
                        wrong.append(f"band {b} cluster {cid}: {e['tier_label']} "
                                     f"at rank {e['health_rank']}")
                    by_id.setdefault(cid, set()).add(e["tier_label"])
            check("every band's tier_label matches that band's own health rank",
                  not wrong, "; ".join(wrong) if wrong
                  else f"{sum(len(bv['tier_map']) for bv in art['bands'].values())} entries")
            flipped = {c: sorted(v) for c, v in by_id.items() if len(v) > 1}
            print(f"  [note] cluster ids whose label differs across age bands: "
                  f"{flipped if flipped else 'none'}")
        else:
            smap = art["tier_merge"]["shipped_tier_map"]
            check("every native cluster's tier_label matches its shipped tier index",
                  all(e["tier_label"] == locked[e["shipped_tier"]] for e in smap.values()),
                  f"{len(smap)} entries")

    merge_art = load_json("insurance_model.json")
    tm = merge_art["tier_merge"]
    print("\ninsurance merge documentation")
    # The merge is the one derived, non-fitted piece of the whole set; assert it carries
    # enough detail to be REVERSED, not merely applied.
    required = ["merged_health_ranks", "native_cluster_ids_merged", "native_cluster_id_kept",
                "shipped_tier_map", "why"]
    missing = [k for k in required if not tm.get(k)]
    check("merge fully documented", not missing, f"missing {missing}" if missing else "")
    covered = (set(int(c) for c in tm["native_cluster_ids_merged"])
               | set(int(c) for c in tm["native_cluster_id_kept"]))
    check("every native cluster is accounted for by the merge",
          covered == set(range(merge_art["k_native"])), str(sorted(covered)))
    check("the pre-merge 3-cluster partition is preserved for audit",
          bool(merge_art["model"]["native_health_order_best_to_worst"])
          and bool(merge_art["model"]["native_tier_benchmark_profile"]))

    print("\n" + "=" * 96)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for name in failures:
            print(f"  - {name}")
        raise SystemExit(1)
    print("all checks passed — artifacts are self-sufficient and faithful")


if __name__ == "__main__":
    main()
