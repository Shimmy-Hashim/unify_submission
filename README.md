# Unify

## 1. What Unify is

Unify is a mobile app and backend that pulls a user's banking, investments and insurance into one place and produces a three-domain financial health score. Each domain (Banking, Investment, Insurance) is placed into a tier by an unsupervised clustering model trained on a synthetic population, and a set of rule-based mismatch flags highlights combinations that look risky (for example, high net worth with weak protection). Banking and investment data come from Plaid (Sandbox only), insurance policies are read from uploaded PDFs by a Gemini-based parser, and results are stored in Firebase.

Project context: NTU final year project CCDS25-1170.


## 2. Repo layout

- `lib/` Flutter app source (screens, services, models, utils).
- `test/` Flutter widget and unit tests.
- `assets/` App images and icons.
- `android/`, `ios/` Flutter platform folders (Firebase config files are not included, see section 4).
- `backend/` FastAPI service: Firebase auth, Plaid sync, insurance parser endpoint, score pipeline, notifications.
  - `backend/ml/` Tier assignment, Layer 3 flag rules, overall score, and the exported model artifacts in `backend/ml/artifacts/`.
  - `backend/insurance_parser/` Gemini PDF extraction and its Pydantic schema.
  - `backend/data/` Plaid, financial profile, deletion and score-profile helpers.
  - `backend/notifications/` Notification rules and the premium reminder job.
  - `backend/scripts/` `verify_*.py` verification scripts.
- `MLAI/` Research code and results.
  - `MLAI/clustering/` Synthetic data generator, benchmarks, selection, domain-split clustering, Layer 3 and score validation, plotting, and saved results.
  - `MLAI/current_snapshot/` Curated copy of the result files and plots behind the report numbers.
  - `MLAI/insurance_eval/` Parser evaluation harness, gold set labels and saved results (no PDFs).
- `firestore.rules`, `firebase.json` Firestore security rules and Firebase tool config.
- `pubspec.yaml`, `pubspec.lock`, `analysis_options.yaml` Flutter project files.

## 3. Running the backend locally

Developed with Flutter 3.41.6, Dart 3.11.4 and Python 3.14.2 (versions reported by the development machine). The Dockerfile builds the backend image from `python:3.11-slim`. No other versions have been tested.

The deployed backend runs on Python 3.11 (see backend/Dockerfile).

```
cd backend
python -m venv venv
venv\Scripts\activate          # Windows; use source venv/bin/activate on macOS or Linux
pip install -r requirements.txt
```

Environment variables are listed in `backend/.env.example`. Copy it to `backend/.env` and fill in your own values:

- `PLAID_CLIENT_ID`, `PLAID_SECRET`, `PLAID_ENV`: your own Plaid Sandbox keys from the Plaid dashboard. `PLAID_ENV` accepts `sandbox` (the default) or `production`. Plaid is read lazily, so the API starts without it and Plaid endpoints return an error until it is set.
- `GOOGLE_APPLICATION_CREDENTIALS`: path to your own Firebase service account key JSON (Firebase console, Project settings, Service accounts, Generate new private key). A relative path is resolved against the `backend/` folder. Never commit this file. The server calls Firebase init at startup, so it will not start without it.
- `GEMINI_API_KEY`: used only by the insurance parser (`POST /insurance/parse`). It is read lazily, so the API and `/health` work without it and the parser endpoint returns 503 until it is set.

Start the server from the `backend/` folder (command taken from the Dockerfile; `main.py` defines `app`):

```
uvicorn main:app --host 0.0.0.0 --port 8080
```

## 4. Running the Flutter app

Dart SDK constraint in `pubspec.yaml`: `^3.11.4`. Developed with Flutter 3.41.6 and Dart 3.11.4. The Flutter release is not pinned in the repo itself.

1. Create your own Firebase project and enable Authentication (Email/Password and Google sign-in, since the code uses `google_sign_in`) and Cloud Firestore.
2. Register an Android app with package name `com.unify.unify` (from `android/app/src/main/kotlin/com/unify/unify/MainActivity.kt`). Check `android/app/build.gradle.kts` if you change it.
3. Config files that are not in this repo:
   - `android/app/google-services.json`: download from your Firebase project. `android/app/google-services.json.example` shows the structure with every value replaced by `YOUR_VALUE_HERE`.
   - `lib/firebase_options.dart`: generate it with the FlutterFire CLI, or copy `lib/firebase_options.dart.example` and fill in the values. `lib/main.dart` imports it. The existing file only configures Android.
4. Deploy `firestore.rules` to your project (for example with the Firebase CLI).
5. Backend URL: set `kBackendUrl` in `lib/utils/constants.dart`. It currently holds the placeholder `YOUR_BACKEND_URL`. For a local backend on the Android emulator, the host machine is reachable at `http://10.0.2.2:8080` (standard Android emulator behaviour, UNVERIFIED in this repo since the app shipped against a deployed URL).
6. Run:

```
flutter pub get
flutter run
```

## 5. Reproducing the ML results and plots

All scripts live in `MLAI/clustering/` and resolve paths from the repo root, so keep the folder layout as is and run them from the repo root. Install `MLAI/requirements.txt` in the same environment as `backend/requirements.txt` (the insurance evaluation imports the backend parser). Clustering scripts need no credentials or network.

Warning: most scripts write into `MLAI/clustering/results/`, which already holds the shipped results. Running a script overwrites those files. Work on a copy of the folder, or use `run_correlation_retest.py --out-dir <name>`, which redirects the benchmark, selection and context chain into `results/<name>/`.

Input for everything: `MLAI/clustering/data/processed/synthetic_users.csv` (7,000 synthetic users, copied as is). `generate_synthetic_data.py` produces it, but do not regenerate it if you want to match the shipped numbers.

Script order, read from each script's inputs and outputs:

1. `generate_synthetic_data.py`: writes `data/processed/synthetic_users.csv` and `generator_metadata.json`.
2. `comprehensive_benchmark_11algo.py`: 12 algorithm benchmark, writes `results/comprehensive_results.json`, `results/algorithm_metrics.csv`, `results/plots/k_comparison.png`, `results/plots/algorithm_heatmap.png`. Alternatively run `run_correlation_retest.py --out-dir <name>` which chains benchmark, `final_selection.py` and `final_selection_context.py`.
3. `ablation_binary_features.py`: writes `results/ablation_binary_features.json`.
4. `final_selection.py` (reads `results/comprehensive_results.json`), then `final_selection_context.py`: write `results/final_selection.json`, `final_selection_grid.csv`, `final_selection_context.json`.
5. `domain_split_clustering.py`: three independent domain clusterings, writes `results/domain_split/domain_split_results.json`, `domain_split_grid.csv` and tier assignment CSVs. Then the follow-ups, which read its results: `domain_split_followup.py`, `investment_recalibration.py`, `investment_degeneracy_probe.py`, `risk_age_diagnostic.py`, `risk_age_followup.py`, `risk_age_final.py`, `adjacent_tier_distinctness.py`, `net_worth_removal_comparison.py`, `investment_shipped_tier_assignments.py`.
6. `build_production_artifacts.py` (use `--check-only` to verify without writing) and `verify_production_artifacts.py`: the exported artifacts live in `backend/ml/artifacts/domain_split/`.
7. Layer 3 and score validation, each writing its own folder under `results/`: `layer3_validation.py`, `layer3_coverage_audit.py`, `layer3_rule4_tightening.py`, `layer3_rule5_tightening.py`, `overall_score_validation.py`, `gated_score_validation.py`, `continuous_score_investigation.py`. They read the artifacts in `backend/ml/artifacts/domain_split/` and `synthetic_users.csv`.
8. Plots: `generate_plots.py`, `plot_report_figures.py`, `plot_decision_figures.py` (see section 6).

Not reproducible from this repo as shipped:

- `hopkins_current_data.py` reads `data/processed/synthetic_users_pre_ablation.csv`, which was left out. Its `results/hopkins_2026-10-10.json` output is included.
- `preprocess.py` (the older SCF banking-only phase) reads `data/raw/scf2022.csv`, which was left out because of its size. It is the Federal Reserve Survey of Consumer Finances 2022 summary extract (https://www.federalreserve.gov/econres/scfindex.htm). `download_scf.py` downloads it from the URL set in that script (`https://www.federalreserve.gov/econres/files/scfp2022excel.zip`) and writes it to `MLAI/clustering/data/raw/scf2022.csv`.
- The `.pkl` label caches written by the clustering scripts were not copied. The scripts recompute them on the first run, which takes longer.
- Pickled scalers: the `.pkl` files in `backend/ml/artifacts/` and `MLAI/clustering/data/processed/` are kept because scripts load them (`verify_tier_assignment.py`, `verify_production_artifacts.py`, `layer3_validation.py`, `k3_rerun.py`, `validate_clusters.py`, `test_users.py`). The backend API itself reads the JSON artifacts, not the pickles. Only unpickle files you trust.
- `benchmark_core.py --export-centroids` refuses to run without `--force` because it overwrites a historical artifact. `fit_production_scaler.py` and `k3_rerun.py` belong to the older single-model design and are kept for the record.

## 6. Reproducing the report figures

| Report figure or table (output file name) | Script | Input | Output file |
|---|---|---|---|
| ablation_silhouette_by_k.png | `MLAI/clustering/plot_report_figures.py` | `results/ablation_rerun_2026-09-27.json` | `MLAI/clustering/results/report_figures_2026-09-27/ablation_silhouette_by_k.png` |
| step5_dual_bar_scatter.png | `MLAI/clustering/plot_report_figures.py` | `results/rerun_2026-09-27/final_selection.json` | `MLAI/clustering/results/report_figures_2026-09-27/step5_dual_bar_scatter.png` |
| 01_binary_variance_share.png | `MLAI/clustering/plot_decision_figures.py` | `results/ablation_rerun_2026-09-27.json` | `MLAI/clustering/results/decision_figures_2026-09-28/01_binary_variance_share.png` |
| 02_size_floor_by_algorithm.png, 03_survivors_macro_silhouette.png, 04_leader_validation_birch.png, 05_leader_split_drivers_birch.png | `MLAI/clustering/plot_decision_figures.py` | `results/rerun_2026-09-27/final_selection.json`, `final_selection_grid.csv` | `MLAI/clustering/results/decision_figures_2026-09-28/` (same file names) |
| 06_net_worth_with_without.png, 07_banking_gate_passers.png, 08_insurance_gate_passers.png, 09_investment_gate_passers.png | `MLAI/clustering/plot_decision_figures.py` | `results/domain_split/domain_split_results.json`, `domain_split_grid.csv`, `risk_age_final.json` | `MLAI/clustering/results/decision_figures_2026-09-28/` (same file names) |
| 10_banking_why_two_tiers.png, 11_investment_third_tier_membership.png, 12_insurance_tier1_vs_tier2.png, 13_archetype_placement_domain_split.png | `MLAI/clustering/plot_decision_figures.py` | `results/domain_split/adjacent_tier_distinctness.json`, `domain_split_tier_assignments.csv`, `synthetic_users.csv`, `results/layer3_coverage_audit/layer3_coverage_audit.json` | `MLAI/clustering/results/decision_figures_2026-09-28/` (same file names) |
| correlation_heatmap.png, feature_distributions.png, dendrogram.png, k_selection.png, silhouette_approachA.png, silhouette_approachB.png | `MLAI/clustering/generate_plots.py` | `synthetic_users.csv`, `results/comprehensive_results.json`, `results/algorithm_metrics.csv` | `MLAI/clustering/results/plots/` (the `plots_rerun_2026-09-27/` copies are the redirected rerun) |
| k_comparison.png, algorithm_heatmap.png | `MLAI/clustering/comprehensive_benchmark_11algo.py` | `synthetic_users.csv` | `MLAI/clustering/results/plots/` (rerun copies in `results/rerun_2026-09-27/plots/`) |
| layer3 plots (layer3_rule_base_rates.png, nw_stable_notcovered_*.png, coverage_by_archetype.png, rule4_candidates.png, rule5_candidates.png) | `layer3_validation.py`, `layer3_coverage_audit.py`, `layer3_rule4_tightening.py`, `layer3_rule5_tightening.py` | `synthetic_users.csv`, `backend/ml/artifacts/domain_split/` | `MLAI/clustering/results/layer3_*/plots/` |
| candidate_score_distributions.png | `MLAI/clustering/continuous_score_investigation.py` | `synthetic_users.csv`, `backend/ml/artifacts/domain_split/` | `MLAI/clustering/results/continuous_score_investigation/candidate_score_distributions.png` |
| Parser evaluation table (evaluation_report.csv, field_summary.json) | `MLAI/insurance_eval/evaluator.py` | PDFs and gold set (PDFs withheld) | `MLAI/insurance_eval/results/` |

`MLAI/current_snapshot/` holds a curated copy of these plots and result files.

## 7. Insurance parser and gold set

The parser (`backend/insurance_parser/parser.py`) sends the raw PDF bytes to Gemini (default model `gemini-3.5-flash`) using structured JSON output driven by the Pydantic schema in `schema.py`, then validates the response. If validation fails, the error is fed back to the model for up to two retries. The FastAPI endpoint `POST /insurance/parse` wraps it and caps uploads at 20 MB. The endpoint is extraction-only.

Evaluation: `MLAI/insurance_eval/evaluator.py` has two tiers. Tier 1 is deterministic scoring of the structured fields (precision, recall, F1, document accuracy, fabrication rate on absent fields). Tier 2 is a G-Eval style Gemini judge for `exclusions` and `coverage_gaps`.

```
python MLAI/insurance_eval/evaluator.py --mode zero_shot
python MLAI/insurance_eval/evaluator.py --mode score_only
```

Options: `--pdfs`, `--gold`, `--output`, `--model`. `zero_shot` extracts with Gemini then scores. `score_only` re-scores an existing `raw_extractions.json`, and the Tier 2 judge still calls Gemini. Both need `GEMINI_API_KEY`. `MLAI/insurance_eval/smoke_test.py` is a liveness check that builds its own fictional one-page PDF.

The gold set PDFs are withheld for privacy, and the published labels are generalised:

- File names are neutral (`gold_set/policy_01.json` to `policy_11.json`, with matching `filename` values `policy_01.pdf` and so on).
- Amounts are rounded to the nearest 100, including amounts that appear in the free-text notes.
- Dates are reduced to year and month (`YYYY-MM`).
- Age phrases and personal details were removed or replaced with "holder".
- `results/evaluation_report.csv` uses the same neutral file names, and `results/raw_extractions.json` is not included.

Because the evaluator compares exact values, exact-match scoring will not reproduce the report numbers from these files, and the evaluation cannot be re-run end to end without your own policy PDFs and exact-value labels. The evaluator, the generalised labels and the saved results are included to show the method and the structure. Expected layout, from the evaluator code:

- `MLAI/insurance_eval/pdfs/<name>.pdf`: the PDFs to extract from.
- `MLAI/insurance_eval/gold_set/<name>.json`: one label file per PDF, matched by the `filename` field inside the JSON, which must equal the PDF file name. Copy `gold_set/template.json` for the structure (`policy_id`, `filename`, `ground_truth`, `absent_fields`, `notes`).
- Results are written to `MLAI/insurance_eval/results/` unless `--output` is given.

## 8. Running tests

Flutter:

```
flutter test
```

Backend verification scripts (in `backend/scripts/`, run from `backend/` as in each script's docstring, for example `python scripts/verify_tier_assignment.py`):

- No Firebase, Plaid or network needed: `verify_tier_assignment.py`, `verify_layer3_flags.py`, `verify_overall_score.py`. `verify_score_pipeline.py`, `verify_score_history.py` and `verify_notifications.py` have pure sections too (`--pure-only` is documented for the first two).
- Need a live Firebase project (and, where named, Plaid Sandbox keys): `verify_firebase.py`, `verify_financial_profile.py`, `verify_home_phase2.py`, `verify_plaid.py`, `verify_plaid_investments.py`, `verify_plaid_refresh.py`, `verify_item_disconnect.py`, `verify_account_deletion.py`, `verify_policy_deletion.py`. These create throwaway users in the project named in the script, so edit the `PROJECT` constant (now `YOUR_FIREBASE_PROJECT_ID`) first.
- Need the Firestore emulator: `verify_score_history_rules.py` (start it with `npx firebase-tools emulators:start --only firestore --project <your project id>`).

## 9. Deployment note

The backend was deployed on Google Cloud Run with secrets supplied from Secret Manager. This repo does not include a deploy script. The `backend/Dockerfile` builds the image (listening on `$PORT`, default 8080), and `.dockerignore` keeps `.env` and key files out of it. A reader should deploy their own instance and set `kBackendUrl` in the app to its URL.


## 10. Limitations

- Sandbox data only: Plaid is used in Sandbox, so no real bank or brokerage data has been processed, and the investment fields in Sandbox are partly fixture data.
- The tier models are trained on a synthetic population, not real users.
- The Firebase service account key reaches the deployed backend as a credential file via Secret Manager; this repo only documents the local `.env` route.
- No CI is configured.
- No reproducible deploy script is included.
- The insurance parser evaluation covers a small gold set (11 labelled policies) and cannot be re-run without your own PDFs.
