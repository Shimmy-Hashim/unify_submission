"""
generate_synthetic_data.py — Synthetic Data Generator for Unify's Cross-Domain ML Benchmark.

Generates a synthetic dataset of Singapore users spanning three financial domains
(banking, investments, insurance) so that clustering algorithms can be benchmarked on
cross-domain financial behaviour. No real cross-domain SG dataset exists, so synthetic data
anchored to published SG benchmarks is the only viable path.

Design principle: feature distributions are anchored to published Singapore benchmark data
(OCBC FWI, MoneySense, MAS, LIA, SingStat) WHERE AVAILABLE; a minority of features have no
SG-specific source — portfolio_concentration uses general retail-investor behaviour
literature, and the equity_pct/risk_age_deviation split uses the standard age-based equity
glidepath heuristic — not SG data. See FEATURE_SOURCES below for per-feature provenance.

Cluster archetypes are only partly emergent: the ~65% "normal" population is genuinely
produced by the marginals plus a Gaussian-copula correlation structure, with no cluster
shape pre-defined. The remaining ~35% are deliberately-injected mismatch (25%) and outlier
(10%) profiles — real feature-level overrides applied after copula generation (see
MISMATCH_TYPES / OUTLIER_EXTREMES) — so those users DO carry pre-defined feature patterns;
only their `profile_type` label is withheld from the model, not the patterns themselves.
The `profile_type` column is a validation label only and must never be fed to any
downstream ML pipeline.

Run standalone:
    cd <repo root>
    python MLAI/clustering/generate_synthetic_data.py

Outputs (to MLAI/clustering/data/processed/, gitignored):
    synthetic_users.csv      — the full dataset
    generator_metadata.json  — benchmark sources, parameters, config, timestamp
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import truncnorm, norm

# --------------------------------------------------------------------------------------
# CONFIGURATION — all tunable parameters live here
# --------------------------------------------------------------------------------------
CONFIG = {
    'n_users': 7000,
    'age_band_distribution': 'equal',   # 'equal' or 'population_weighted'
    'normal_pct': 0.65,
    'mismatch_pct': 0.25,               # split evenly across 6 mismatch types
    'outlier_pct': 0.10,
    'correlation_strength': 'moderate',  # 'loose', 'moderate', or 'tight'
    'random_seed': 42,
    'output_dir': 'MLAI/clustering/data/processed',
}

# SingStat 2023 working-age estimates — used only when age_band_distribution == 'population_weighted'
POPULATION_WEIGHTS = {
    1: 0.22,  # 20–29
    2: 0.24,  # 30–39
    3: 0.22,  # 40–49
    4: 0.19,  # 50–59
    5: 0.13,  # 60–70
}

AGE_BANDS = [1, 2, 3, 4, 5]
AGE_MIDPOINTS = {1: 25, 2: 35, 3: 45, 4: 55, 5: 65}

# Age-appropriate equity targets by band midpoint (used for risk_age_deviation).
# Convention: "100 minus age" percent in equities — a widely-used personal-finance
# heuristic with no single traceable origin (it emerged through financial-planning
# practice and media in the 1980s-90s), not credited to any specific study or
# institution, and with NO Singapore-specific basis. Standardized July 2026:
# backend/ml/scoring.py's RISK_AGE_EQUITY_BASE previously used a "110 minus age"
# variant while this generator used 100 — training data and live scoring must share
# one convention, so both now use 100 (scoring.py asserts its side too).
EQUITY_TARGETS = {1: 0.75, 2: 0.65, 3: 0.55, 4: 0.45, 5: 0.35}
# Machine check: targets must equal (100 - band midpoint)/100 so the generator and
# scoring.py can never silently diverge on the base.
assert all(abs(EQUITY_TARGETS[b] - (100 - AGE_MIDPOINTS[b]) / 100) < 1e-9
           for b in AGE_BANDS), "EQUITY_TARGETS must follow the 100-minus-age rule"

# --------------------------------------------------------------------------------------
# MARGINAL DISTRIBUTIONS — per feature per age band: (mean, std, min, max)
# Continuous features sampled via truncated normal. Order here defines copula index 0..10.
# healthy_debt_ratio and insurance_spend_ratio added July 2026 (binary-feature ablation
# study) — see FEATURE_SOURCES for their citation state.
# --------------------------------------------------------------------------------------
CONTINUOUS_FEATURES = [
    'savings_rate',
    'emergency_fund_months',
    'dti',
    'net_worth_to_income',
    'investment_rate',
    'portfolio_concentration',
    'equity_pct',
    'life_coverage_ratio',
    'ci_coverage_ratio',
    'healthy_debt_ratio',
    'insurance_spend_ratio',
]
# Binary flags occupy the copula indices after the continuous block (latent continuous,
# thresholded against band-specific probabilities): hospitalisation_coverage, then
# income_protection_coverage (added July 2026 — required by the binary-feature ablation's
# all-candidate configuration; previously never generated).

MARGINALS = {
    'savings_rate': {
        1: (0.20, 0.08, 0.02, 0.55),
        2: (0.24, 0.08, 0.02, 0.58),
        3: (0.27, 0.09, 0.02, 0.65),
        4: (0.22, 0.09, 0.01, 0.60),
        5: (0.17, 0.10, -0.05, 0.45),
    },
    'emergency_fund_months': {
        1: (2.5, 1.5, 0.0, 9.0),
        2: (4.0, 2.0, 0.0, 14.0),
        3: (5.0, 2.5, 0.0, 20.0),
        4: (6.5, 3.0, 0.0, 28.0),
        5: (8.0, 4.0, 0.0, 36.0),
    },
    'dti': {
        1: (0.10, 0.08, 0.0, 0.40),
        2: (0.30, 0.10, 0.0, 0.55),
        3: (0.25, 0.10, 0.0, 0.55),
        4: (0.15, 0.09, 0.0, 0.45),
        5: (0.07, 0.06, 0.0, 0.30),
    },
    # net_worth_to_income — revised July 2026 (means lowered, tails widened): the
    # earlier means (0.5/3.5/7/11/14) ran meaningfully above the common personal-
    # finance convention (~1x income by 30, ~2-3x by 35, ~4-6x through the 40s-50s,
    # gently rising toward retirement). No official SG age-stratified net-worth
    # benchmark exists — confirmed absent after directly checking MAS's Financial
    # Stability Review, SingStat's Household Sector Balance Sheet Dashboard, and
    # MOF's Feb 2026 Occasional Paper on wealth inequality (all three break wealth
    # down by wealth quintile only, never by age). Max bounds sit ~5 std above the
    # mean so genuinely wealthy users can reach high multiples (teens+) without
    # dragging the central tendency up; the injected wealthy_exposed (uniform 10-35)
    # and outlier (fixed 50 / -2) profiles bypass these marginals entirely.
    'net_worth_to_income': {
        1: (1.0, 1.2, -1.5, 7.0),
        2: (2.5, 2.0, -0.5, 12.5),
        3: (5.0, 3.0, 0.0, 20.0),
        4: (7.0, 4.0, 0.0, 27.0),
        5: (9.0, 5.0, 0.0, 34.0),
    },
    # investment_rate — INVESTOR-CONDITIONAL marginals (hurdle/mixture model, July
    # 2026): these describe only the ~88% of users who invest at all (INVESTOR_PROB
    # below, OCBC FWI 2024 participation); the remaining ~12% get investment_rate
    # exactly 0.0. Median ~10% of income around the official MoneySense/MAS Basic
    # Financial Planning Guide MINIMUM ("invest at least 10% of income" — the Guide
    # states no upper bound; see FEATURE_SOURCES for the corrected citation).
    # Central mass ~5-15% with tails beyond permitted; the mild life-cycle gradient
    # (peak-earning bands slightly higher) is reasoned, unsourced.
    'investment_rate': {
        1: (0.09, 0.035, 0.0, 0.25),
        2: (0.10, 0.035, 0.0, 0.28),
        3: (0.11, 0.040, 0.0, 0.30),
        4: (0.10, 0.040, 0.0, 0.28),
        5: (0.08, 0.040, 0.0, 0.25),
    },
    'portfolio_concentration': {
        1: (0.47, 0.15, 0.10, 0.95),
        2: (0.38, 0.14, 0.10, 0.90),
        3: (0.32, 0.13, 0.08, 0.88),
        4: (0.28, 0.12, 0.05, 0.85),
        5: (0.25, 0.12, 0.05, 0.80),
    },
    'equity_pct': {
        1: (0.62, 0.20, 0.05, 1.00),
        2: (0.55, 0.18, 0.05, 1.00),
        3: (0.48, 0.17, 0.05, 0.95),
        4: (0.42, 0.17, 0.05, 0.90),
        5: (0.35, 0.18, 0.05, 0.85),
    },
    'life_coverage_ratio': {
        1: (2.5, 2.0, 0.0, 12.0),
        2: (5.5, 3.0, 0.0, 16.0),
        3: (7.0, 3.5, 0.0, 20.0),
        4: (5.0, 3.0, 0.0, 16.0),
        5: (2.5, 2.0, 0.0, 10.0),
    },
    'ci_coverage_ratio': {
        1: (1.0, 1.0, 0.0, 6.0),
        2: (1.8, 1.2, 0.0, 7.0),
        3: (2.2, 1.3, 0.0, 8.0),
        4: (1.8, 1.2, 0.0, 7.0),
        5: (1.2, 1.0, 0.0, 5.0),
    },
    # Proportion of a user's debt that is structured/"healthy" (mortgage, student, auto)
    # vs unsecured (credit card, personal loan, line of credit). Aggregate anchor only —
    # corrected July 2026 after manual verification (an earlier comment cited an
    # unverified "~75% of household debt, MAS FSR 2024" figure): SingStat's Household
    # Sector Balance Sheet Dashboard states "Mortgage loans accounted for at least 70
    # per cent of household liabilities since 1999 Q1"; the live 2026 Q1 figure is
    # 71.4%. That share is dollar-weighted (mortgages are individually large) and
    # mortgage-only (student/auto loans add a few points to the aggregate "healthy"
    # share), so the band-weighted person-level mean of these marginals (~0.67) sits
    # deliberately below it. The age gradient (unsecured-heavy when young,
    # mortgage-dominated mid-life, unsecured-heavy again post-mortgage) is a reasoned
    # estimate — no SG debt-composition-by-age breakdown is published. Band 3's 0.78
    # sits ABOVE the all-ages aggregate by design, reviewed and retained July 2026:
    # 40-49 is peak mortgage-holding, and the all-ages aggregate is dragged down by age
    # groups outside their mortgage years — the below-aggregate reasoning applies to the
    # population-level mean, not to each band individually. Simplification: generated
    # for every user, including ones who would be debt-free in production (where this
    # feature is None, not a number).
    'healthy_debt_ratio': {
        1: (0.55, 0.20, 0.0, 1.0),
        2: (0.75, 0.15, 0.0, 1.0),
        3: (0.78, 0.15, 0.0, 1.0),
        4: (0.70, 0.18, 0.0, 1.0),
        5: (0.55, 0.22, 0.0, 1.0),
    },
    # Total annual insurance premiums / annual income. Anchor is a guideline CEILING,
    # not an observed average: MoneySense/MAS Basic Financial Planning Guide (2023)
    # recommends protection premiums <= 15% of take-home pay (for term cover). No
    # official SG average-spend-vs-income statistic was found, so the typical-spend
    # means below (4-7.5%, rising with age as premiums do) are reasoned estimates
    # under that cap — same disclosure standard as portfolio_concentration.
    'insurance_spend_ratio': {
        1: (0.040, 0.025, 0.0, 0.15),
        2: (0.060, 0.030, 0.0, 0.18),
        3: (0.070, 0.030, 0.0, 0.20),
        4: (0.075, 0.035, 0.0, 0.20),
        5: (0.060, 0.035, 0.0, 0.18),
    },
}

# investment_rate hurdle: probability a user invests at all. OCBC FWI 2024's 88%
# participation figure, applied flat across bands (no per-band participation
# breakdown was sourced). Non-investors draw investment_rate = 0.0 exactly;
# has_investments reads from this mixture draw, not from any threshold.
INVESTOR_PROB = 0.88

# hospitalisation_coverage — band-specific probability of holding an Integrated Shield Plan
HOSP_PROB = {1: 0.73, 2: 0.76, 3: 0.73, 4: 0.65, 5: 0.55}

# income_protection_coverage — band-specific probability of holding disability income /
# income protection cover. Anchor (attribution corrected July 2026 — an earlier comment
# here misattributed this to "Singlife 2023"): the AIA Disability Income Survey 2022
# (April 2022, 1,000+ full-time working SG residents aged 18+) reported 53% of SG
# workers have NO disability income protection (~47% covered) — a single-insurer,
# self-reported survey, i.e. a weak source, disclosed as such. Band gradient reasoned:
# take-up is employment-linked and standalone DI is rarely bought young; DI policies
# typically cease around 65, so band 5 drops sharply.
IP_PROB = {1: 0.35, 2: 0.48, 3: 0.50, 4: 0.45, 5: 0.20}

# Citation strings recorded in metadata
FEATURE_SOURCES = {
    'savings_rate': 'OCBC FWI 2024 (avg 25% nationally)',
    'emergency_fund_months': 'MoneySense 6-month benchmark; OCBC FWI 2024 (53% of SG adults lack 6-month fund)',
    'dti': ('MAS TDSR regulatory cap 55% (mas.gov.sg/regulation/explainers/tdsr-for-property-loans, confirmed); '
            'DBS "7 financial ratios to gauge your financial health": "Ideally, a ratio of 35% or below means you '
            'have ample income to fulfil your monthly debt repayments" — a recommended ideal, not a compliance '
            'ceiling. Band means peak in the 30s (mortgage years), reasoned. An earlier "mortgage loans ~82% of '
            'annual PDI" claim was removed July 2026 — confirmed absent from MAS FSR 2025\'s actual data'),
    'net_worth_to_income': ('Widely-cited personal-finance heuristic (~1x income by 30, rising with age) with NO '
                            'official SG source — confirmed absent after directly checking MAS FSR, SingStat\'s '
                            'Household Sector Balance Sheet Dashboard, and MOF\'s Feb 2026 Occasional Paper on '
                            'wealth inequality (all break wealth down by quintile only, never by age). Earlier '
                            'project values ran meaningfully higher than this convention; revised downward July '
                            '2026 (see commit for before/after)'),
    'investment_rate': ('MoneySense/MAS Basic Financial Planning Guide: "invest at least 10% of income" — an '
                        'official MINIMUM, confirmed across three official documents (general, Fresh Entrant, and '
                        'Streamlined guides); NO official upper bound exists — an earlier "10-15%" citation '
                        'conflated this floor with the same Guide\'s separate 15% INSURANCE-spend ceiling, a '
                        'different category entirely. Upper-range values used in this project are independently-'
                        'chosen round numbers, not sourced. Mixture: OCBC FWI 2024 88% participation (INVESTOR_PROB)'),
    'has_investments': ('OCBC FWI 2024: 88% of SG adults hold investments — implemented as a hurdle/mixture '
                        '(INVESTOR_PROB = 0.88 flat; non-investors have investment_rate exactly 0.0). The flag '
                        'reads from the mixture draw, reconciled with mismatch/outlier overrides — not derived '
                        'via a threshold on investment_rate'),
    'portfolio_concentration': 'Derived from retail investor behaviour literature; no direct SG data',
    'equity_pct': 'Standard (100−age) equity rule; OCBC FWI 2024; wide variance reflects SG retail deviation',
    'life_coverage_ratio': 'LIA PGS 2022 (21% mortality gap, 79% of needs met; recommends 9–10×; avg ~7.9×)',
    'ci_coverage_ratio': ('LIA PGS 2022 (74% CI gap; avg CI coverage ~SGD 193,300 ≈ 2.1× income; recommends ~3.9×). '
                          'Precision caveat: MoneySense\'s public Guide rounds this to 4x; the more precise 3.9x '
                          'traces to the underlying LIA Protection Gap Study 2022, which corroborates but was not '
                          'independently re-verified at this exact decimal precision'),
    'hospitalisation_coverage': '~71% of SG residents have Integrated Shield Plans (Prudential/MOH 2023); declines in older bands',
    'healthy_debt_ratio': ('SingStat Household Sector Balance Sheet Dashboard: "Mortgage loans accounted for at least '
                           '70 per cent of household liabilities since 1999 Q1" (2026 Q1: 71.4%) — aggregate, '
                           'dollar-weighted, mortgage-only; person-level age gradient is a reasoned estimate, '
                           'no SG debt-composition-by-age data published'),
    'insurance_spend_ratio': ('MoneySense/MAS Basic Financial Planning Guide 2023 (premiums <=15% of take-home pay '
                              '— a guideline MAXIMUM for term cover, not an average); typical-spend means are reasoned '
                              'estimates under that cap, no official SG average-spend statistic found'),
    'income_protection_coverage': ('AIA Disability Income Survey 2022 (April 2022, 1,000+ full-time working SG residents): '
                                   '53% of SG workers lack disability income protection (~47% covered) '
                                   '— single-insurer self-reported survey, weak source, disclosed; band gradient reasoned '
                                   '(employment-linked take-up, DI cover typically ceases ~65)'),
}

# --------------------------------------------------------------------------------------
# CORRELATION STRUCTURE — Gaussian copula
# Index order: 0 savings_rate, 1 emergency_fund, 2 dti, 3 net_worth, 4 investment_rate,
#              5 portfolio_conc, 6 equity_pct, 7 life_cov, 8 ci_cov,
#              9 healthy_debt_ratio, 10 insurance_spend_ratio, 11 hosp, 12 income_prot
# The original 10-variable block (sr..cc, hosp) is unchanged; rows/cols for the two new
# continuous features and the income-protection latent were inserted July 2026. New
# correlations are reasoned, moderate-scale values in the same style as the original
# matrix (e.g. insurance_spend correlates with the coverage features it funds).
#
# ======================================================================================
# ACCUMULATION <-> PROTECTION CROSS-DOMAIN CORRELATION — REVISED July 2026
# ======================================================================================
# WHY: the gate-based algorithm-selection task concluded that accumulation
# (savings_rate, investment_rate, net_worth_to_income) and protection
# (life_coverage_ratio, ci_coverage_ratio, insurance_spend_ratio) behave as two
# largely INDEPENDENT axes, which blocked any single ordered health tiering. A
# follow-up literature check asked whether that independence is real or an artifact of
# this matrix: the previous cross-domain block averaged only ~0.18, which understates
# the empirical wealth->insurance relationship.
#
# CITATION BASIS (both US proxies — NO Singapore-specific coefficient exists; this is a
# disclosed limitation, not a verified SG figure):
#   * Armantier, Foncel & Treich (2023), Journal of Financial Economics — standardized
#     wealth -> insurance-demand beta of ~0.37.
#   * Gropper & Kuhnen (2025), Journal of Finance — implies ~0.28 on the same
#     relationship.
# Target band therefore ~0.25-0.35. The block below averages 0.29, sitting between the
# two cited estimates rather than at either endpoint.
#
# WHY THE VALUES ARE NOT FLAT 0.30 (per-pair judgement, deliberate):
#   * net_worth_to_income is a STOCK-of-wealth measure and therefore the closest proxy
#     to the "wealth" term in the cited betas -> highest values (0.30-0.35).
#   * savings_rate is a FLOW measure, a weaker proxy for accumulated wealth -> mid
#     (0.27-0.30).
#   * investment_rate gets the LOWEST values (0.25-0.28): investing and insuring compete
#     for the same discretionary dollar, so a substitution effect partly offsets the
#     income effect that drives the positive correlation.
#   * insurance_spend_ratio is the weakest link within each row: it is a
#     premium-affordability measure confounded by product mix (term vs. whole-life), so
#     it tracks accumulation less tightly than sum-assured coverage does.
#
# NOT CHANGED, and why: savings_rate <-> investment_rate (the within-accumulation block)
# was flagged as possibly needing a bump to ~0.4 — it is ALREADY 0.40 (row 0, col 4).
# No change made; the secondary recommendation was already satisfied.
#
# ⚠️ IMPORTANT CONFOUND, do not read this fix as making the two domains correlated
# population-wide: the `protection_first` and `wealthy_exposed` mismatch profiles below
# are applied as HARD OVERRIDES AFTER copula generation and are deliberately
# accumulation/protection-INVERTED by construction (wealthy_exposed = high net worth +
# zero coverage; protection_first = high coverage + low savings/net worth). Those two
# types are ~8% of the population (2 of 6 types across the 25% mismatch share) and are
# untouched by this matrix. This fix raises the correlation among the 65% normal-profile
# users only; the injected anti-correlated archetypes remain by design, since they are
# the stated justification for having a Layer 2 ML component at all.
# --------------------------------------------------------------------------------------
CORR_MODERATE = np.array([
    #   sr     ef    dti    nw     ir     pc     eq     lc     cc    hdr    isr   hosp    ip
    [1.00,  0.50, -0.45,  0.45,  0.40, -0.20,  0.10,  0.30,  0.28,  0.15,  0.27,  0.15,  0.10],
    [0.50,  1.00, -0.35,  0.50,  0.30, -0.15,  0.05,  0.15,  0.12,  0.12,  0.08,  0.12,  0.08],
    [-0.45, -0.35,  1.00, -0.40, -0.25,  0.10, -0.05, -0.15, -0.10,  0.05, -0.08, -0.12, -0.08],
    [0.45,  0.50, -0.40,  1.00,  0.45, -0.25,  0.15,  0.35,  0.32,  0.30,  0.30,  0.18,  0.12],
    [0.40,  0.30, -0.25,  0.45,  1.00, -0.30,  0.20,  0.28,  0.26,  0.10,  0.25,  0.12,  0.08],
    [-0.20, -0.15,  0.10, -0.25, -0.30,  1.00,  0.10, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05],
    [0.10,  0.05, -0.05,  0.15,  0.20,  0.10,  1.00,  0.10,  0.08,  0.05,  0.05,  0.05,  0.03],
    [0.30,  0.15, -0.15,  0.35,  0.28, -0.05,  0.10,  1.00,  0.60,  0.15,  0.55,  0.22,  0.30],
    [0.28,  0.12, -0.10,  0.32,  0.26, -0.05,  0.08,  0.60,  1.00,  0.12,  0.50,  0.20,  0.25],
    [0.15,  0.12,  0.05,  0.30,  0.10, -0.05,  0.05,  0.15,  0.12,  1.00,  0.10,  0.10,  0.08],
    [0.27,  0.08, -0.08,  0.30,  0.25, -0.05,  0.05,  0.55,  0.50,  0.10,  1.00,  0.30,  0.35],
    [0.15,  0.12, -0.12,  0.18,  0.12, -0.05,  0.05,  0.22,  0.20,  0.10,  0.30,  1.00,  0.25],
    [0.10,  0.08, -0.08,  0.12,  0.08, -0.05,  0.03,  0.30,  0.25,  0.08,  0.35,  0.25,  1.00],
])

# The 9 accumulation x protection pairs revised above, kept machine-checkable so the
# intended block cannot silently drift (asserted in main()).
ACCUMULATION_IDX = {'savings_rate': 0, 'net_worth_to_income': 3, 'investment_rate': 4}
PROTECTION_IDX = {'life_coverage_ratio': 7, 'ci_coverage_ratio': 8,
                  'insurance_spend_ratio': 10}
CROSS_DOMAIN_TARGET_BAND = (0.25, 0.35)
CROSS_DOMAIN_PREVIOUS_MEAN = 0.182   # the pre-fix block mean, for the record

# Copula index order (all 13 latent variables). Binary flags are latent-continuous in the
# copula and thresholded to 0/1 against their band-specific probabilities.
BINARY_FLAGS = ['hospitalisation_coverage', 'income_protection_coverage']
BINARY_PROBS = {'hospitalisation_coverage': HOSP_PROB, 'income_protection_coverage': IP_PROB}
COPULA_FEATURES = CONTINUOUS_FEATURES + BINARY_FLAGS

# --------------------------------------------------------------------------------------
# SIX MISMATCH PROFILE TYPES — overrides applied AFTER copula generation.
# Each entry maps feature -> ('uniform', low, high) or ('fixed', value).
# --------------------------------------------------------------------------------------
MISMATCH_TYPES = {
    'conservative_saver': {
        'savings_rate': ('uniform', 0.28, 0.55),
        'emergency_fund_months': ('uniform', 5.0, 18.0),
        'investment_rate': ('uniform', 0.00, 0.03),
    },
    'over_invested': {
        'investment_rate': ('uniform', 0.22, 0.40),
        'emergency_fund_months': ('uniform', 0.0, 1.5),
        'savings_rate': ('uniform', 0.02, 0.10),
    },
    'protection_first': {
        'life_coverage_ratio': ('uniform', 8.0, 18.0),
        'ci_coverage_ratio': ('uniform', 3.0, 7.0),
        'hospitalisation_coverage': ('fixed', 1),
        'income_protection_coverage': ('fixed', 1),
        'insurance_spend_ratio': ('uniform', 0.08, 0.16),
        'savings_rate': ('uniform', 0.03, 0.12),
        'net_worth_to_income': ('uniform', 0.0, 2.5),
    },
    'wealthy_exposed': {
        'net_worth_to_income': ('uniform', 10.0, 35.0),
        'savings_rate': ('uniform', 0.30, 0.60),
        'life_coverage_ratio': ('uniform', 0.0, 1.0),
        'ci_coverage_ratio': ('uniform', 0.0, 0.5),
        'hospitalisation_coverage': ('fixed', 0),
        'income_protection_coverage': ('fixed', 0),
        'insurance_spend_ratio': ('uniform', 0.0, 0.01),
    },
    'high_debt_investor': {
        'investment_rate': ('uniform', 0.18, 0.38),
        'dti': ('uniform', 0.40, 0.55),
        'savings_rate': ('uniform', 0.01, 0.08),
        'emergency_fund_months': ('uniform', 0.0, 2.0),
        'healthy_debt_ratio': ('uniform', 0.0, 0.35),
    },
    'underinsured_saver': {
        'savings_rate': ('uniform', 0.25, 0.55),
        'emergency_fund_months': ('uniform', 4.0, 15.0),
        'life_coverage_ratio': ('uniform', 0.0, 1.5),
        'ci_coverage_ratio': ('uniform', 0.0, 0.8),
        'hospitalisation_coverage': ('fixed', 0),
        'income_protection_coverage': ('fixed', 0),
        'insurance_spend_ratio': ('uniform', 0.0, 0.015),
    },
}
MISMATCH_ORDER = list(MISMATCH_TYPES.keys())

# --------------------------------------------------------------------------------------
# OUTLIER PROFILES — push 1 or 2 randomly-chosen features to an extreme (high or low).
# --------------------------------------------------------------------------------------
OUTLIER_EXTREMES = {
    'savings_rate': {'high': 0.68, 'low': 0.005},
    'emergency_fund_months': {'high': 40.0, 'low': 0.0},
    'dti': {'high': 0.58, 'low': 0.0},
    'net_worth_to_income': {'high': 50.0, 'low': -2.0},
    'investment_rate': {'high': 0.50, 'low': 0.0},
    'portfolio_concentration': {'high': 0.97, 'low': 0.05},
    'life_coverage_ratio': {'high': 28.0, 'low': 0.0},
    'ci_coverage_ratio': {'high': 14.0, 'low': 0.0},
    'healthy_debt_ratio': {'high': 1.0, 'low': 0.0},
    'insurance_spend_ratio': {'high': 0.22, 'low': 0.0},
}
OUTLIER_FEATURES = list(OUTLIER_EXTREMES.keys())

# Final CSV column order
OUTPUT_COLUMNS = [
    'user_id', 'age_band', 'age_midpoint', 'profile_type',
    'savings_rate', 'emergency_fund_months', 'dti', 'net_worth_to_income',
    'investment_rate', 'has_investments', 'portfolio_concentration', 'equity_pct',
    'risk_age_deviation', 'life_coverage_ratio', 'ci_coverage_ratio',
    'healthy_debt_ratio', 'insurance_spend_ratio', 'hospitalisation_coverage',
    'income_protection_coverage',
]
FLOAT_COLUMNS = [
    'savings_rate', 'emergency_fund_months', 'dti', 'net_worth_to_income',
    'investment_rate', 'portfolio_concentration', 'equity_pct', 'risk_age_deviation',
    'life_coverage_ratio', 'ci_coverage_ratio', 'healthy_debt_ratio',
    'insurance_spend_ratio',
]


# --------------------------------------------------------------------------------------
# Correlation helpers
# --------------------------------------------------------------------------------------
def build_correlation_matrix(strength):
    """Return the 13x13 correlation matrix for the requested strength."""
    R = CORR_MODERATE.copy()
    off_diag = ~np.eye(R.shape[0], dtype=bool)
    if strength == 'loose':
        R[off_diag] *= 0.5
    elif strength == 'tight':
        R[off_diag] *= 1.4
        R[off_diag] = np.clip(R[off_diag], -0.92, 0.92)
    elif strength != 'moderate':
        raise ValueError(f"Unknown correlation_strength: {strength!r}")
    return R


def nearest_psd(R):
    """Return (R_corrected, was_corrected, min_eigenvalue).

    The minimum eigenvalue is the PRE-correction one either way, so callers can report
    how far from PSD the input was, not just whether it needed fixing. Correction is by
    eigenvalue clipping.
    """
    eigvals = np.linalg.eigvalsh(R)
    min_eig = float(eigvals.min())
    if min_eig >= 0:
        return R, False, min_eig

    # Clip negative eigenvalues to a small positive value and reconstruct.
    vals, vecs = np.linalg.eigh(R)
    vals_clipped = np.clip(vals, 1e-6, None)
    R_fixed = vecs @ np.diag(vals_clipped) @ vecs.T
    # Re-symmetrise and rescale the diagonal back to 1 (valid correlation matrix).
    R_fixed = (R_fixed + R_fixed.T) / 2
    d = np.sqrt(np.diag(R_fixed))
    R_fixed = R_fixed / np.outer(d, d)
    return R_fixed, True, min_eig


# --------------------------------------------------------------------------------------
# Age band assignment
# --------------------------------------------------------------------------------------
def assign_age_bands(n, mode, rng):
    if mode == 'equal':
        probs = np.full(len(AGE_BANDS), 1.0 / len(AGE_BANDS))
    elif mode == 'population_weighted':
        probs = np.array([POPULATION_WEIGHTS[b] for b in AGE_BANDS], dtype=float)
        probs = probs / probs.sum()
    else:
        raise ValueError(f"Unknown age_band_distribution: {mode!r}")
    return rng.choice(AGE_BANDS, size=n, p=probs)


# --------------------------------------------------------------------------------------
# Copula generation
# --------------------------------------------------------------------------------------
def generate_copula(n, bands, R, rng):
    """Generate correlated continuous features + latent binary uniforms via Gaussian copula."""
    L = np.linalg.cholesky(R)
    Z = rng.standard_normal((n, R.shape[0])) @ L.T
    U = norm.cdf(Z)  # correlated uniforms, shape (n, 13)

    # Transform each continuous column through its band-specific truncnorm inverse CDF.
    data = {f: np.empty(n) for f in CONTINUOUS_FEATURES}
    for b in AGE_BANDS:
        mask = bands == b
        if not mask.any():
            continue
        for i, feat in enumerate(CONTINUOUS_FEATURES):
            mean, std, lo, hi = MARGINALS[feat][b]
            # truncnorm takes its bounds in standard deviations from the mean.
            z_lo, z_hi = (lo - mean) / std, (hi - mean) / std
            if feat == 'investment_rate':
                # Hurdle/mixture: the bottom (1 - INVESTOR_PROB) of the latent
                # propensity are non-investors with rate exactly 0.0; the rest are
                # investors whose latent is rescaled back to U(0,1) before the
                # inverse CDF, so the investor-conditional marginal keeps its
                # intended shape. Using the shared copula latent (rather than an
                # independent coin flip) keeps non-investing correlated with the
                # rest of the profile the same way the rate itself is.
                u = U[mask, i]
                investor = u >= (1.0 - INVESTOR_PROB)
                vals = np.zeros(u.size)
                u_inv = (u[investor] - (1.0 - INVESTOR_PROB)) / INVESTOR_PROB
                vals[investor] = truncnorm.ppf(u_inv, z_lo, z_hi, loc=mean, scale=std)
                data[feat][mask] = vals
            else:
                data[feat][mask] = truncnorm.ppf(U[mask, i], z_lo, z_hi,
                                                 loc=mean, scale=std)

    # Binary flags: threshold each latent uniform against its band probability.
    for flag in BINARY_FLAGS:
        idx = COPULA_FEATURES.index(flag)
        col = np.zeros(n, dtype=int)
        for b in AGE_BANDS:
            mask = bands == b
            col[mask] = (U[mask, idx] < BINARY_PROBS[flag][b]).astype(int)
        data[flag] = col
    return data


# --------------------------------------------------------------------------------------
# Profile assignment
# --------------------------------------------------------------------------------------
def profile_counts(n):
    """Exact profile_type counts for n users: normal, each mismatch type, then outlier.

    The single source of truth for the split — assign_profiles() builds the label array
    from this and run_validation() checks the generated data against it. Keeping two
    copies of the arithmetic would let validation drift into confirming whatever the
    generator happened to do rather than what was specified.
    """
    n_normal = int(round(CONFIG['normal_pct'] * n))
    n_outlier = int(round(CONFIG['outlier_pct'] * n))
    n_mismatch = n - n_normal - n_outlier
    # Split mismatch as evenly as possible across the 6 types; the first `rem` types in
    # MISMATCH_ORDER absorb the remainder.
    base, rem = divmod(n_mismatch, len(MISMATCH_ORDER))
    counts = {'normal': n_normal}
    for j, t in enumerate(MISMATCH_ORDER):
        counts[t] = base + (1 if j < rem else 0)
    counts['outlier'] = n_outlier
    return counts


def assign_profiles(n, rng):
    """Return an object array of profile_type labels, shuffled across all users."""
    counts = profile_counts(n)
    # ⚠️ The build order (normal, then outlier, then the mismatch types in
    # MISMATCH_ORDER) is load-bearing: rng.shuffle permutes the array as constructed, so
    # reordering these appends changes which user receives which profile for a given
    # seed — and every committed result downstream is keyed to the current assignment.
    labels = ['normal'] * counts['normal'] + ['outlier'] * counts['outlier']
    for t in MISMATCH_ORDER:
        labels += [t] * counts[t]

    labels = np.array(labels, dtype=object)
    rng.shuffle(labels)
    return labels


def apply_mismatch_overrides(data, profiles, rng):
    for t in MISMATCH_ORDER:
        idx = np.where(profiles == t)[0]
        if idx.size == 0:
            continue
        for feat, spec in MISMATCH_TYPES[t].items():
            if spec[0] == 'uniform':
                data[feat][idx] = rng.uniform(spec[1], spec[2], size=idx.size)
            else:  # 'fixed'
                data[feat][idx] = spec[1]


def apply_outlier_overrides(data, profiles, rng):
    idx = np.where(profiles == 'outlier')[0]
    for i in idx:
        k = rng.integers(1, 3)  # 1 or 2 features
        feats = rng.choice(OUTLIER_FEATURES, size=k, replace=False)
        for feat in feats:
            side = 'high' if rng.random() < 0.5 else 'low'
            data[feat][i] = OUTLIER_EXTREMES[feat][side]


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------
def assemble_dataframe(n, bands, data, profiles):
    df = pd.DataFrame()
    df['user_id'] = [f"U{i + 1:05d}" for i in range(n)]
    df['age_band'] = bands.astype(int)
    df['age_midpoint'] = df['age_band'].map(AGE_MIDPOINTS).astype(int)
    df['profile_type'] = profiles

    for feat in CONTINUOUS_FEATURES:
        df[feat] = data[feat]

    # Derived feature: deviation of equity allocation from age-appropriate target.
    targets = df['age_band'].map(EQUITY_TARGETS).to_numpy()
    df['risk_age_deviation'] = data['equity_pct'] - targets

    df['has_investments'] = data['has_investments'].astype(int)
    for flag in BINARY_FLAGS:
        df[flag] = data[flag].astype(int)

    df[FLOAT_COLUMNS] = df[FLOAT_COLUMNS].round(4)
    return df[OUTPUT_COLUMNS]


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
def run_validation(df, was_corrected, min_eig):
    n = len(df)
    print("\n" + "=" * 78)
    print("VALIDATION")
    print("=" * 78)

    # 1. Correlation matrix PSD
    print(f"\n[1] Correlation matrix PSD check: min eigenvalue = {min_eig:.2e}")
    if was_corrected:
        print("    -> Matrix was NOT PSD; nearest-PSD correction (eigenvalue clipping) applied.")
    else:
        print("    -> Matrix is PSD; no correction needed.")

    # 2. Profile type counts
    print("\n[2] Profile type counts (actual vs expected):")
    expected = profile_counts(n)
    actual = df['profile_type'].value_counts().to_dict()
    all_types = ['normal'] + MISMATCH_ORDER + ['outlier']
    print(f"    {'type':<22}{'actual':>8}{'expected':>10}{'actual %':>10}")
    for t in all_types:
        a = actual.get(t, 0)
        e = expected.get(t, 0)
        print(f"    {t:<22}{a:>8}{e:>10}{100 * a / n:>9.2f}%")

    # 3. Age band counts
    print("\n[3] Age band counts (actual):")
    band_counts = df['age_band'].value_counts().to_dict()
    for b in AGE_BANDS:
        c = band_counts.get(b, 0)
        print(f"    band {b}: {c:>6}  ({100 * c / n:5.2f}%)")

    # 4. Per-feature mean/std + >15% deviation flag (target = band-weighted expected mean)
    print("\n[4] Per-feature mean / std (flag if |actual-target| > 15% of target):")
    print(f"    {'feature':<26}{'mean':>10}{'std':>10}{'target':>10}{'dev%':>9}")
    for feat in CONTINUOUS_FEATURES:
        actual_mean = float(df[feat].mean())
        actual_std = float(df[feat].std())
        # Band-weighted expected mean using actual band membership.
        target_mean = 0.0
        for b in AGE_BANDS:
            cnt = band_counts.get(b, 0)
            target_mean += (cnt / n) * MARGINALS[feat][b][0]
        if feat == 'investment_rate':
            # Marginal is investor-conditional; population target scales by the hurdle.
            target_mean *= INVESTOR_PROB
        if abs(target_mean) > 1e-9:
            dev = 100 * (actual_mean - target_mean) / abs(target_mean)
        else:
            dev = float('nan')
        flag = "  <-- FLAG" if (abs(dev) > 15) else ""
        print(f"    {feat:<26}{actual_mean:>10.4f}{actual_std:>10.4f}"
              f"{target_mean:>10.4f}{dev:>8.1f}%{flag}")
    print("    (deviations are expected where mismatch/outlier overrides shift a feature.)")

    # 5. Null check
    print("\n[5] Null check:")
    null_total = int(df.isnull().sum().sum())
    if null_total > 0:
        raise ValueError(f"Validation FAILED: {null_total} null value(s) found in dataset.")
    print("    -> No null values found.")

    # 6. Investor share vs the hurdle target (flag if >5pp off; overrides shift it —
    #    outlier 'low' forces some rates to 0, mismatch overrides force nonzero rates)
    inv_rate = float(df['has_investments'].mean())
    diff_pp = abs(inv_rate - INVESTOR_PROB) * 100
    flag = "  <-- FLAG" if diff_pp > 5 else ""
    print(f"\n[6] has_investments share: actual={inv_rate:5.3f}  "
          f"target={INVESTOR_PROB:5.3f}  diff={diff_pp:4.1f}pp{flag}")

    # 6b. Binary flag rate per band (flag if >5pp from target)
    for flag_name in BINARY_FLAGS:
        print(f"\n[6b] {flag_name} rate per band (flag if >5pp from target):")
        for b in AGE_BANDS:
            sub = df[df['age_band'] == b]
            rate = float(sub[flag_name].mean()) if len(sub) else 0.0
            target = BINARY_PROBS[flag_name][b]
            diff_pp = abs(rate - target) * 100
            marker = "  <-- FLAG" if diff_pp > 5 else ""
            print(f"    band {b}: actual={rate:5.3f}  target={target:5.3f}  "
                  f"diff={diff_pp:4.1f}pp{marker}")

    # 7. Summary
    pct_normal = 100 * actual.get('normal', 0) / n
    pct_outlier = 100 * actual.get('outlier', 0) / n
    pct_mismatch = 100 - pct_normal - pct_outlier
    print("\n[7] Summary:")
    print(f"    Generated {n} users across {df['age_band'].nunique()} age bands. "
          f"Profile types: {pct_normal:.0f}% normal, {pct_mismatch:.0f}% mismatch, "
          f"{pct_outlier:.0f}% outlier.")


# --------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------
def build_metadata(df, strength):
    feature_summary = {}
    all_features = CONTINUOUS_FEATURES + ['has_investments'] + BINARY_FLAGS
    for feat in all_features:
        feature_summary[feat] = {
            'mean': round(float(df[feat].mean()), 4),
            'std': round(float(df[feat].std()), 4),
            'min': round(float(df[feat].min()), 4),
            'max': round(float(df[feat].max()), 4),
            'source': FEATURE_SOURCES[feat],
        }

    profile_counts = {t: int((df['profile_type'] == t).sum())
                      for t in ['normal'] + MISMATCH_ORDER + ['outlier']}
    age_band_counts = {str(b): int((df['age_band'] == b).sum()) for b in AGE_BANDS}

    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'config': CONFIG,
        'profile_type_counts': profile_counts,
        'age_band_counts': age_band_counts,
        'feature_summary': feature_summary,
        'correlation_strength_used': strength,
        'notes': (
            "Synthetic data generated for cross-domain ML benchmark. Distributions anchored "
            "to SG benchmarks. No archetypes pre-defined. profile_type column is for "
            "validation only — do not feed to clustering."
        ),
    }


# --------------------------------------------------------------------------------------
# Path resolution — resolve output_dir relative to the repo root so the script runs
# regardless of the current working directory.
# --------------------------------------------------------------------------------------
def resolve_output_dir():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    out = CONFIG['output_dir']
    return out if os.path.isabs(out) else os.path.join(repo_root, out)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    n = CONFIG['n_users']
    strength = CONFIG['correlation_strength']
    rng = np.random.default_rng(CONFIG['random_seed'])

    print("=" * 78)
    print("Unify — Synthetic Data Generator (cross-domain ML benchmark)")
    print("=" * 78)
    print(f"n_users={n}  age_band_distribution={CONFIG['age_band_distribution']}  "
          f"correlation_strength={strength}  seed={CONFIG['random_seed']}")

    # Build + verify correlation matrix BEFORE generation.
    R = build_correlation_matrix(strength)

    # Assert the July 2026 accumulation<->protection revision is present and in band.
    # Guards against a silent revert of the cross-domain block (see CORR_MODERATE).
    # Deliberately reads CORR_MODERATE, NOT the R built above: this checks that the
    # revised constants are still in the source, which is the thing that could silently
    # regress. Under the 'loose'/'tight' strengths R is a rescaled copy whose cross-domain
    # values would fall outside the cited band by design, so asserting on R would fail
    # for a legitimate configuration.
    cross = [CORR_MODERATE[ai, pi]
             for ai in ACCUMULATION_IDX.values() for pi in PROTECTION_IDX.values()]
    cross_mean = float(np.mean(cross))
    lo, hi = CROSS_DOMAIN_TARGET_BAND
    assert all(lo <= c <= hi for c in cross), (
        f"accumulation<->protection correlations outside the cited {lo}-{hi} band: {cross}")
    assert lo <= cross_mean <= hi, f"cross-domain block mean {cross_mean:.3f} outside band"
    print(f"cross-domain accumulation<->protection correlation: mean {cross_mean:.3f} "
          f"(was {CROSS_DOMAIN_PREVIOUS_MEAN:.3f}), all 9 pairs within {lo}-{hi} "
          f"[Armantier et al. 2023 ~0.37 / Gropper & Kuhnen 2025 ~0.28, US proxies]")

    R, was_corrected, min_eig = nearest_psd(R)
    psd_note = "NOT PSD, eigenvalue-clipped" if was_corrected else "PSD, no correction needed"
    print(f"correlation matrix PSD check: min eigenvalue {min_eig:.6f} -> {psd_note}")
    if was_corrected:
        # Confirm the correction actually produced a PSD matrix.
        _, still_broken, post_eig = nearest_psd(R)
        if still_broken:
            raise ValueError(
                f"Correlation matrix not fixable: post-correction min eigenvalue {post_eig:.2e}")

    # Generate.
    bands = assign_age_bands(n, CONFIG['age_band_distribution'], rng)
    data = generate_copula(n, bands, R, rng)
    profiles = assign_profiles(n, rng)
    apply_mismatch_overrides(data, profiles, rng)
    apply_outlier_overrides(data, profiles, rng)
    # has_investments reads from the hurdle draw, reconciled with overrides: under
    # the mixture, investment_rate is exactly 0.0 iff the user is a non-investor
    # (copula non-investors, plus outlier 'low' overrides that force 0.0), so the
    # flag IS the zero spike — no arbitrary threshold involved. Computed on the
    # rate rounded to the CSV's 4-dp precision so the stored flag and stored rate
    # can never disagree (an unrounded near-zero investor rate would otherwise
    # display as 0.0 while flagged 1).
    data['has_investments'] = (np.round(data['investment_rate'], 4) > 0.0).astype(int)
    df = assemble_dataframe(n, bands, data, profiles)

    # Validate.
    run_validation(df, was_corrected, min_eig)

    # Write outputs.
    out_dir = resolve_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'synthetic_users.csv')
    meta_path = os.path.join(out_dir, 'generator_metadata.json')

    df.to_csv(csv_path, index=False)
    metadata = build_metadata(df, strength)
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 78)
    print(f"Wrote {len(df)} rows -> {csv_path}")
    print(f"Wrote metadata    -> {meta_path}")
    print("Done.")
    print("=" * 78)


if __name__ == '__main__':
    main()
