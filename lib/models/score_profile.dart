import 'package:cloud_firestore/cloud_firestore.dart';

/// The Phase 3 score, as cached on users/{uid}/financialProfile/profile.
///
/// This file is deserialization ONLY — it re-derives no threshold, no tier and
/// no formula. Every value here was produced by the backend pipeline
/// (`ml/score_pipeline.py` -> `data/score_profile.py::cache_payload`) and is
/// read back verbatim. The wire keys below are that module's keys; if one is
/// renamed there, it must be renamed here.
///
/// WHY THE PARSE TAKES THE WHOLE PROFILE DOCUMENT
/// The cached score is spread across four Section 8 fields, not one:
///   overall_score           top level (duplicated inside ml_profile)
///   checklist_last_updated  top level — the "has it ever run?" signal
///   checklist_flags         layer1 checks + layer3 flags
///   ml_profile              everything else (age band, domains, breakdown)
/// The Layer 1 checklist exists ONLY in checklist_flags.layer1, and the
/// never-computed signal exists ONLY in checklist_last_updated, so a parse of
/// ml_profile alone would be missing both.

/// The three scored domains, in the order every user-facing surface shows them
/// (`score_pipeline.DOMAIN_ORDER`).
enum ScoreDomain {
  banking('banking'),
  investment('investment'),
  insurance('insurance');

  const ScoreDomain(this.key);

  /// The Firestore map key and artifact filename stem — same spelling.
  final String key;
}

/// `score_pipeline.STATUS_ASSESSED` / `STATUS_NOT_ASSESSED`.
enum DomainStatus { assessed, notAssessed }

/// Every reason code the pipeline can attach to a domain, to the overall score,
/// or to Layer 3.
///
/// Deliberately a superset of [ScoreUnavailableReason]: a whole-pipeline
/// failure stamps the SAME code onto all three domains (`_unavailable()`), so a
/// domain's reason can be any of these six. [unknown] exists so a reason code
/// added to the backend later renders as "not assessed" rather than crashing
/// the parse.
enum ScoreReason {
  incompleteDomains('incomplete_domains'),
  missingFeatures('missing_features'),
  incomeNotSet('income_not_set'),
  unsupportedAge('unsupported_age'),
  dateOfBirthNotSet('date_of_birth_not_set'),
  noFinancialProfile('no_financial_profile'),
  unknown('');

  const ScoreReason(this.code);

  final String code;

  static ScoreReason? parse(Object? value) {
    if (value == null) return null;
    for (final reason in ScoreReason.values) {
      if (reason.code == value) return reason;
    }
    return ScoreReason.unknown;
  }
}

/// The four codes that mean "no part of the score is computable", as opposed to
/// the ordinary partial case.
///
/// Kept separate from [ScoreReason] on purpose. These four are the only ones
/// that reach the "score unavailable" surface, they each need a materially
/// different message and a different next action, and a five-member enum can be
/// switched exhaustively by the UI. Folding them into the seven-member
/// [ScoreReason] would force every copy switch to also handle codes that can
/// never appear there.
enum ScoreUnavailableReason {
  /// monthly_income_sgd is 0 or absent -> ask for income.
  incomeNotSet('income_not_set'),

  /// Age resolves outside the models' 20-70 range -> nothing the user can do.
  unsupportedAge('unsupported_age'),

  /// users/{uid}.date_of_birth missing -> finish onboarding.
  dateOfBirthNotSet('date_of_birth_not_set'),

  /// No financialProfile document at all -> finish onboarding.
  noFinancialProfile('no_financial_profile'),

  /// A code this build does not recognise. Treated as unavailable rather than
  /// guessed at.
  unknown('');

  const ScoreUnavailableReason(this.code);

  final String code;

  static ScoreUnavailableReason? parse(Object? value) {
    if (value == null) return null;
    for (final reason in ScoreUnavailableReason.values) {
      if (reason.code == value) return reason;
    }
    return ScoreUnavailableReason.unknown;
  }
}

/// Human-readable names for the 10 Layer 2 clustering features, so a
/// `missing_features` list can be shown without the UI hardcoding the backend's
/// artifact spelling. Keys are `scoring.py`'s per-domain feature lists verbatim
/// (artifact spelling — note Banking's `dti`, not `debt_to_income_ratio`).
const Map<String, String> kFeatureLabels = {
  // Banking
  'savings_rate': 'Savings rate',
  'emergency_fund_months': 'Emergency fund',
  'dti': 'Debt-to-income ratio',
  'healthy_debt_ratio': 'Healthy debt ratio',
  // Investment
  'investment_rate': 'Investment rate',
  'portfolio_concentration': 'Portfolio concentration',
  'risk_age_deviation': 'Risk-age alignment',
  // Insurance
  'life_coverage_ratio': 'Life coverage',
  'ci_coverage_ratio': 'Critical illness coverage',
  'insurance_spend_ratio': 'Insurance spend',
};

/// Human-readable names for the two scored Layer 1 checks
/// (`overall_score.LAYER1_CHECKS`).
const Map<String, String> kChecklistLabels = {
  'income_protection_coverage': 'Income protection cover',
  'hospitalisation_coverage': 'Hospitalisation cover',
};

/// One domain's Layer 2 result — `ml_profile.domains[domain]`.
class DomainScore {
  const DomainScore({
    required this.domain,
    required this.status,
    this.tierIndex,
    this.tierLabel,
    this.reason,
    this.missingFeatures = const [],
  });

  final ScoreDomain domain;
  final DomainStatus status;

  /// The authoritative key for downstream rules (0 = healthiest). The label is
  /// display wording that can be re-worded without recomputing anyone, which is
  /// why the backend caches both.
  final int? tierIndex;
  final String? tierLabel;

  /// Null when [status] is assessed.
  final ScoreReason? reason;

  /// Populated only for [ScoreReason.missingFeatures]; artifact feature names.
  final List<String> missingFeatures;

  bool get isAssessed => status == DomainStatus.assessed;

  /// Readable names for [missingFeatures], falling back to the raw key so an
  /// unrecognised feature is still shown rather than silently dropped.
  List<String> get missingFeatureLabels =>
      [for (final f in missingFeatures) kFeatureLabels[f] ?? f];

  factory DomainScore.fromMap(ScoreDomain domain, Map<String, dynamic> data) {
    return DomainScore(
      domain: domain,
      status: data['status'] == 'assessed'
          ? DomainStatus.assessed
          : DomainStatus.notAssessed,
      tierIndex: (data['tier_index'] as num?)?.toInt(),
      tierLabel: data['tier_label'] as String?,
      reason: ScoreReason.parse(data['reason']),
      missingFeatures: [
        for (final f in (data['missing_features'] as List? ?? const []))
          f.toString(),
      ],
    );
  }

  /// The shape a domain takes when the pipeline never got far enough to look at
  /// it — used to keep [ScoreProfile.domains] complete for all three domains
  /// even if the cached map is missing one.
  factory DomainScore.notAssessed(ScoreDomain domain, ScoreReason? reason) =>
      DomainScore(
        domain: domain,
        status: DomainStatus.notAssessed,
        reason: reason,
      );
}

/// `compute_score_breakdown()`'s working, cached — the score's explanation.
class ScoreBreakdown {
  const ScoreBreakdown({
    required this.bankingScore,
    required this.investmentScore,
    required this.insuranceIndicators,
    required this.insuranceIndicatorsCleared,
    required this.insuranceThird,
    required this.optionBScore,
    required this.ceiling,
    required this.capped,
    required this.overallScore,
  });

  /// 0.0 or 100.0 — Banking and Investment ship as two tiers each.
  final double bankingScore;
  final double investmentScore;

  /// The three 0/1 indicators, in the order insurance tier / income protection
  /// / hospitalisation.
  final List<double> insuranceIndicators;
  final int insuranceIndicatorsCleared;
  final double insuranceThird;

  /// The uncapped score, before the insurance ceiling is applied.
  final double optionBScore;

  /// 39.0 / 69.0 / 100.0, by indicators cleared.
  final double ceiling;

  /// True when the ceiling bound — i.e. closing the protection gap would raise
  /// this score. This is the brief's "critical gaps surface regardless of
  /// overall score" requirement, cached.
  final bool capped;

  final double overallScore;

  factory ScoreBreakdown.fromMap(Map<String, dynamic> data) {
    return ScoreBreakdown(
      bankingScore: (data['banking_score'] as num?)?.toDouble() ?? 0,
      investmentScore: (data['investment_score'] as num?)?.toDouble() ?? 0,
      insuranceIndicators: [
        for (final v in (data['insurance_indicators'] as List? ?? const []))
          (v as num).toDouble(),
      ],
      insuranceIndicatorsCleared:
          (data['insurance_indicators_cleared'] as num?)?.toInt() ?? 0,
      insuranceThird: (data['insurance_third'] as num?)?.toDouble() ?? 0,
      optionBScore: (data['option_b_score'] as num?)?.toDouble() ?? 0,
      ceiling: (data['ceiling'] as num?)?.toDouble() ?? 100,
      capped: data['capped'] as bool? ?? false,
      overallScore: (data['overall_score'] as num?)?.toDouble() ?? 0,
    );
  }
}

/// One benchmark-anchored coverage line — `ml_profile.insurance_coverage[key]`.
///
/// Every number here is the backend's, including the arithmetic between them.
/// [ratio] is `compute_insurance_features()`'s own value (so it can never
/// disagree with the Insurance tier shown beside it), [benchmarkRatio] is the
/// user's age band's value out of the committed artifacts, and the SGD figures
/// were multiplied out against the same income both were computed from. This
/// class re-derives none of it — see `ml/coverage_gaps.py`.
class CoverageGap {
  const CoverageGap({
    required this.feature,
    required this.ratio,
    required this.benchmarkRatio,
    required this.meetsBenchmark,
    required this.coveredSgd,
    required this.recommendedSgd,
    required this.shortfallSgd,
  });

  /// Which `compute_insurance_features()` key this line came from.
  final String feature;

  /// Multiples of annual income actually covered.
  final double ratio;

  /// The LIA-anchored multiple for this user's age band — 9x life (5x in band
  /// 5, 60-70), 3.9x critical illness.
  final double benchmarkRatio;

  final bool meetsBenchmark;
  final double coveredSgd;
  final double recommendedSgd;

  /// 0.0 when the benchmark is met — the backend floors it, so an over-covered
  /// user never reports a negative gap.
  final double shortfallSgd;

  /// No policy of this type at all, which needs different copy from "some
  /// cover, but short of the benchmark".
  bool get hasNoCover => ratio <= 0;

  factory CoverageGap.fromMap(Map<String, dynamic> data) => CoverageGap(
        feature: data['feature'] as String? ?? '',
        ratio: (data['ratio'] as num?)?.toDouble() ?? 0,
        benchmarkRatio: (data['benchmark_ratio'] as num?)?.toDouble() ?? 0,
        meetsBenchmark: data['meets_benchmark'] as bool? ?? false,
        coveredSgd: (data['covered_sgd'] as num?)?.toDouble() ?? 0,
        recommendedSgd: (data['recommended_sgd'] as num?)?.toDouble() ?? 0,
        shortfallSgd: (data['shortfall_sgd'] as num?)?.toDouble() ?? 0,
      );
}

/// `ml_profile.insurance_coverage` — the two continuous coverage ratios.
///
/// Null whenever the ratios were not computable (every whole-pipeline
/// unavailable state), and also null on a cache written before schema 1.1.0
/// added the block. Both mean the same thing to a renderer: nothing can be said
/// about this user's coverage yet, so say that rather than showing zeroes.
class InsuranceCoverage {
  const InsuranceCoverage({required this.life, required this.criticalIllness});

  final CoverageGap life;
  final CoverageGap criticalIllness;

  /// Both lines, in the order the Insurance screen shows them.
  List<CoverageGap> get all => [life, criticalIllness];

  static InsuranceCoverage? fromMap(Object? value) {
    final data = _asMap(value);
    // Partial is treated as absent: one half of a coverage assessment is not a
    // coverage assessment, and rendering the half we got would quietly imply
    // the other half is fine.
    if (data['life'] == null || data['critical_illness'] == null) return null;
    return InsuranceCoverage(
      life: CoverageGap.fromMap(_asMap(data['life'])),
      criticalIllness: CoverageGap.fromMap(_asMap(data['critical_illness'])),
    );
  }
}

/// One Layer 1 checklist entry — `checklist_flags.layer1[i]`.
class ChecklistCheck {
  const ChecklistCheck({
    required this.check,
    required this.passed,
    required this.message,
    required this.scored,
  });

  /// A `LAYER1_CHECKS` member.
  final String check;

  /// Null means "could not be evaluated", which is deliberately NOT a third
  /// failure mode — an incomplete profile must never read as a confirmed
  /// protection gap.
  final bool? passed;

  /// Backend-owned copy. Do not re-derive it client-side: the wording branches
  /// on dependants, which the client does not evaluate.
  final String message;

  /// Always true today; present so a future advisory check cannot be mistaken
  /// for a scored one.
  final bool scored;

  String get label => kChecklistLabels[check] ?? check;

  factory ChecklistCheck.fromMap(Map<String, dynamic> data) => ChecklistCheck(
        check: data['check'] as String? ?? '',
        passed: data['passed'] as bool?,
        message: data['message'] as String? ?? '',
        scored: data['scored'] as bool? ?? true,
      );
}

/// Which artifact build produced a cached score. If the models are rebuilt,
/// every cached score silently goes stale; this makes that detectable.
class ArtifactProvenance {
  const ArtifactProvenance({
    required this.schemaVersion,
    required this.generatedUtc,
  });

  final String? schemaVersion;
  final String? generatedUtc;

  factory ArtifactProvenance.fromMap(Map<String, dynamic> data) =>
      ArtifactProvenance(
        schemaVersion: data['schema_version'] as String?,
        generatedUtc: data['generated_utc'] as String?,
      );
}

/// One complete cached score run.
class ScoreProfile {
  const ScoreProfile({
    required this.schemaVersion,
    required this.computedAt,
    required this.domains,
    required this.checklist,
    required this.layer3Flags,
    required this.layer3Evaluated,
    this.layer3Reason,
    this.ageBand,
    this.ageBandRange,
    this.overallScore,
    this.overallScoreReason,
    this.unavailableReason,
    this.breakdown,
    this.insuranceCoverage,
    this.healthyDebtRatio,
    this.artifacts,
  });

  /// The cache shape this run was written under (`ML_PROFILE_SCHEMA_VERSION`).
  final String? schemaVersion;

  /// When the pipeline last ran. Equal to `checklist_last_updated` by
  /// construction — they describe the same computation.
  final DateTime computedAt;

  /// 1-5, or null when the age band could not be determined.
  final int? ageBand;

  /// e.g. "40-49".
  final String? ageBandRange;

  /// Always all three domains, in [ScoreDomain] order.
  final Map<ScoreDomain, DomainScore> domains;

  /// Layer 1 — always present, one entry per check, `passed` null where unknown.
  final List<ChecklistCheck> checklist;

  /// Layer 3 — 0-2 active cross-domain flag names.
  final List<String> layer3Flags;

  /// False means "not all three tiers were available, so nothing was tested" —
  /// NOT "tested and nothing fired". A UI showing "no issues found" must check
  /// this before trusting an empty [layer3Flags].
  final bool layer3Evaluated;
  final ScoreReason? layer3Reason;

  final double? overallScore;

  /// Null when [overallScore] is present; otherwise why it was withheld
  /// (normally [ScoreReason.incompleteDomains]).
  final ScoreReason? overallScoreReason;

  /// Set only on a whole-pipeline failure. When set, every domain carries the
  /// same reason.
  final ScoreUnavailableReason? unavailableReason;

  final ScoreBreakdown? breakdown;

  /// The Coverage Gaps section's numbers. Null when they were not computable —
  /// see [InsuranceCoverage].
  final InsuranceCoverage? insuranceCoverage;

  /// `ml_profile.healthy_debt_ratio` — the share (0-1) of classified debt that
  /// is structured (mortgage, student, auto, home equity) rather than revolving
  /// or unsecured, exactly as the Banking tier was assigned from. Cached by the
  /// backend so the client never re-classifies account subtypes.
  ///
  /// Null when not computable (no classifiable debt, or no bank accounts) and
  /// on runs cached before schema 1.2.0 — a different thing from 0.0, "every
  /// classified dollar is unhealthy debt". Independent of Banking's
  /// assessment: an assessed Banking tier can sit beside a null ratio.
  final double? healthyDebtRatio;

  final ArtifactProvenance? artifacts;

  /// The domains in the canonical display order.
  List<DomainScore> get orderedDomains =>
      [for (final d in ScoreDomain.values) domains[d]!];

  List<DomainScore> get assessedDomains =>
      [for (final d in orderedDomains) if (d.isAssessed) d];

  List<DomainScore> get unassessedDomains =>
      [for (final d in orderedDomains) if (!d.isAssessed) d];

  /// Builds from the WHOLE financialProfile/profile document — see the file
  /// header for why ml_profile alone is not enough.
  ///
  /// Callers must have already established that a score exists
  /// (`checklist_last_updated` and `ml_profile` both non-null); this
  /// constructor does not re-check.
  factory ScoreProfile.fromProfileData(Map<String, dynamic> data) {
    final ml = _asMap(data['ml_profile']);
    final checklistFlags = _asMap(data['checklist_flags']);

    final unavailable = ScoreUnavailableReason.parse(ml['unavailable_reason']);
    final fallbackReason = ScoreReason.parse(ml['unavailable_reason']);

    final rawDomains = _asMap(ml['domains']);
    final domains = <ScoreDomain, DomainScore>{
      for (final domain in ScoreDomain.values)
        domain: rawDomains[domain.key] == null
            ? DomainScore.notAssessed(domain, fallbackReason)
            : DomainScore.fromMap(domain, _asMap(rawDomains[domain.key])),
    };

    // computed_at and checklist_last_updated are written from one value; prefer
    // ml_profile's so the parsed run is internally consistent, and fall back
    // rather than making computedAt nullable for a case that cannot happen.
    final computedAt = _asDate(ml['computed_at']) ??
        _asDate(data['checklist_last_updated']) ??
        DateTime.fromMillisecondsSinceEpoch(0);

    return ScoreProfile(
      schemaVersion: ml['schema_version'] as String?,
      computedAt: computedAt,
      ageBand: (ml['age_band'] as num?)?.toInt(),
      ageBandRange: ml['age_band_range'] as String?,
      domains: domains,
      checklist: [
        for (final entry in (checklistFlags['layer1'] as List? ?? const []))
          ChecklistCheck.fromMap(_asMap(entry)),
      ],
      // ml_profile and checklist_flags carry the same Layer 3 payload by
      // design; ml_profile is the model-provenance copy and is read here so one
      // parse reads one source.
      layer3Flags: [
        for (final flag in (ml['layer3_flags'] as List? ?? const []))
          flag.toString(),
      ],
      layer3Evaluated: ml['layer3_evaluated'] as bool? ?? false,
      layer3Reason: ScoreReason.parse(checklistFlags['layer3_reason']),
      // Read from the top-level Section 8 field, which is the one the rest of
      // the app (and the notification hook) treats as authoritative;
      // ml_profile's copy is the self-contained record of the same run.
      overallScore: (data['overall_score'] as num?)?.toDouble() ??
          (ml['overall_score'] as num?)?.toDouble(),
      overallScoreReason: ScoreReason.parse(ml['overall_score_reason']),
      unavailableReason: unavailable,
      breakdown: ml['score_breakdown'] == null
          ? null
          : ScoreBreakdown.fromMap(_asMap(ml['score_breakdown'])),
      insuranceCoverage: InsuranceCoverage.fromMap(ml['insurance_coverage']),
      healthyDebtRatio: (ml['healthy_debt_ratio'] as num?)?.toDouble(),
      artifacts: ml['artifacts'] == null
          ? null
          : ArtifactProvenance.fromMap(_asMap(ml['artifacts'])),
    );
  }
}

Map<String, dynamic> _asMap(Object? value) =>
    (value as Map?)?.cast<String, dynamic>() ?? const {};

DateTime? _asDate(Object? value) {
  if (value is Timestamp) return value.toDate();
  if (value is DateTime) return value;
  return null;
}
