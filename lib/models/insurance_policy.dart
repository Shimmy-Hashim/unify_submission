import 'package:cloud_firestore/cloud_firestore.dart';

/// One document from users/{uid}/insurancePolicies/{policyId}, written by the
/// backend `POST /insurance/parse` endpoint (see Section 8 of the project
/// brief). The client only ever reads these.
class InsurancePolicy {
  const InsurancePolicy({
    required this.policyId,
    required this.policyName,
    required this.insurer,
    required this.policyType,
    this.policyTypeRaw,
    this.sumAssuredSgd,
    this.annualPremiumSgd,
    this.renewalDate,
    this.exclusions = const [],
    this.coverageGaps = const [],
    this.originalFilename,
    this.uploadedAt,
  });

  final String policyId;
  final String policyName;
  final String insurer;

  /// Strict enum from the parser: life | critical_illness | hospitalisation |
  /// income_protection | personal_accident | other.
  final String policyType;

  /// The model's verbatim read of the policy's own stated type (free text).
  final String? policyTypeRaw;

  /// Null is expected (not an error) for personal_accident/other policies —
  /// the parser's fabrication-avoidance rule.
  final double? sumAssuredSgd;
  final double? annualPremiumSgd;
  final DateTime? renewalDate;
  final List<String> exclusions;
  final List<String> coverageGaps;
  final String? originalFilename;
  final DateTime? uploadedAt;

  factory InsurancePolicy.fromFirestore(
      String id, Map<String, dynamic> data) {
    return InsurancePolicy(
      policyId: id,
      policyName: data['policy_name'] as String? ?? 'Untitled policy',
      insurer: data['insurer'] as String? ?? 'Unknown insurer',
      policyType: data['policy_type'] as String? ?? 'other',
      policyTypeRaw: data['policy_type_raw'] as String?,
      sumAssuredSgd: (data['sum_assured_sgd'] as num?)?.toDouble(),
      annualPremiumSgd: (data['annual_premium_sgd'] as num?)?.toDouble(),
      renewalDate: (data['renewal_date'] as Timestamp?)?.toDate(),
      exclusions: (data['exclusions'] as List?)?.cast<String>() ?? const [],
      coverageGaps:
          (data['coverage_gaps'] as List?)?.cast<String>() ?? const [],
      originalFilename: data['original_filename'] as String?,
      uploadedAt: (data['uploaded_at'] as Timestamp?)?.toDate(),
    );
  }

  /// Human-readable label for the policy_type enum value.
  String get policyTypeLabel => switch (policyType) {
        'life' => 'Life',
        'critical_illness' => 'Critical Illness',
        'hospitalisation' => 'Hospitalisation',
        'income_protection' => 'Income Protection',
        'personal_accident' => 'Personal Accident',
        _ => 'Other',
      };
}

/// Sums computable directly from the policy list — powers the Total Coverage
/// card. Deliberately contains no gap/risk analysis (that needs
/// financialProfile income data and lives in the backend checklist logic).
class CoverageSummary {
  const CoverageSummary({
    required this.totalSumAssured,
    required this.totalAnnualPremiums,
    required this.policyCount,
    required this.lifeSumAssured,
    required this.lifePolicyCount,
    required this.accidentSumAssured,
    required this.accidentPolicyCount,
    required this.ciSumAssured,
    required this.ciPolicyCount,
  });

  final double totalSumAssured;
  final double totalAnnualPremiums;
  final int policyCount;
  final double lifeSumAssured;
  final int lifePolicyCount;

  /// "Accident" column groups personal_accident + hospitalisation.
  final double accidentSumAssured;
  final int accidentPolicyCount;
  final double ciSumAssured;
  final int ciPolicyCount;

  factory CoverageSummary.fromPolicies(List<InsurancePolicy> policies) {
    double total = 0, premiums = 0, life = 0, accident = 0, ci = 0;
    int lifeCount = 0, accidentCount = 0, ciCount = 0;
    for (final p in policies) {
      total += p.sumAssuredSgd ?? 0;
      premiums += p.annualPremiumSgd ?? 0;
      switch (p.policyType) {
        case 'life':
          lifeCount++;
          life += p.sumAssuredSgd ?? 0;
        case 'personal_accident' || 'hospitalisation':
          accidentCount++;
          accident += p.sumAssuredSgd ?? 0;
        case 'critical_illness':
          ciCount++;
          ci += p.sumAssuredSgd ?? 0;
      }
    }
    return CoverageSummary(
      totalSumAssured: total,
      totalAnnualPremiums: premiums,
      policyCount: policies.length,
      lifeSumAssured: life,
      lifePolicyCount: lifeCount,
      accidentSumAssured: accident,
      accidentPolicyCount: accidentCount,
      ciSumAssured: ci,
      ciPolicyCount: ciCount,
    );
  }
}
