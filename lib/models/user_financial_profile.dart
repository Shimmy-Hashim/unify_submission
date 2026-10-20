import 'package:cloud_firestore/cloud_firestore.dart';

/// Firestore users/{uid}/financialProfile document (Section 8 of the
/// project brief). Section 8 specifies a single document, not a list —
/// implemented as a one-document subcollection with the fixed ID "profile"
/// (see FirebaseService.financialProfileDoc).
///
/// Write paths (Home Phase 1):
///   - monthly_income_sgd / income_source: written once by the client at
///     onboarding (the only client write path — accounts predating this
///     collection are test/dev-only and get deleted, not retrofitted)
///   - monthly_expenses_sgd / net_worth_sgd: derived — recomputed by the
///     backend on every Plaid sync and on POST /profile/refresh; the client
///     only seeds them at 0.0 on creation
///   - total_assets_sgd / total_debts_sgd: derived, backend-only, written in
///     the same pass as net_worth_sgd (`compute_balance_totals()`), so
///     assets − debts == net worth by construction. Never seeded by the
///     client, so a document created before they existed has neither
///   - overall_score / checklist_flags / checklist_last_updated /
///     ml_profile: Phase 3 outputs, null until the ML inference endpoint
///     exists
class UserFinancialProfile {
  const UserFinancialProfile({
    required this.monthlyIncomeSgd,
    required this.incomeSource,
    this.monthlyExpensesSgd,
    this.netWorthSgd,
    this.totalAssetsSgd,
    this.totalDebtsSgd,
    this.overallScore,
    this.checklistFlags,
    this.checklistLastUpdated,
    this.mlProfile,
  });

  final double monthlyIncomeSgd;
  final String incomeSource; // "user_entered" | "plaid_derived"
  final double? monthlyExpensesSgd;
  final double? netWorthSgd;

  /// Null on a document the backend has not refreshed since these fields
  /// were added — which is "unknown", not zero.
  final double? totalAssetsSgd;
  final double? totalDebtsSgd;
  final double? overallScore;
  final Map<String, dynamic>? checklistFlags;
  final DateTime? checklistLastUpdated;
  final Map<String, dynamic>? mlProfile;

  factory UserFinancialProfile.fromMap(Map<String, dynamic> data) {
    return UserFinancialProfile(
      monthlyIncomeSgd: (data['monthly_income_sgd'] as num?)?.toDouble() ?? 0,
      incomeSource: data['income_source'] as String? ?? 'user_entered',
      monthlyExpensesSgd: (data['monthly_expenses_sgd'] as num?)?.toDouble(),
      netWorthSgd: (data['net_worth_sgd'] as num?)?.toDouble(),
      totalAssetsSgd: (data['total_assets_sgd'] as num?)?.toDouble(),
      totalDebtsSgd: (data['total_debts_sgd'] as num?)?.toDouble(),
      overallScore: (data['overall_score'] as num?)?.toDouble(),
      checklistFlags: (data['checklist_flags'] as Map?)?.cast<String, dynamic>(),
      checklistLastUpdated:
          (data['checklist_last_updated'] as Timestamp?)?.toDate(),
      mlProfile: (data['ml_profile'] as Map?)?.cast<String, dynamic>(),
    );
  }

  factory UserFinancialProfile.fromDoc(
          DocumentSnapshot<Map<String, dynamic>> doc) =>
      UserFinancialProfile.fromMap(doc.data() ?? const {});

  /// Fields written when the document is first created. Income is always
  /// user-entered in Phase 1 — deriving it from Plaid transaction patterns
  /// is a documented future enhancement, not built. Derived fields start at
  /// 0.0 and are overwritten by the backend refresh; Phase 3 fields start
  /// null and are never written by the client.
  static Map<String, dynamic> createMap(double monthlyIncomeSgd) => {
        'monthly_income_sgd': monthlyIncomeSgd,
        'income_source': 'user_entered',
        'monthly_expenses_sgd': 0.0,
        'net_worth_sgd': 0.0,
        'overall_score': null,
        'checklist_flags': null,
        'checklist_last_updated': null,
        'ml_profile': null,
      };

  /// Validator for the onboarding monthly-income input. Zero is allowed
  /// (students / NS / between jobs); scoring layers that divide by income
  /// handle that case explicitly.
  static String? validateIncomeInput(String? value) {
    final text = value?.trim() ?? '';
    if (text.isEmpty) return 'Enter your monthly income';
    final parsed = double.tryParse(text.replaceAll(',', ''));
    if (parsed == null) return 'Enter a valid amount';
    if (parsed < 0) return 'Income cannot be negative';
    if (parsed > 1000000) return 'Enter your monthly (not annual) income';
    return null;
  }

  /// Parses the validated input text ("4,500" → 4500.0).
  static double parseIncomeInput(String value) =>
      double.parse(value.trim().replaceAll(',', ''));
}
