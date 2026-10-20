import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:unify/models/insurance_policy.dart';
import 'package:unify/utils/formatters.dart';

InsurancePolicy _policy({
  String type = 'life',
  double? sum,
  double? premium,
}) {
  return InsurancePolicy(
    policyId: 'p1',
    policyName: 'Test Plan',
    insurer: 'Prudential',
    policyType: type,
    sumAssuredSgd: sum,
    annualPremiumSgd: premium,
  );
}

void main() {
  group('formatSgd', () {
    test('adds comma separators and rounds', () {
      expect(formatSgd(245847), '\$245,847');
      expect(formatSgd(0), '\$0');
      expect(formatSgd(999), '\$999');
      expect(formatSgd(1000), '\$1,000');
      expect(formatSgd(7680.4), '\$7,680');
      expect(formatSgd(1234567.6), '\$1,234,568');
    });
  });

  group('formatShortDate', () {
    test('matches the design date style', () {
      expect(formatShortDate(DateTime(2027, 1, 1)), '1 Jan 2027');
      expect(formatShortDate(DateTime(2026, 9, 16)), '16 Sep 2026');
    });
  });

  group('InsurancePolicy.fromFirestore', () {
    test('parses a full backend-written document', () {
      final policy = InsurancePolicy.fromFirestore('abc123', {
        'policy_name': 'PruExtra Premier',
        'insurer': 'Prudential',
        'policy_type': 'hospitalisation',
        'policy_type_raw': 'Hospitalisation rider',
        'sum_assured_sgd': null,
        'annual_premium_sgd': 480.0,
        'renewal_date': Timestamp.fromDate(DateTime.utc(2026, 9, 16)),
        'exclusions': ['Pre-existing conditions'],
        'coverage_gaps': [],
        'original_filename': 'pruExtra.pdf',
        'uploaded_at': Timestamp.fromDate(DateTime.utc(2026, 7, 4)),
      });
      expect(policy.policyId, 'abc123');
      expect(policy.policyType, 'hospitalisation');
      expect(policy.policyTypeLabel, 'Hospitalisation');
      expect(policy.sumAssuredSgd, isNull);
      expect(policy.annualPremiumSgd, 480.0);
      // Timestamp.toDate() converts to local time; compare in UTC.
      expect(policy.renewalDate!.toUtc(), DateTime.utc(2026, 9, 16));
      expect(policy.exclusions, ['Pre-existing conditions']);
      expect(policy.originalFilename, 'pruExtra.pdf');
    });

    test('tolerates missing optional fields', () {
      final policy = InsurancePolicy.fromFirestore('x', {
        'policy_name': 'Bare',
        'insurer': 'AIA',
        'policy_type': 'other',
      });
      expect(policy.renewalDate, isNull);
      expect(policy.exclusions, isEmpty);
      expect(policy.policyTypeLabel, 'Other');
    });
  });

  group('CoverageSummary.fromPolicies', () {
    test('empty list yields zero everything', () {
      final s = CoverageSummary.fromPolicies([]);
      expect(s.policyCount, 0);
      expect(s.totalSumAssured, 0);
      expect(s.totalAnnualPremiums, 0);
      expect(s.lifePolicyCount, 0);
      expect(s.accidentPolicyCount, 0);
      expect(s.ciPolicyCount, 0);
    });

    test('groups personal_accident + hospitalisation as Accident', () {
      final s = CoverageSummary.fromPolicies([
        _policy(type: 'life', sum: 25000, premium: 1200),
        _policy(type: 'personal_accident', sum: null, premium: 480),
        _policy(type: 'hospitalisation', sum: null, premium: 600),
        _policy(type: 'critical_illness', sum: 100000, premium: 900),
      ]);
      expect(s.policyCount, 4);
      expect(s.totalSumAssured, 125000);
      expect(s.totalAnnualPremiums, 3180);
      expect(s.lifePolicyCount, 1);
      expect(s.lifeSumAssured, 25000);
      expect(s.accidentPolicyCount, 2);
      expect(s.accidentSumAssured, 0); // null sums by fabrication rule
      expect(s.ciPolicyCount, 1);
      expect(s.ciSumAssured, 100000);
    });

    test('income_protection and other count toward totals only', () {
      final s = CoverageSummary.fromPolicies([
        _policy(type: 'income_protection', sum: 50000, premium: 300),
        _policy(type: 'other', sum: null, premium: 200),
      ]);
      expect(s.policyCount, 2);
      expect(s.totalSumAssured, 50000);
      expect(s.totalAnnualPremiums, 500);
      expect(s.lifePolicyCount, 0);
      expect(s.accidentPolicyCount, 0);
      expect(s.ciPolicyCount, 0);
    });
  });
}
