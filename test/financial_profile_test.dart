import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:unify/models/user_financial_profile.dart';

void main() {
  group('UserFinancialProfile.createMap', () {
    test('writes income fields, seeds derived at 0.0, Phase 3 fields null',
        () {
      final map = UserFinancialProfile.createMap(4500);
      expect(map['monthly_income_sgd'], 4500.0);
      expect(map['income_source'], 'user_entered');
      expect(map['monthly_expenses_sgd'], 0.0);
      expect(map['net_worth_sgd'], 0.0);
      expect(map['overall_score'], isNull);
      expect(map['checklist_flags'], isNull);
      expect(map['checklist_last_updated'], isNull);
      expect(map['ml_profile'], isNull);
      // Section 8 schema — exactly these eight fields, nothing extra.
      expect(map.length, 8);
    });
  });

  group('UserFinancialProfile.fromMap', () {
    test('parses a full document', () {
      final profile = UserFinancialProfile.fromMap({
        'monthly_income_sgd': 4500,
        'income_source': 'user_entered',
        'monthly_expenses_sgd': 2100.5,
        'net_worth_sgd': 15000,
        'overall_score': 72.5,
        'checklist_flags': {'has_emergency_fund': true},
        'checklist_last_updated': Timestamp.fromDate(DateTime(2026, 7, 6)),
        'ml_profile': null,
      });
      expect(profile.monthlyIncomeSgd, 4500.0);
      expect(profile.incomeSource, 'user_entered');
      expect(profile.monthlyExpensesSgd, 2100.5);
      expect(profile.netWorthSgd, 15000.0);
      expect(profile.overallScore, 72.5);
      expect(profile.checklistFlags, {'has_emergency_fund': true});
      expect(profile.checklistLastUpdated, DateTime(2026, 7, 6));
      expect(profile.mlProfile, isNull);
    });

    test('tolerates a Phase 1 document (nulls for score fields)', () {
      final profile = UserFinancialProfile.fromMap(
          UserFinancialProfile.createMap(3200));
      expect(profile.monthlyIncomeSgd, 3200.0);
      expect(profile.overallScore, isNull);
      expect(profile.checklistLastUpdated, isNull);
    });

    test('parses total assets and debts; absent means null, not zero', () {
      final profile = UserFinancialProfile.fromMap({
        'monthly_income_sgd': 4500,
        'net_worth_sgd': -116876.25,
        'total_assets_sgd': 264106.0,
        'total_debts_sgd': 380982.25,
      });
      expect(profile.totalAssetsSgd, 264106.0);
      expect(profile.totalDebtsSgd, 380982.25);

      final old = UserFinancialProfile.fromMap(
          UserFinancialProfile.createMap(3200));
      expect(old.totalAssetsSgd, isNull);
      expect(old.totalDebtsSgd, isNull);
    });
  });

  group('UserFinancialProfile.validateIncomeInput', () {
    test('accepts plain, decimal, and comma-grouped amounts', () {
      expect(UserFinancialProfile.validateIncomeInput('4500'), isNull);
      expect(UserFinancialProfile.validateIncomeInput('4500.50'), isNull);
      expect(UserFinancialProfile.validateIncomeInput('4,500'), isNull);
      expect(UserFinancialProfile.validateIncomeInput(' 4500 '), isNull);
    });

    test('accepts zero (students / NS / between jobs)', () {
      expect(UserFinancialProfile.validateIncomeInput('0'), isNull);
    });

    test('rejects empty, non-numeric, negative, implausible', () {
      expect(UserFinancialProfile.validateIncomeInput(null), isNotNull);
      expect(UserFinancialProfile.validateIncomeInput(''), isNotNull);
      expect(UserFinancialProfile.validateIncomeInput('abc'), isNotNull);
      expect(UserFinancialProfile.validateIncomeInput('-100'), isNotNull);
      expect(UserFinancialProfile.validateIncomeInput('2000000'), isNotNull);
    });
  });

  group('UserFinancialProfile.parseIncomeInput', () {
    test('parses what the validator accepts', () {
      expect(UserFinancialProfile.parseIncomeInput('4500'), 4500.0);
      expect(UserFinancialProfile.parseIncomeInput('4,500.75'), 4500.75);
      expect(UserFinancialProfile.parseIncomeInput(' 0 '), 0.0);
    });
  });
}
