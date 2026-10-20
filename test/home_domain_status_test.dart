import 'dart:convert';
import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:unify/models/home_score_state.dart';
import 'package:unify/models/score_profile.dart';
import 'package:unify/screens/home_screen.dart';
import 'package:unify/services/health_score_service.dart';
import 'package:unify/utils/app_styles.dart';

// The `domains` maps below are verbatim output of the real
// backend/ml/score_pipeline.compute_score_profile() for each scenario (run
// locally with synthetic inputs), not hand-written approximations.

/// (a) Checking + credit card + mortgage, monthly_expenses_sgd = 0.
const _zeroExpensesDomains = {
  'banking': {
    'status': 'not_assessed',
    'tier_index': null,
    'tier_label': null,
    'reason': 'missing_features',
    'missing_features': ['emergency_fund_months'],
  },
  'investment': _noInvestments,
  'insurance': _notCovered,
};

/// (b) One healthy checking account, no debt of any kind.
const _debtFreeDomains = {
  'banking': {
    'status': 'not_assessed',
    'tier_index': null,
    'tier_label': null,
    'reason': 'missing_features',
    'missing_features': ['healthy_debt_ratio'],
  },
  'investment': _noInvestments,
  'insurance': _notCovered,
};

/// Control: no plaidAccounts at all.
const _noAccountsDomains = {
  'banking': {
    'status': 'not_assessed',
    'tier_index': null,
    'tier_label': null,
    'reason': 'missing_features',
    'missing_features': [
      'savings_rate',
      'emergency_fund_months',
      'dti',
      'healthy_debt_ratio',
    ],
  },
  'investment': _noInvestments,
  'insurance': _notCovered,
};

const _noInvestments = {
  'status': 'not_assessed',
  'tier_index': null,
  'tier_label': null,
  'reason': 'missing_features',
  'missing_features': [
    'investment_rate',
    'portfolio_concentration',
    'risk_age_deviation',
  ],
};

const _notCovered = {
  'status': 'assessed',
  'tier_index': 1,
  'tier_label': 'Not Covered',
  'reason': null,
  'missing_features': <String>[],
};

/// A partial cached run, shaped as `cache_payload()` writes it.
ScoreState _partial(Map<String, dynamic> domains) {
  final at = Timestamp.fromDate(DateTime.utc(2026, 9, 13));
  final state = HealthScoreService.stateFromProfileData({
    'monthly_income_sgd': 5000.0,
    'overall_score': null,
    'checklist_last_updated': at,
    'checklist_flags': {'layer1': const [], 'layer3': const []},
    'ml_profile': {
      'schema_version': '1.2.0',
      'computed_at': at,
      'age_band': 2,
      'overall_score': null,
      'overall_score_reason': 'incomplete_domains',
      'layer3_flags': const <String>[],
      'layer3_evaluated': false,
      'domains': domains,
    },
  });
  expect(state, isA<ScorePartial>());
  return state;
}

Future<void> _pumpCard(
  WidgetTester tester,
  ScoreState state,
  AccountPresence presence,
) async {
  // These tests check copy, not layout. flutter_test renders text in Ahem,
  // where every glyph is a full em wide — roughly double Inter's width — so
  // at a real phone width the rows overflow in a way they do not on device.
  // A wide surface keeps that artifact out of the assertions.
  tester.view.physicalSize = const Size(1000, 1000);
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(MaterialApp(
    theme: AppTheme.lightTheme,
    home: Scaffold(
      body: SizedBox(
        width: 900,
        child: HealthScoreCard(state: state, presence: presence),
      ),
    ),
  ));
}

/// The row value text for one domain — the Text beside the emoji and name.
String _rowValue(WidgetTester tester, String name) {
  final nameFinder = find.textContaining(name);
  final row = find.ancestor(of: nameFinder, matching: find.byType(Row)).first;
  final texts = tester
      .widgetList<Text>(find.descendant(of: row, matching: find.byType(Text)))
      .map((t) => t.data)
      .toList();
  return texts.last!;
}

void main() {
  const connected = AccountPresence(
    hasBankAccounts: true,
    hasInvestmentAccounts: false,
  );
  const nothingConnected = AccountPresence(
    hasBankAccounts: false,
    hasInvestmentAccounts: false,
  );

  group('domainRowStatus', () {
    DomainScore banking(Map<String, dynamic> domains) => DomainScore.fromMap(
        ScoreDomain.banking, domains['banking'] as Map<String, dynamic>);

    test('(a) zero expenses with bank accounts -> cannot calculate yet', () {
      expect(
        domainRowStatus(
            ScoreDomain.banking, banking(_zeroExpensesDomains), connected),
        DomainRowStatus.cannotCalculateYet,
      );
    });

    test('(b) debt-free with a bank account -> cannot calculate yet', () {
      expect(
        domainRowStatus(
            ScoreDomain.banking, banking(_debtFreeDomains), connected),
        DomainRowStatus.cannotCalculateYet,
      );
    });

    test('a credit-only user missing both depository features is connected',
        () {
      const creditOnly = DomainScore(
        domain: ScoreDomain.banking,
        status: DomainStatus.notAssessed,
        reason: ScoreReason.missingFeatures,
        missingFeatures: ['savings_rate', 'emergency_fund_months'],
      );
      expect(domainRowStatus(ScoreDomain.banking, creditOnly, connected),
          DomainRowStatus.cannotCalculateYet);
    });

    test('(c) never computed follows account presence per domain', () {
      const both = AccountPresence(
          hasBankAccounts: true, hasInvestmentAccounts: true);
      for (final domain in ScoreDomain.values) {
        expect(domainRowStatus(domain, null, both),
            DomainRowStatus.cannotCalculateYet,
            reason: domain.name);
      }
      expect(domainRowStatus(ScoreDomain.investment, null, connected),
          DomainRowStatus.notConnected);
    });

    test('zero accounts of the type -> not connected, computed or not', () {
      expect(
        domainRowStatus(
            ScoreDomain.banking, banking(_noAccountsDomains), nothingConnected),
        DomainRowStatus.notConnected,
      );
      expect(domainRowStatus(ScoreDomain.banking, null, nothingConnected),
          DomainRowStatus.notConnected);
    });

    test('presence not loaded yet claims neither', () {
      expect(
        domainRowStatus(
            ScoreDomain.banking, null, const AccountPresence()),
        DomainRowStatus.unknown,
      );
    });

    test('a real tier always wins over presence', () {
      const stable = DomainScore(
        domain: ScoreDomain.banking,
        status: DomainStatus.assessed,
        tierIndex: 0,
        tierLabel: 'Stable',
      );
      expect(domainRowStatus(ScoreDomain.banking, stable, nothingConnected),
          DomainRowStatus.assessed);
    });

    test('insurance is never "not connected" — there is nothing to connect',
        () {
      for (final presence in const [
        AccountPresence(),
        nothingConnected,
      ]) {
        expect(domainRowStatus(ScoreDomain.insurance, null, presence),
            DomainRowStatus.cannotCalculateYet);
      }
    });
  });

  group('HealthScoreCard rows', () {
    testWidgets('(a) zero expenses: Liquidity reads "Can\'t score yet"',
        (tester) async {
      await _pumpCard(tester, _partial(_zeroExpensesDomains), connected);
      expect(_rowValue(tester, 'Liquidity'), "Can't score yet");
      expect(_rowValue(tester, 'Growth'), 'Not connected');
      expect(_rowValue(tester, 'Protection'), 'Not Covered');
    });

    testWidgets('(b) debt-free: Liquidity reads "Can\'t score yet"',
        (tester) async {
      await _pumpCard(tester, _partial(_debtFreeDomains), connected);
      expect(_rowValue(tester, 'Liquidity'), "Can't score yet");
      expect(_rowValue(tester, 'Growth'), 'Not connected');
    });

    testWidgets('(c) never computed with bank + investment accounts',
        (tester) async {
      await _pumpCard(
        tester,
        const ScoreNeverComputed(),
        const AccountPresence(
            hasBankAccounts: true, hasInvestmentAccounts: true),
      );
      expect(find.text('Not connected'), findsNothing);
      expect(find.text("Can't score yet"), findsNWidgets(3));
    });

    testWidgets('(c) never computed with bank accounts only', (tester) async {
      await _pumpCard(tester, const ScoreNeverComputed(), connected);
      expect(_rowValue(tester, 'Liquidity'), "Can't score yet");
      expect(_rowValue(tester, 'Growth'), 'Not connected');
      expect(_rowValue(tester, 'Protection'), "Can't score yet");
    });

    testWidgets('zero accounts still reads "Not connected"', (tester) async {
      await _pumpCard(tester, _partial(_noAccountsDomains), nothingConnected);
      expect(_rowValue(tester, 'Liquidity'), 'Not connected');
      expect(_rowValue(tester, 'Growth'), 'Not connected');

      await _pumpCard(tester, const ScoreNeverComputed(), nothingConnected);
      expect(_rowValue(tester, 'Liquidity'), 'Not connected');
      expect(_rowValue(tester, 'Growth'), 'Not connected');
    });

    testWidgets('bar sub-labels match the artifacts\' locked tier labels',
        (tester) async {
      final manifest = jsonDecode(
        File('backend/ml/artifacts/domain_split/manifest.json')
            .readAsStringSync(),
      ) as Map<String, dynamic>;
      final domains = manifest['domains'] as Map<String, dynamic>;

      await _pumpCard(tester, const ScoreNeverComputed(), nothingConnected);
      for (final domain in ScoreDomain.values) {
        final labels = ((domains[domain.key] as Map)['tier_labels'] as List)
            .cast<String>();
        expect(labels, hasLength(2));
        for (final label in labels) {
          expect(find.text(label), findsOneWidget,
              reason: '${domain.key} "$label"');
        }
      }
      // The pre-fix spellings must be gone.
      for (final stale in ['Getting started', 'On track', 'Not covered']) {
        expect(find.text(stale), findsNothing);
      }
    });
  });
}
