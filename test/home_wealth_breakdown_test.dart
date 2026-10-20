import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:unify/models/home_score_state.dart';
import 'package:unify/models/insurance_policy.dart';
import 'package:unify/models/investment_holding.dart';
import 'package:unify/models/plaid_account.dart';
import 'package:unify/models/score_profile.dart';
import 'package:unify/screens/home_screen.dart';
import 'package:unify/services/app_preferences.dart';
import 'package:unify/services/health_score_service.dart';
import 'package:unify/utils/app_styles.dart';
import 'package:unify/utils/formatters.dart';

typedef _Docs = List<QueryDocumentSnapshot<Map<String, dynamic>>>;

// The card only ever reads `id` and `data()` off a snapshot. The package marks
// the snapshot classes @sealed (a lint, not a language modifier) and ships no
// public constructor, and the project has no fake_cloud_firestore — so a
// test-only implementation is the only way to hand the card real doc lists.
// ignore: subtype_of_sealed_class
class _Doc implements QueryDocumentSnapshot<Map<String, dynamic>> {
  _Doc(this.id, this._data);

  @override
  final String id;
  final Map<String, dynamic> _data;

  @override
  Map<String, dynamic> data() => _data;

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

_Docs _docs(Map<String, Map<String, dynamic>> byId) =>
    [for (final e in byId.entries) _Doc(e.key, e.value)];

/// A plaidAccounts document as the backend writes it.
Map<String, dynamic> _acct(
  String institution,
  String type,
  String subtype,
  String mask,
  double? current, {
  double? available,
}) =>
    {
      'account_name':
          'Plaid ${subtype[0].toUpperCase()}${subtype.substring(1)}',
      'account_type': type,
      'account_subtype': subtype,
      'mask': mask,
      'institution_name': institution,
      'balance_current': current,
      'balance_available': available,
    };

/// Mixed depository + debt across three institutions. First Platypus Bank is
/// the real O7oEGbN3 Sandbox user's balances (the Savings bug-fix fixture:
/// $62,589.00 of balance_current cash, $126,994.06 owed). Tartan Bank adds a
/// second bank holding both savings and a card; Houndstooth holds only a loan
/// — so Debts has to aggregate across all three, and Savings has to drop one.
final _bank = _docs({
  'd1': _acct('First Platypus Bank', 'depository', 'cd', '1111', 1000),
  'd2': _acct('First Platypus Bank', 'depository', 'checking', '0000', 110,
      available: 100),
  'd3': _acct('First Platypus Bank', 'depository', 'money market', '4444',
      43200, available: 43200),
  'd4': _acct('First Platypus Bank', 'depository', 'savings', '1111', 210,
      available: 200),
  'd5': _acct('First Platypus Bank', 'depository', 'cash management', '9002',
      12060, available: 12060),
  'd6': _acct('First Platypus Bank', 'depository', 'hsa', '6009', 6009,
      available: 6009),
  'c1': _acct('First Platypus Bank', 'credit', 'credit card', '3333', 410),
  'c2': _acct('First Platypus Bank', 'credit', 'credit card', '9999', 5020,
      available: 4980),
  'l1': _acct('First Platypus Bank', 'loan', 'student', '4321', 65262),
  'l2': _acct('First Platypus Bank', 'loan', 'mortgage', '8888', 56302.06),
  't1': _acct('Tartan Bank', 'depository', 'savings', '7070', 2450.40,
      available: 2400),
  't2': _acct('Tartan Bank', 'credit', 'credit card', '7734', 1289.55,
      available: 8710.45),
  'h1': _acct('Houndstooth Bank', 'loan', 'auto', '5150', 18000),
});

final _investments = _docs({
  'brk': {
    'account_name': 'Plaid Brokerage',
    'total_value_sgd': 20000.0,
    'holdings': [
      {'ticker': 'EWZ', 'name': 'iShares MSCI Brazil ETF', 'market_value': 5000.0},
      {'ticker': 'AAPL', 'name': 'Apple Inc.', 'market_value': 15000.0},
    ],
  },
  'ira': {
    'account_name': 'Plaid IRA',
    'total_value_sgd': 4500.0,
    'holdings': [
      {'ticker': 'BTC', 'market_value': 4500.0},
    ],
  },
});

final _policies = _docs({
  'p1': {
    'policy_name': 'Term Life',
    'insurer': 'Insurer A',
    'policy_type': 'life',
    'sum_assured_sgd': 500000.0,
    'annual_premium_sgd': 1200.0,
  },
  'p2': {
    'policy_name': 'PA Plan',
    'insurer': 'Insurer B',
    'policy_type': 'personal_accident',
    'sum_assured_sgd': null, // parser rule: null for personal_accident
    'annual_premium_sgd': 300.0,
  },
});

/// A cached run carrying [ratio], shaped as `cache_payload()` writes it.
ScoreState _scoreWithRatio(double? ratio) =>
    HealthScoreService.stateFromProfileData({
      'overall_score': null,
      'checklist_last_updated': Timestamp.fromDate(DateTime.utc(2026, 9, 13)),
      'checklist_flags': {'layer1': const [], 'layer3': const []},
      'ml_profile': {
        'computed_at': Timestamp.fromDate(DateTime.utc(2026, 9, 13)),
        'overall_score': null,
        'overall_score_reason': 'incomplete_domains',
        'layer3_evaluated': false,
        'domains': const {},
        'healthy_debt_ratio': ratio,
      },
    });

Future<void> _pump(
  WidgetTester tester, {
  required _Docs? bank,
  required _Docs? investments,
  required _Docs? policies,
  ScoreState score = const ScoreLoading(),
  VoidCallback? onOpenInvestments,
}) async {
  tester.view.physicalSize = const Size(1000, 3000);
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(MaterialApp(
    theme: AppTheme.lightTheme,
    home: Scaffold(
      body: SizedBox(
        width: 900,
        child: SingleChildScrollView(
          child: WealthBreakdownCard(
            bankDocs: bank,
            investmentDocs: investments,
            policyDocs: policies,
            score: score,
            onOpenInvestments: onOpenInvestments,
          ),
        ),
      ),
    ),
  ));
}

Future<void> _pumpFull(WidgetTester tester, {ScoreState? score}) => _pump(
      tester,
      bank: _bank,
      investments: _investments,
      policies: _policies,
      score: score ?? const ScoreLoading(),
    );

/// Rows render top to bottom in this order.
const _rows = ['Savings', 'Investments', 'Insurance Coverage', 'Debts'];

CrossFadeState _fade(WidgetTester tester, String label) => tester
    .widget<AnimatedCrossFade>(
        find.byType(AnimatedCrossFade).at(_rows.indexOf(label)))
    .crossFadeState;

List<CrossFadeState> _fades(WidgetTester tester) =>
    [for (final r in _rows) _fade(tester, r)];

Future<void> _expand(WidgetTester tester, String label) async {
  await tester.tap(find.text(label));
  await tester.pumpAndSettle();
}

Future<void> _expandAll(WidgetTester tester) async {
  for (final label in _rows) {
    await tester.tap(find.text(label));
  }
  await tester.pumpAndSettle();
}

/// The figure printed beside [label] in its own header row.
Text _headerTotal(WidgetTester tester, String label) {
  final header =
      find.ancestor(of: find.text(label), matching: find.byType(Row)).first;
  final texts = tester
      .widgetList<Text>(find.descendant(of: header, matching: find.byType(Text)))
      .toList();
  expect(texts.first.data, label);
  return texts[1];
}

/// The texts inside one row's expanded body, in order.
List<String> _bodyTexts(WidgetTester tester, String label) {
  final body = find.byType(AnimatedCrossFade).at(_rows.indexOf(label));
  return [
    for (final t in tester.widgetList<Text>(
        find.descendant(of: body, matching: find.byType(Text))))
      t.data!,
  ];
}

/// The money figures (and dashes) in one row's body.
List<String> _money(WidgetTester tester, String label) => [
      for (final t in _bodyTexts(tester, label))
        if (RegExp(r'^-?\$[\d,]+$').hasMatch(t) || t == '-') t,
    ];

double _parse(String money) =>
    double.parse(money.replaceAll(RegExp(r'[$,]'), ''));

List<PlaidAccount> _plaid(_Docs docs) =>
    [for (final d in docs) PlaidAccount.fromFirestore(d.id, d.data())];

const _red = Color(0xFFC62828);

void main() {
  group('rows and totals', () {
    testWidgets('four rows in order, totals from the real summary classes',
        (tester) async {
      await _pumpFull(tester);

      expect(find.byType(AnimatedCrossFade), findsNWidgets(4));
      final tops = [for (final r in _rows) tester.getTopLeft(find.text(r)).dy];
      expect(tops, orderedEquals([...tops]..sort()));

      final accounts = _plaid(_bank);
      final savings =
          SavingsSummary(accounts: accounts, transactionsByAccount: const {});
      final portfolio = PortfolioSummary(accounts: [
        for (final d in _investments)
          InvestmentAccount.fromFirestore(d.id, d.data()),
      ]);
      final coverage = CoverageSummary.fromPolicies([
        for (final d in _policies)
          InsurancePolicy.fromFirestore(d.id, d.data()),
      ]);
      expect(_headerTotal(tester, 'Savings').data,
          formatSgd(savings.totalSavings));
      expect(_headerTotal(tester, 'Investments').data,
          formatSgd(portfolio.totalValue));
      expect(_headerTotal(tester, 'Insurance Coverage').data,
          formatSgd(coverage.totalSumAssured));
      expect(_headerTotal(tester, 'Debts').data, formatSgd(-accounts.owedTotal));

      // Literal figures: $62,589.00 + Tartan's $2,450.40 of balance_current;
      // $126,994.06 + $1,289.55 + $18,000 owed across all three institutions.
      expect(_headerTotal(tester, 'Savings').data, '\$65,039');
      expect(_headerTotal(tester, 'Investments').data, '\$24,500');
      expect(_headerTotal(tester, 'Insurance Coverage').data, '\$500,000');
      expect(_headerTotal(tester, 'Debts').data, '-\$146,284');
    });

    testWidgets('Debts reads red, label and figure, when there is debt',
        (tester) async {
      await _pumpFull(tester);
      expect(tester.widget<Text>(find.text('Debts')).style!.color, _red);
      expect(_headerTotal(tester, 'Debts').style!.color, _red);
      expect(tester.widget<Text>(find.text('Savings')).style!.color,
          isNot(_red));
      expect(_headerTotal(tester, 'Savings').style!.color, isNot(_red));
    });

    testWidgets('collapsed by default; each row toggles on its own',
        (tester) async {
      await _pumpFull(tester);
      expect(_fades(tester), everyElement(CrossFadeState.showFirst));

      await _expand(tester, 'Debts');
      await _expand(tester, 'Investments');
      expect(_fades(tester), [
        CrossFadeState.showFirst,
        CrossFadeState.showSecond,
        CrossFadeState.showFirst,
        CrossFadeState.showSecond,
      ]);

      await _expand(tester, 'Debts');
      expect(_fades(tester), [
        CrossFadeState.showFirst,
        CrossFadeState.showSecond,
        CrossFadeState.showFirst,
        CrossFadeState.showFirst,
      ]);
    });

    testWidgets('not loaded (null docs): dashes, no header tappable',
        (tester) async {
      await _pump(tester, bank: null, investments: null, policies: null);
      expect(find.text('-'), findsNWidgets(4));
      await _expandAll(tester);
      expect(_fades(tester), everyElement(CrossFadeState.showFirst));
      expect(
        tester.widgetList<InkWell>(find.byType(InkWell)).map((w) => w.onTap),
        everyElement(isNull),
      );
    });

    testWidgets('Savings and Debts share the bank listener: load together',
        (tester) async {
      await _pump(tester, bank: null, investments: _investments, policies: null);
      await _expandAll(tester);
      expect(_fades(tester), [
        CrossFadeState.showFirst,
        CrossFadeState.showSecond,
        CrossFadeState.showFirst,
        CrossFadeState.showFirst,
      ]);
    });
  });

  group('Savings detail', () {
    testWidgets('grouped by bank, depository only, balance_current per row',
        (tester) async {
      await _pumpFull(tester);
      await _expand(tester, 'Savings');

      final texts = _bodyTexts(tester, 'Savings');
      // SavingsSummary.byInstitution's order; the loan-only bank drops out.
      expect(texts.where((t) => t.endsWith('Bank')),
          ['First Platypus Bank', 'Tartan Bank']);
      expect(texts, contains('•••• 4444 · Money market'));
      expect(texts, isNot(contains('•••• 3333 · Credit card')));
      // balance_current, not available: checking is $110 (available $100).
      expect(texts, contains('\$110'));
      expect(texts, isNot(contains('\$100')));
      // Within a bank, byInstitution keeps List.sort's order for equal names,
      // which Dart does not promise is stable — so compare as a multiset.
      expect(
        _money(tester, 'Savings'),
        unorderedEquals([
          '\$1,000', '\$110', '\$43,200', '\$210', '\$12,060', '\$6,009',
          '\$2,450',
        ]),
      );
    });

    testWidgets('rows sum to the header total', (tester) async {
      await _pumpFull(tester);
      await _expand(tester, 'Savings');

      // On the underlying figures the rows print: equal to the header's sum.
      final summary = SavingsSummary(
          accounts: _plaid(_bank), transactionsByAccount: const {});
      final rowSum = savingsByInstitution(summary)
          .values
          .expand((accounts) => accounts)
          .fold(0.0, (acc, a) => acc + a.balanceCurrent!);
      expect(rowSum, closeTo(summary.totalSavings, 1e-6));
      expect(rowSum, closeTo(65039.40, 1e-6));

      // On screen as well, for this data: the printed rows add to the
      // printed header exactly.
      final shown =
          _money(tester, 'Savings').map(_parse).reduce((a, b) => a + b);
      expect(shown, _parse(_headerTotal(tester, 'Savings').data!));
    });

    testWidgets('rounding: printed rows can miss the printed header by \$1',
        (tester) async {
      // Pinned so it is a known property, not a later "discovery": every
      // figure rounds to whole dollars on its own.
      await _pump(tester,
          bank: _docs({
            'a': _acct('Bank A', 'depository', 'savings', '1', 1200.5),
            'b': _acct('Bank A', 'depository', 'checking', '2', 5300.5),
          }),
          investments: const [],
          policies: const []);
      await _expand(tester, 'Savings');
      expect(_headerTotal(tester, 'Savings').data, '\$6,501');
      expect(_money(tester, 'Savings'), ['\$1,201', '\$5,301']); // = 6,502
    });

    testWidgets('null balance_current prints a dash, not \$0', (tester) async {
      await _pump(tester,
          bank: _docs({'a': _acct('Bank A', 'depository', 'savings', '1', null)}),
          investments: const [],
          policies: const []);
      await _expand(tester, 'Savings');
      expect(_money(tester, 'Savings'), ['-']);
    });
  });

  group('Investments detail', () {
    testWidgets('every holding, value-descending, value and % of portfolio',
        (tester) async {
      await _pumpFull(tester);
      await _expand(tester, 'Investments');
      expect(_bodyTexts(tester, 'Investments'), [
        'AAPL', 'Apple Inc.', '\$15,000', '61.2%',
        'EWZ', 'iShares MSCI Brazil ETF', '\$5,000', '20.4%',
        // No name: the bare ticker would repeat, so the account names it.
        'BTC', 'Plaid IRA', '\$4,500', '18.4%',
      ]);
    });
  });

  group('Investments cap', () {
    /// [count] holdings, spread over three accounts, worth $1,000..$count,000
    /// — distinct values so the order is unambiguous.
    _Docs holdings(int count) {
      final byAccount = <String, List<Map<String, dynamic>>>{};
      for (var i = 1; i <= count; i++) {
        byAccount.putIfAbsent('acct${i % 3}', () => []).add(
            {'ticker': 'H$i', 'name': 'Holding $i', 'market_value': i * 1000.0});
      }
      return _docs({
        for (final e in byAccount.entries)
          e.key: {
            'account_name': e.key,
            'total_value_sgd': e.value
                .fold(0.0, (acc, h) => acc + (h['market_value'] as double)),
            'holdings': e.value,
          },
      });
    }

    List<String> tickers(WidgetTester tester) => [
          for (final t in _bodyTexts(tester, 'Investments'))
            if (RegExp(r'^H\d+$').hasMatch(t)) t,
        ];

    testWidgets('more than 10: the top 10 by value, then "See all N"',
        (tester) async {
      var opened = 0;
      await _pump(tester,
          bank: const [],
          investments: holdings(13),
          policies: const [],
          onOpenInvestments: () => opened++);
      await _expand(tester, 'Investments');

      expect(tickers(tester),
          ['H13', 'H12', 'H11', 'H10', 'H9', 'H8', 'H7', 'H6', 'H5', 'H4']);
      expect(_bodyTexts(tester, 'Investments').last,
          'See all 13 holdings on Investments →');

      // Display cap only: shares are still of the whole $91,000 portfolio.
      final texts = _bodyTexts(tester, 'Investments');
      expect(texts[texts.indexOf('H13') + 3], '14.3%'); // 13,000 / 91,000
      expect(texts[texts.indexOf('H4') + 3], '4.4%'); // 4,000 / 91,000
      expect(_headerTotal(tester, 'Investments').data, '\$91,000');

      await tester.tap(find.text('See all 13 holdings on Investments →'));
      await tester.pump();
      expect(opened, 1);
    });

    testWidgets('exactly 10: every holding, no link', (tester) async {
      await _pump(tester,
          bank: const [],
          investments: holdings(10),
          policies: const [],
          onOpenInvestments: () {});
      await _expand(tester, 'Investments');
      expect(tickers(tester), hasLength(10));
      expect(find.textContaining('See all'), findsNothing);
    });

    testWidgets('fewer than 10 (the 3-holding fixture): no link',
        (tester) async {
      await _pumpFull(tester);
      await _expand(tester, 'Investments');
      expect(find.textContaining('See all'), findsNothing);
    });

    testWidgets('no navigation callback: the link is plain text, no arrow',
        (tester) async {
      await _pump(tester,
          bank: const [], investments: holdings(11), policies: const []);
      await _expand(tester, 'Investments');
      expect(_bodyTexts(tester, 'Investments').last,
          'See all 11 holdings on Investments');
      final link = find.ancestor(
          of: find.text('See all 11 holdings on Investments'),
          matching: find.byType(InkWell));
      expect(tester.widget<InkWell>(link.first).onTap, isNull);
    });
  });

  group('Insurance detail', () {
    testWidgets('each policy: name, type · insurer, sum assured; premiums',
        (tester) async {
      await _pumpFull(tester);
      await _expand(tester, 'Insurance Coverage');
      expect(_bodyTexts(tester, 'Insurance Coverage'), [
        'Term Life', 'Life · Insurer A', '\$500,000',
        // Null sum assured on personal accident is correct: a dash, not $0.
        'PA Plan', 'Personal Accident · Insurer B', '-',
        '\$1,500 a year in premiums',
      ]);
    });
  });

  group('Debts detail', () {
    testWidgets('every debt account across ALL institutions, largest first',
        (tester) async {
      await _pumpFull(tester, score: _scoreWithRatio(0.8703));
      await _expand(tester, 'Debts');

      final texts = _bodyTexts(tester, 'Debts');
      expect(texts.sublist(0, 18), [
        'First Platypus Bank', '•••• 4321 · Student', '\$65,262',
        'First Platypus Bank', '•••• 8888 · Mortgage', '\$56,302',
        'Houndstooth Bank', '•••• 5150 · Auto', '\$18,000',
        'First Platypus Bank', '•••• 9999 · Credit card', '\$5,020',
        'Tartan Bank', '•••• 7734 · Credit card', '\$1,290',
        'First Platypus Bank', '•••• 3333 · Credit card', '\$410',
      ]);
      // The card with available set reads balance_current, never its
      // remaining credit line ($4,980).
      expect(texts, isNot(contains('\$4,980')));

      final accounts = _plaid(_bank);
      expect(
        debtAccountsByOwed(accounts).fold(0.0, (acc, a) => acc + a.balanceOwed),
        closeTo(accounts.owedTotal, 1e-6),
      );
      expect(accounts.owedTotal, closeTo(146283.61, 1e-6));

      expect(
        texts.last,
        'Of the debt we can classify, 87% is structured (mortgage, student, '
        'auto, home equity) and 13% is revolving or unsecured',
      );
    });

    testWidgets('no banks connected: unknown — n/a, not \$0 or a dash',
        (tester) async {
      await _pump(tester,
          bank: const [], investments: const [], policies: const []);
      expect(_headerTotal(tester, 'Savings').data, '\$0');
      expect(_headerTotal(tester, 'Debts').data, 'n/a');
      expect(tester.widget<Text>(find.text('Debts')).style!.color,
          isNot(_red));
      await _expand(tester, 'Debts');
      expect(_fade(tester, 'Debts'), CrossFadeState.showSecond);
      expect(_bodyTexts(tester, 'Debts'), [
        'No banks connected yet, so we can\'t see any debt. Connect a bank '
            'from the Savings tab.',
      ]);
    });

    testWidgets('banks connected, no debt accounts: good news, a real \$0',
        (tester) async {
      await _pump(tester,
          bank: _docs({'a': _acct('Bank A', 'depository', 'savings', '1', 900)}),
          investments: const [],
          policies: const [],
          score: _scoreWithRatio(null));
      expect(_headerTotal(tester, 'Debts').data, '\$0');
      expect(_headerTotal(tester, 'Debts').style!.color, isNot(_red));
      await _expand(tester, 'Debts');
      // No split line where there is no debt to split.
      expect(_bodyTexts(tester, 'Debts'),
          ['No debt on your connected accounts, nice.']);
    });

    testWidgets('has debt: the accounts plus the split line', (tester) async {
      await _pump(tester,
          bank: _docs({'c': _acct('Bank A', 'credit', 'credit card', '7', 250)}),
          investments: const [],
          policies: const [],
          score: _scoreWithRatio(0.0));
      expect(_headerTotal(tester, 'Debts').data, '-\$250');
      await _expand(tester, 'Debts');
      expect(_bodyTexts(tester, 'Debts'), [
        'Bank A', '•••• 7 · Credit card', '\$250',
        // 0.0 is real — every classified dollar unhealthy — not missing.
        'Of the debt we can classify, 0% is structured (mortgage, student, '
            'auto, home equity) and 100% is revolving or unsecured',
      ]);
    });

    testWidgets('split line "not available" on every state without a ratio',
        (tester) async {
      const notAvailable = 'Debt split not available yet. It is worked out '
          'when your score next refreshes';
      final states = <String, ScoreState>{
        'Loading': const ScoreLoading(),
        'NeverComputed': const ScoreNeverComputed(),
        'Failed': const ScoreFailed('Could not load your score.'),
        'profile with a null ratio': _scoreWithRatio(null),
      };
      for (final entry in states.entries) {
        await _pumpFull(tester, score: entry.value);
        await _expand(tester, 'Debts');
        expect(_bodyTexts(tester, 'Debts').last, notAvailable,
            reason: entry.key);
        expect(find.textContaining('is structured'), findsNothing,
            reason: entry.key);
        // A fresh tree per state, so no open state carries over.
        await tester.pumpWidget(const SizedBox());
      }
    });
  });

  group('empty states and helpers', () {
    testWidgets('empty collections: existing copy plus a pointer to the tab',
        (tester) async {
      await _pump(tester,
          bank: const [], investments: const [], policies: const []);
      await _expandAll(tester);
      expect(_fades(tester), everyElement(CrossFadeState.showSecond));
      expect(_bodyTexts(tester, 'Savings'), [
        'No bank accounts connected yet. Connect a bank from the Savings tab.',
      ]);
      expect(_bodyTexts(tester, 'Investments'), [
        'No investment accounts connected yet. Connect a broker from the '
            'Investments tab.',
      ]);
      expect(_bodyTexts(tester, 'Insurance Coverage'), [
        'No policies uploaded yet. Upload one from the Insurance tab.',
      ]);
    });

    testWidgets('debt-only banks: Savings says so instead of "no banks"',
        (tester) async {
      await _pump(tester,
          bank: _docs({
            'card': _acct('Tartan Bank', 'credit', 'credit card', '7734', 410),
          }),
          investments: const [],
          policies: const []);
      await _expandAll(tester);
      expect(_bodyTexts(tester, 'Savings'),
          ['No savings or checking accounts connected yet']);
      expect(_bodyTexts(tester, 'Debts').first, 'Tartan Bank');
    });

    testWidgets('investment accounts with no holdings', (tester) async {
      await _pump(tester,
          bank: const [],
          investments: _docs({
            'a': {'account_name': 'Empty', 'total_value_sgd': 0.0},
          }),
          policies: const []);
      await _expand(tester, 'Investments');
      expect(_bodyTexts(tester, 'Investments'),
          ['No holdings in your connected investment accounts yet']);
    });

    testWidgets('hide balances masks every figure in every row',
        (tester) async {
      addTearDown(() => AppPreferences.instance.setHideBalances(false));
      await AppPreferences.instance.setHideBalances(true);
      await _pumpFull(tester, score: _scoreWithRatio(0.5));
      await _expandAll(tester);
      for (final label in _rows) {
        final all = [
          _headerTotal(tester, label).data!,
          ..._bodyTexts(tester, label),
        ];
        expect(all.where((t) => RegExp(r'\$\d').hasMatch(t)), isEmpty,
            reason: label);
      }
      expect(_headerTotal(tester, 'Debts').data, kMaskedAmount);
    });

    test('debtSplitLabel: one rounding, the two shares sum to 100%', () {
      ScoreProfile profile(double ratio) =>
          (_scoreWithRatio(ratio) as ScorePartial).profile;
      // 0.875 is exact in binary: 87.5 rounds half away from zero to 88.
      expect(debtSplitLabel(profile(0.875)), contains('88% is structured'));
      expect(debtSplitLabel(profile(0.875)), contains('12% is revolving'));
      expect(debtSplitLabel(profile(1.0)), contains('0% is revolving'));
      expect(debtSplitLabel(null), startsWith('Debt split not available'));
    });
  });
}
