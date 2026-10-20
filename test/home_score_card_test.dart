import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:unify/models/home_score_state.dart';
import 'package:unify/models/score_history.dart';
import 'package:unify/screens/home_screen.dart';
import 'package:unify/services/health_score_service.dart';
import 'package:unify/utils/app_styles.dart';
import 'package:unify/widgets/tier_segment_bar.dart';

ScoreHistoryPoint _point(String date, double? score) =>
    ScoreHistoryPoint(date: parseSnapshotDate(date), overallScore: score);

/// A computation at 10:00 UTC on 13 Sep 2026 — snapshot day 2026-09-13.
final _now = DateTime.utc(2026, 9, 13, 10);

ScoreTrend? _trend(double current, List<ScoreHistoryPoint> history,
        {DateTime? at}) =>
    computeScoreTrend(
      currentScore: current,
      currentComputedAt: at ?? _now,
      history: history,
    );

const _assessed = {
  'banking': {
    'status': 'assessed',
    'tier_index': 0,
    'tier_label': 'Stable',
    'reason': null,
    'missing_features': <String>[],
  },
  'investment': {
    'status': 'assessed',
    'tier_index': 0,
    'tier_label': 'On Track',
    'reason': null,
    'missing_features': <String>[],
  },
  'insurance': {
    'status': 'assessed',
    'tier_index': 1,
    'tier_label': 'Not Covered',
    'reason': null,
    'missing_features': <String>[],
  },
};

/// A complete cached run shaped as `cache_payload()` writes it, with the
/// breakdown `compute_score_breakdown()` produces for these inputs.
ScoreComplete _complete({
  required double overall,
  required double optionB,
  required double ceiling,
  required int cleared,
}) {
  final at = Timestamp.fromDate(_now);
  final state = HealthScoreService.stateFromProfileData({
    'overall_score': overall,
    'checklist_last_updated': at,
    'checklist_flags': {'layer1': const [], 'layer3': const []},
    'ml_profile': {
      'schema_version': '1.2.0',
      'computed_at': at,
      'age_band': 2,
      'overall_score': overall,
      'layer3_flags': const <String>[],
      'layer3_evaluated': true,
      'domains': _assessed,
      'score_breakdown': {
        'banking_score': 100.0,
        'investment_score': 100.0,
        'insurance_indicators': [
          for (var i = 0; i < 3; i++) i < cleared ? 1.0 : 0.0,
        ],
        'insurance_indicators_cleared': cleared,
        'insurance_third': cleared / 3 * 100,
        'option_b_score': optionB,
        'ceiling': ceiling,
        'capped': optionB > ceiling,
        'overall_score': overall,
      },
    },
  });
  return state as ScoreComplete;
}

Future<void> _pumpCard(
  WidgetTester tester,
  ScoreState state, {
  List<ScoreHistoryPoint>? history,
  bool refreshing = false,
  Future<void> Function()? onRerun,
  VoidCallback? onOpenInsurance,
}) async {
  // Wide surface: Ahem glyphs are ~2x Inter's width (see
  // home_domain_status_test.dart), so a phone width overflows only in tests.
  tester.view.physicalSize = const Size(1000, 1000);
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(MaterialApp(
    theme: AppTheme.lightTheme,
    home: Scaffold(
      body: SizedBox(
        width: 900,
        child: HealthScoreCard(
          state: state,
          presence: const AccountPresence(
              hasBankAccounts: true, hasInvestmentAccounts: true),
          history: history,
          refreshing: refreshing,
          onRerun: onRerun,
          onOpenInsurance: onOpenInsurance,
          now: _now,
        ),
      ),
    ),
  ));
}

void main() {
  group('computeScoreTrend', () {
    test('no history at all -> no trend', () {
      expect(_trend(72, const []), isNull);
    });

    test('only one snapshot ever (today\'s own) -> no trend', () {
      expect(_trend(72, [_point('2026-09-13', 72)]), isNull);
    });

    test('compares against the most recent earlier real score', () {
      final trend = _trend(72, [
        _point('2026-09-13', 72),
        _point('2026-09-12', 68),
        _point('2026-09-01', 50),
      ])!;
      expect(trend.deltaPoints, 4);
      expect(trend.since, DateTime.utc(2026, 9, 12));
    });

    test('skips null gap days rather than diffing against yesterday', () {
      final trend = _trend(72, [
        _point('2026-09-13', 72),
        _point('2026-09-12', null),
        _point('2026-09-11', null),
        _point('2026-09-05', 75),
      ])!;
      expect(trend.deltaPoints, -3);
      expect(trend.since, DateTime.utc(2026, 9, 5));
    });

    test('a history of only gap days -> no trend', () {
      expect(
        _trend(72, [
          _point('2026-09-13', 72),
          _point('2026-09-12', null),
          _point('2026-09-10', null),
        ]),
        isNull,
      );
    });

    test('order of the input does not matter', () {
      final trend = _trend(40, [
        _point('2026-08-01', 10),
        _point('2026-09-10', 39),
        _point('2026-09-13', 40),
        _point('2026-09-02', 20),
      ])!;
      expect(trend.since, DateTime.utc(2026, 9, 10));
      expect(trend.deltaPoints, 1);
    });

    test('delta is between rounded scores, matching the numbers on screen',
        () {
      // 71.6 shows as 72, 68.4 shows as 68: the trend must read +4, not +3.
      expect(_trend(71.6, [_point('2026-09-12', 68.4)])!.deltaPoints, 4);
    });

    test('no change is a real trend of zero, not an absent one', () {
      expect(_trend(69, [_point('2026-09-12', 69)])!.deltaPoints, 0);
    });

    test('snapshots dated on or after the current day are never "prior"', () {
      expect(
        _trend(72, [_point('2026-09-13', 60), _point('2026-09-14', 50)]),
        isNull,
      );
    });

    test('current day is the UTC day, the same rule that keys the documents',
        () {
      // 07:00 SGT on 13 Sep is 23:00 UTC on 12 Sep — that run's snapshot is
      // the 12 Sep document, so 12 Sep is "today" and 11 Sep is the prior.
      final at = DateTime.utc(2026, 9, 12, 23);
      final trend = _trend(
        72,
        [_point('2026-09-12', 72), _point('2026-09-11', 70)],
        at: at,
      )!;
      expect(trend.since, DateTime.utc(2026, 9, 11));
      expect(trend.deltaPoints, 2);
    });

    test('malformed snapshot dates are ignored, not thrown on', () {
      final points = [
        ScoreHistoryPoint.fromMap({'date': '13/09/2026', 'overall_score': 10}),
        ScoreHistoryPoint.fromMap({'date': null, 'overall_score': 10}),
        ScoreHistoryPoint.fromMap({'date': '2026-09-01', 'overall_score': 60}),
      ];
      expect(_trend(72, points)!.since, DateTime.utc(2026, 9, 1));
    });

    test('fromMap keeps a null overall_score as null, never 0', () {
      final p = ScoreHistoryPoint.fromMap(
          {'date': '2026-09-12', 'overall_score': null});
      expect(p.overallScore, isNull);
      expect(p.date, DateTime.utc(2026, 9, 12));
    });
  });

  group('copy', () {
    test('trend label: two lines — change, then date', () {
      final now = DateTime(2026, 9, 13);
      (String, String) label(int d, DateTime since) {
        final l = scoreTrendLabel(
            ScoreTrend(deltaPoints: d, since: since), now: now);
        return (l.change, l.since);
      }
      final sep2 = DateTime.utc(2026, 9, 2);
      expect(label(4, sep2), ('↑ +4 pts', 'since 2 Sep'));
      expect(label(-3, sep2), ('↓ -3 pts', 'since 2 Sep'));
      expect(label(1, sep2), ('↑ +1 pt', 'since 2 Sep'));
      expect(label(0, sep2), ('No change', 'since 2 Sep'));
      expect(label(5, DateTime.utc(2025, 12, 30)),
          ('↑ +5 pts', 'since 30 Dec 2025'));
      // The widest reachable strings (score range 0-100, other-year date) —
      // measured at 60px and 102px in Inter 600 11px against the 126px column.
      expect(label(-100, DateTime.utc(2025, 5, 30)),
          ('↓ -100 pts', 'since 30 May 2025'));
    });

    test('last calculated label', () {
      final now = DateTime(2026, 9, 13, 18);
      expect(lastCalculatedLabel(DateTime(2026, 9, 13, 1), now: now),
          'Last calculated today');
      expect(lastCalculatedLabel(DateTime(2026, 9, 12, 23), now: now),
          'Last calculated yesterday');
      expect(lastCalculatedLabel(DateTime(2026, 9, 2, 9), now: now),
          'Last calculated on 2 Sep 2026');
    });

    test('capped message is null when uncapped', () {
      final b = _complete(overall: 85, optionB: 85, ceiling: 100, cleared: 2)
          .breakdown;
      expect(cappedScoreMessage(b), isNull);
    });

    test('capped message reads the cached breakdown', () {
      final b = _complete(overall: 39, optionB: 66.7, ceiling: 39, cleared: 0)
          .breakdown;
      final m = cappedScoreMessage(b)!;
      expect(m.headline, 'Score capped by your insurance gap');
      expect(m.checks, '0 of 3 checks met');
      expect(m.uncapped, 'Uncapped 67');
    });

    test('score colour follows the 39/69 bands using the bar\'s own fills', () {
      expect(TierSegmentBar.scoreColor(39), const Color(0xFFEF4444));
      expect(TierSegmentBar.scoreColor(40), const Color(0xFFF59E0B));
      expect(TierSegmentBar.scoreColor(69.4), const Color(0xFFF59E0B));
      expect(TierSegmentBar.scoreColor(70), const Color(0xFF16A34A));
    });
  });

  group('HealthScoreCard', () {
    testWidgets('capped: coloured number, tappable message opens Insurance',
        (tester) async {
      var opened = 0;
      await _pumpCard(
        tester,
        _complete(overall: 39, optionB: 66.7, ceiling: 39, cleared: 0),
        onOpenInsurance: () => opened++,
      );
      final number = tester.widget<Text>(find.text('39'));
      expect(number.style!.color, const Color(0xFFEF4444));
      expect(find.text('Stable'), findsWidgets); // domain sub-labels only
      expect(find.text('At Risk'), findsNothing); // no band word

      expect(find.text('0 of 3 checks met'), findsOneWidget);
      expect(find.text('Uncapped 67'), findsOneWidget);
      await tester.tap(find.textContaining('Score capped by your insurance'));
      expect(opened, 1);
      // Tapping the detail lines lands on the same target.
      await tester.tap(find.text('Uncapped 67'));
      expect(opened, 2);
    });

    testWidgets('uncapped: no capped message at all', (tester) async {
      await _pumpCard(
        tester,
        _complete(overall: 85, optionB: 85, ceiling: 100, cleared: 2),
        onOpenInsurance: () {},
      );
      expect(find.text('85'), findsOneWidget);
      expect(find.textContaining('capped'), findsNothing);
    });

    testWidgets('trend shown against real prior history', (tester) async {
      await _pumpCard(
        tester,
        _complete(overall: 85, optionB: 85, ceiling: 100, cleared: 2),
        history: [_point('2026-09-13', 85), _point('2026-09-10', 81)],
      );
      // Two separate Texts, never one string left to wrap.
      expect(find.text('↑ +4 pts'), findsOneWidget);
      expect(find.text('since 10 Sep'), findsOneWidget);
    });

    testWidgets('only one snapshot: no trend line rendered', (tester) async {
      await _pumpCard(
        tester,
        _complete(overall: 85, optionB: 85, ceiling: 100, cleared: 2),
        history: [_point('2026-09-13', 85)],
      );
      expect(find.textContaining('since'), findsNothing);
      expect(find.textContaining('pts'), findsNothing);
    });

    testWidgets('Re-run calls the refresh trigger; disabled while refreshing',
        (tester) async {
      var calls = 0;
      Future<void> rerun() async => calls++;
      final state =
          _complete(overall: 85, optionB: 85, ceiling: 100, cleared: 2);

      await _pumpCard(tester, state, onRerun: rerun);
      expect(find.text('Last calculated today'), findsOneWidget);
      await tester.tap(find.text('Re-run Score'));
      expect(calls, 1);

      await _pumpCard(tester, state, onRerun: rerun, refreshing: true);
      expect(find.text('Re-running…'), findsOneWidget);
      await tester.tap(find.text('Re-running…'));
      expect(calls, 1);
    });
  });
}
