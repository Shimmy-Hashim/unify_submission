import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:unify/models/home_score_state.dart';
import 'package:unify/models/score_history.dart';
import 'package:unify/models/score_profile.dart';
import 'package:unify/screens/home_screen.dart';
import 'package:unify/services/health_score_service.dart';
import 'package:unify/utils/app_styles.dart';
import 'package:unify/widgets/score_history_chart.dart';

final _now = DateTime.utc(2026, 9, 13, 10);

/// A snapshot shaped as `snapshot_payload()` writes it.
Map<String, dynamic> _doc(
  String date,
  double? score, {
  bool? capped,
  Map<String, int?> tiers = const {},
}) =>
    {
      'date': date,
      'overall_score': score,
      'score_breakdown': capped == null ? null : {'capped': capped},
      'domains': {
        for (final d in ['banking', 'investment', 'insurance'])
          d: {
            'status': tiers[d] == null ? 'not_assessed' : 'assessed',
            'tier_index': tiers[d],
          },
      },
    };

ScoreHistoryPoint _p(String date, double? score,
        {bool? capped, Map<String, int?> tiers = const {}}) =>
    ScoreHistoryPoint.fromMap(_doc(date, score, capped: capped, tiers: tiers));

ScoreState _complete(double score) {
  final at = Timestamp.fromDate(_now);
  return HealthScoreService.stateFromProfileData({
    'overall_score': score,
    'checklist_last_updated': at,
    'checklist_flags': {'layer1': const [], 'layer3': const []},
    'ml_profile': {
      'computed_at': at,
      'overall_score': score,
      'layer3_evaluated': true,
      'domains': const {},
      'score_breakdown': {
        'option_b_score': score,
        'ceiling': 100.0,
        'capped': false,
        'overall_score': score,
      },
    },
  });
}

Future<void> _pump(WidgetTester tester, ScoreState state,
    List<ScoreHistoryPoint>? history) async {
  tester.view.physicalSize = const Size(1000, 1400);
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(MaterialApp(
    theme: AppTheme.lightTheme,
    home: Scaffold(
      body: SizedBox(
        width: 900,
        child: SingleChildScrollView(
          child: ScoreHistoryCard(state: state, history: history, now: _now),
        ),
      ),
    ),
  ));
}

void main() {
  group('fromMap', () {
    test('reads capped and only ASSESSED tier indexes', () {
      final p = _p('2026-09-13', 39, capped: true, tiers: {
        'banking': 0,
        'investment': null,
        'insurance': 1,
      });
      expect(p.capped, isTrue);
      expect(p.domainTierIndex, {
        ScoreDomain.banking: 0,
        ScoreDomain.insurance: 1,
      });
    });

    test('a gap day has capped null, not false', () {
      expect(_p('2026-09-12', null).capped, isNull);
    });
  });

  group('currentCappedRun', () {
    test('not capped now -> null, whatever came before', () {
      expect(
        currentCappedRun([
          _p('2026-09-10', 39, capped: true),
          _p('2026-09-13', 80, capped: false),
        ]),
        isNull,
      );
    });

    test('newest day is a gap -> null (nothing capped to show)', () {
      expect(
        currentCappedRun([
          _p('2026-09-12', 39, capped: true),
          _p('2026-09-13', null),
        ]),
        isNull,
      );
    });

    test('run starts after the most recent explicit uncapped day', () {
      final run = currentCappedRun([
        _p('2026-09-01', 39, capped: true), // an older, separate run
        _p('2026-09-05', 80, capped: false),
        _p('2026-09-08', 39, capped: true),
        _p('2026-09-13', 39, capped: true),
      ])!;
      expect(run.since, DateTime.utc(2026, 9, 8));
      expect(run.startsAtEarliestLoaded, isFalse);
    });

    test('gap days do not end a run, and never become its start', () {
      final run = currentCappedRun([
        _p('2026-09-04', 80, capped: false),
        _p('2026-09-06', null),
        _p('2026-09-07', 39, capped: true),
        _p('2026-09-10', null),
        _p('2026-09-13', 39, capped: true),
      ])!;
      expect(run.since, DateTime.utc(2026, 9, 7));
    });

    test('whole history capped -> starts at earliest loaded', () {
      final run = currentCappedRun([
        _p('2026-09-13', 39, capped: true),
        _p('2026-09-11', 39, capped: true),
      ])!;
      expect(run.since, DateTime.utc(2026, 9, 11));
      expect(run.startsAtEarliestLoaded, isTrue);
    });

    test('capped since label: "at least" only when the window may be cut', () {
      final run = CappedRun(
          since: DateTime.utc(2026, 9, 2), startsAtEarliestLoaded: true);
      expect(cappedSinceLabel(run, windowFull: false, now: _now),
          'Capped by your insurance gap since 2 Sep');
      expect(cappedSinceLabel(run, windowFull: true, now: _now),
          'Capped by your insurance gap since at least 2 Sep');
      final bounded = CappedRun(
          since: DateTime.utc(2025, 12, 30), startsAtEarliestLoaded: false);
      expect(cappedSinceLabel(bounded, windowFull: true, now: _now),
          'Capped by your insurance gap since 30 Dec 2025');
    });
  });

  group('domainTrends', () {
    test('lower tier_index is better; earliest vs latest assessed', () {
      final trends = domainTrends([
        _p('2026-09-13', 67, tiers: {'banking': 0, 'investment': 1, 'insurance': 0}),
        _p('2026-09-01', 33, tiers: {'banking': 1, 'investment': 1, 'insurance': 0}),
        _p('2026-09-05', 50, tiers: {'banking': 0, 'investment': 0, 'insurance': 1}),
      ]);
      expect(trends[ScoreDomain.banking], DomainTrend.improved);
      expect(trends[ScoreDomain.investment], DomainTrend.unchanged);
      expect(trends[ScoreDomain.insurance], DomainTrend.unchanged);
    });

    test('worse tier reads worsened', () {
      final trends = domainTrends([
        _p('2026-09-01', 70, tiers: {'insurance': 0}),
        _p('2026-09-13', 39, tiers: {'insurance': 1}),
      ]);
      expect(trends[ScoreDomain.insurance], DomainTrend.worsened);
    });

    test('one snapshot, or a domain assessed only once -> not enough', () {
      final one = domainTrends([_p('2026-09-13', 67, tiers: {'banking': 0})]);
      expect(one.values.toSet(), {DomainTrend.notEnoughHistory});

      final partial = domainTrends([
        _p('2026-09-01', null, tiers: {'banking': 1}),
        _p('2026-09-13', 67, tiers: {'banking': 0, 'investment': 0}),
      ]);
      expect(partial[ScoreDomain.banking], DomainTrend.improved);
      expect(partial[ScoreDomain.investment], DomainTrend.notEnoughHistory);
    });

    test('assessed throughout: earliest vs latest, gaps in between skipped',
        () {
      final trends = domainTrends([
        _p('2026-09-01', 40, tiers: {'banking': 1}),
        _p('2026-09-05', null),
        _p('2026-09-13', 70, tiers: {'banking': 0}),
      ]);
      expect(trends[ScoreDomain.banking], DomainTrend.improved);
    });

    test('latest day unassessed -> unavailable, even with 2+ earlier days',
        () {
      final trends = domainTrends([
        _p('2026-09-01', 40, tiers: {'banking': 1, 'insurance': 0}),
        _p('2026-09-05', 55, tiers: {'banking': 0, 'insurance': 0}),
        _p('2026-09-13', 60, tiers: {'insurance': 1}),
      ]);
      expect(trends[ScoreDomain.banking], DomainTrend.notEnoughHistory);
      // Same window, latest day assessed: the comparison still runs.
      expect(trends[ScoreDomain.insurance], DomainTrend.worsened);
    });

    test('only one assessed day -> unavailable', () {
      final trends = domainTrends([
        _p('2026-09-01', null),
        _p('2026-09-13', 67, tiers: {'banking': 0}),
      ]);
      expect(trends[ScoreDomain.banking], DomainTrend.notEnoughHistory);
    });

    test('empty list -> unavailable for every domain', () {
      final trends = domainTrends(const []);
      expect(trends.keys.toSet(), ScoreDomain.values.toSet());
      expect(trends.values.toSet(), {DomainTrend.notEnoughHistory});
    });
  });

  group('scoreHistorySummary — shares the score card trend', () {
    test('trend present: the score card\'s own change text', () {
      final s = scoreHistorySummary(
        state: _complete(67),
        history: [_p('2026-09-11', 56), _p('2026-09-12', null), _p('2026-09-13', 67)],
        now: _now,
      );
      expect(s.text, '↑ +11 pts');
      expect(s.trend!.deltaPoints, 11);
    });

    test('no trend: counts recorded days instead', () {
      expect(
        scoreHistorySummary(
                state: _complete(67), history: [_p('2026-09-13', 67)], now: _now)
            .text,
        '1 day recorded',
      );
      expect(
        scoreHistorySummary(
                state: const ScoreNeverComputed(), history: const [], now: _now)
            .text,
        'No history yet',
      );
    });
  });

  group('ScoreHistoryCard', () {
    testWidgets('collapsed by default; tap expands and collapses',
        (tester) async {
      await _pump(tester, _complete(67), [
        _p('2026-09-11', 56, capped: false, tiers: {'banking': 1}),
        _p('2026-09-13', 67, capped: false, tiers: {'banking': 0}),
      ]);
      expect(find.text('Score History'), findsOneWidget);
      expect(find.text('↑ +11 pts'), findsOneWidget);
      CrossFadeState fade() => tester
          .widget<AnimatedCrossFade>(find.byType(AnimatedCrossFade))
          .crossFadeState;
      expect(fade(), CrossFadeState.showFirst);

      await tester.tap(find.text('Score History'));
      await tester.pumpAndSettle();
      expect(fade(), CrossFadeState.showSecond);
      expect(find.byType(ScoreHistoryChart), findsOneWidget);
      expect(find.text('Better'), findsOneWidget);
      expect(find.textContaining('Capped by'), findsNothing);

      await tester.tap(find.text('Score History'));
      await tester.pumpAndSettle();
      expect(fade(), CrossFadeState.showFirst);
    });

    testWidgets('capped now: callout with the run start', (tester) async {
      await _pump(tester, _complete(39), [
        _p('2026-09-05', 80, capped: false),
        _p('2026-09-09', 39, capped: true),
        _p('2026-09-13', 39, capped: true),
      ]);
      await tester.tap(find.text('Score History'));
      await tester.pumpAndSettle();
      expect(find.text('Capped by your insurance gap since 9 Sep'),
          findsOneWidget);
    });

    testWidgets('single snapshot: chart, a note, dashes for domains',
        (tester) async {
      await _pump(tester, _complete(67), [
        _p('2026-09-13', 67, capped: false, tiers: {'banking': 0}),
      ]);
      expect(find.text('1 day recorded'), findsOneWidget);
      await tester.tap(find.text('Score History'));
      await tester.pumpAndSettle();
      expect(find.byType(ScoreHistoryChart), findsOneWidget);
      expect(find.textContaining('One scored day so far'), findsOneWidget);
      expect(find.text('-'), findsNWidgets(3));
    });

    testWidgets('history still loading: row not tappable', (tester) async {
      await _pump(tester, const ScoreLoading(), null);
      await tester.tap(find.text('Score History'));
      await tester.pumpAndSettle();
      expect(
        tester
            .widget<AnimatedCrossFade>(find.byType(AnimatedCrossFade))
            .crossFadeState,
        CrossFadeState.showFirst,
      );
    });
  });
}
