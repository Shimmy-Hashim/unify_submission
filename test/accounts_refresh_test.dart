import 'dart:async';
import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:unify/models/home_score_state.dart';
import 'package:unify/models/investment_holding.dart';
import 'package:unify/models/plaid_account.dart';
import 'package:unify/models/plaid_refresh_result.dart';
import 'package:unify/services/health_score_service.dart';
import 'package:unify/services/home_score_controller.dart';
import 'package:unify/services/plaid_service.dart';
import 'package:unify/utils/formatters.dart';
import 'package:unify/widgets/accounts_refresh.dart';

class _FakeScoreService extends HealthScoreService {
  _FakeScoreService() : super.forTesting();

  final controller = StreamController<ScoreState>.broadcast();
  int refreshCalls = 0;

  @override
  Stream<ScoreState> watch(String uid) => controller.stream;

  @override
  Future<void> requestRefresh(
      {Duration timeout = const Duration(seconds: 45)}) async {
    refreshCalls++;
  }
}

/// Stands in for PlaidService.refresh: counts calls, and can be held open
/// with [gate] to simulate a refresh in flight.
class _FakePlaidRefresh {
  int calls = 0;
  Completer<PlaidRefreshResult>? gate;
  PlaidRefreshResult result =
      const PlaidRefreshResult(status: PlaidRefreshStatus.refreshed);
  Object? error;

  Future<PlaidRefreshResult> call() async {
    calls++;
    final g = gate;
    if (g != null) return g.future;
    final e = error;
    if (e != null) throw e;
    return result;
  }
}

PlaidRefreshResult _result(String status, [List<Map<String, dynamic>>? items]) =>
    PlaidRefreshResult.fromJson({'status': status, 'items': items ?? []});

Map<String, dynamic> _failedItem(String? bank) => {
      'item_id': 'item-${bank ?? 'none'}-${bank.hashCode}',
      'institution_name': bank,
      'status': 'failed',
      'error_code': 'ITEM_LOGIN_REQUIRED',
    };

Widget _pullHost(Future<PlaidRefreshResult?> Function() onRefresh) =>
    MaterialApp(
      home: Scaffold(
        body: AccountsRefreshIndicator(
          onRefresh: onRefresh,
          child: ListView(
            physics: const AlwaysScrollableScrollPhysics(),
            children: const [SizedBox(height: 80, child: Text('short list'))],
          ),
        ),
      ),
    );

Future<void> _pull(WidgetTester tester) async {
  await tester.fling(find.byType(ListView), const Offset(0, 300), 1000);
  // Let the indicator settle into its refresh state without waiting for the
  // spinner, which animates forever while the refresh is held open.
  for (var i = 0; i < 5; i++) {
    await tester.pump(const Duration(milliseconds: 200));
  }
}

void main() {
  // -------------------------------------------------------------------------
  group('PlaidRefreshResult.fromJson', () {
    test('parses every status', () {
      const expected = {
        'refreshed': PlaidRefreshStatus.refreshed,
        'partial': PlaidRefreshStatus.partial,
        'failed': PlaidRefreshStatus.failed,
        'skipped_recent': PlaidRefreshStatus.skippedRecent,
        'no_connections': PlaidRefreshStatus.noConnections,
        'something_new': PlaidRefreshStatus.unknown,
      };
      expected.forEach((raw, status) {
        expect(PlaidRefreshResult.fromJson({'status': raw}).status, status,
            reason: raw);
      });
    });

    test('a full response parses every field', () {
      final r = PlaidRefreshResult.fromJson({
        'status': 'partial',
        'refreshed_at': '2026-10-10T08:44:22.436000+00:00',
        'items': [
          {
            'item_id': 'a',
            'institution_name': 'First Platypus Bank',
            'status': 'refreshed',
            'error_code': null,
            'transactions': {'added': 3, 'modified': 1, 'removed': 2},
            'accounts_updated': 12,
            'investments_synced': true,
          },
          _failedItem('Tartan Bank'),
        ],
        'financial_profile': {'net_worth_sgd': 1.0},
        'score': {'overall_score': 50},
      });
      expect(r.status, PlaidRefreshStatus.partial);
      expect(r.refreshedAt, DateTime.utc(2026, 10, 10, 8, 44, 22, 436));
      expect(r.items, hasLength(2));
      final a = r.items.first;
      expect([a.itemId, a.institutionName, a.refreshed, a.errorCode],
          ['a', 'First Platypus Bank', true, null]);
      expect([a.added, a.modified, a.removed, a.accountsUpdated],
          [3, 1, 2, 12]);
      expect(a.investmentsSynced, isTrue);
      expect(r.failedItems.single.errorCode, 'ITEM_LOGIN_REQUIRED');
      expect(r.failedItems.single.institutionName, 'Tartan Bank');
    });

    test('missing fields default rather than throw', () {
      final empty = PlaidRefreshResult.fromJson({});
      expect(empty.status, PlaidRefreshStatus.unknown);
      expect(empty.refreshedAt, isNull);
      expect(empty.items, isEmpty);

      final bare = PlaidRefreshResult.fromJson({
        'status': 'refreshed',
        'items': [<String, dynamic>{}],
      }).items.single;
      expect(bare.itemId, '');
      expect(bare.institutionName, isNull);
      expect(bare.refreshed, isFalse);
      expect([bare.added, bare.modified, bare.removed, bare.accountsUpdated],
          [0, 0, 0, 0]);
      expect(bare.investmentsSynced, isFalse);
    });

    test('extra and malformed fields are ignored', () {
      final r = PlaidRefreshResult.fromJson({
        'status': 'skipped_recent',
        'last_synced': '2026-10-10T08:44:22+00:00',
        'cooldown_seconds': 30,
        'refreshed_at': 'not a date',
        'items': [
          'not a map',
          42,
          {'item_id': 'x', 'transactions': 'not a map', 'surprise': [1, 2]},
        ],
      });
      expect(r.status, PlaidRefreshStatus.skippedRecent);
      expect(r.refreshedAt, isNull);
      expect(r.items.single.itemId, 'x');
      expect(r.items.single.added, 0);
    });

    test('items is not a list: no items, no throw', () {
      expect(
          PlaidRefreshResult.fromJson({'status': 'failed', 'items': {}}).items,
          isEmpty);
    });
  });

  // -------------------------------------------------------------------------
  group('refreshMessage', () {
    test('every status', () {
      expect(refreshMessage(result: _result('refreshed')), 'Updated');
      expect(refreshMessage(result: _result('skipped_recent')),
          'Already up to date');
      expect(refreshMessage(result: _result('no_connections')),
          'No connected accounts to refresh');
      expect(refreshMessage(result: _result('failed')),
          "Couldn't refresh your accounts. Try again later.");
      expect(refreshMessage(result: _result('something_new')),
          "Couldn't refresh your accounts. Try again later.");
    });

    test('partial names the one failed bank', () {
      expect(
          refreshMessage(
              result: _result('partial', [
            {'item_id': 'ok', 'institution_name': 'Good Bank', 'status': 'refreshed'},
            _failedItem('Tartan Bank'),
          ])),
          "Couldn't refresh Tartan Bank. In Settings, disconnect it, then connect it again.");
    });

    test('two failed Items at one bank name it once', () {
      expect(
          refreshMessage(
              result: _result('partial',
                  [_failedItem('Tartan Bank'), _failedItem('Tartan Bank')])),
          "Couldn't refresh Tartan Bank. In Settings, disconnect it, then connect it again.");
    });

    test('several failed banks are listed and take "them"', () {
      expect(
          refreshMessage(
              result: _result('partial',
                  [_failedItem('Tartan Bank'), _failedItem('Platypus Bank')])),
          "Couldn't refresh Tartan Bank and Platypus Bank. "
          'In Settings, disconnect them, then connect them again.');
      expect(
          refreshMessage(
              result: _result('partial', [
            _failedItem('A Bank'),
            _failedItem('B Bank'),
            _failedItem('C Bank'),
          ])),
          "Couldn't refresh A Bank, B Bank and C Bank. "
          'In Settings, disconnect them, then connect them again.');
    });

    test('a failed Item without an institution name still reads', () {
      expect(refreshMessage(result: _result('partial', [_failedItem(null)])),
          "Couldn't refresh one of your banks. In Settings, disconnect it, then connect it again.");
    });

    test('a timeout or network error is a delay, not a failure', () {
      final msg = refreshMessage(
          error: PlaidServiceException('x', isNetwork: true));
      expect(msg, 'Taking longer than expected. Pull to refresh again shortly.');
      expect(msg.toLowerCase(), isNot(contains('fail')));
      expect(msg.toLowerCase(), isNot(contains("couldn't")));
    });

    test('server errors', () {
      expect(refreshMessage(error: PlaidServiceException('x', statusCode: 401)),
          'Your session has expired. Sign in again.');
      expect(refreshMessage(error: PlaidServiceException('x', statusCode: 500)),
          "Couldn't refresh your accounts. Try again later.");
      expect(refreshMessage(error: const FormatException('bad json')),
          "Couldn't refresh your accounts. Try again later.");
    });

    test('no user-facing refresh string contains an em dash', () {
      final messages = [
        for (final s in [
          'refreshed', 'partial', 'failed', 'skipped_recent', 'no_connections'
        ])
          refreshMessage(result: _result(s, [_failedItem('X'), _failedItem('Y')])),
        refreshMessage(error: PlaidServiceException('x', isNetwork: true)),
        refreshMessage(error: PlaidServiceException('x', statusCode: 401)),
        refreshMessage(error: Exception('x')),
        updatedLabel(DateTime(2026, 10, 10, 9), now: DateTime(2026, 10, 10, 15)),
      ];
      for (final m in messages) {
        expect(m, isNot(contains('—')), reason: m);
      }
    });
  });

  // -------------------------------------------------------------------------
  group('updatedLabel', () {
    final now = DateTime(2026, 10, 10, 15, 42); // local 3:42 pm

    String at(Duration ago) => updatedLabel(now.subtract(ago), now: now);

    test('under a minute is just now, including clock skew', () {
      expect(at(Duration.zero), 'Updated just now');
      expect(at(const Duration(seconds: 59)), 'Updated just now');
      expect(at(const Duration(seconds: -20)), 'Updated just now');
    });

    test('minutes under an hour', () {
      expect(at(const Duration(seconds: 60)), 'Updated 1 min ago');
      expect(at(const Duration(minutes: 5)), 'Updated 5 min ago');
      expect(at(const Duration(minutes: 59, seconds: 59)), 'Updated 59 min ago');
    });

    test('an hour or more, same day: the clock time', () {
      expect(at(const Duration(hours: 1)), 'Updated today 2:42 pm');
      expect(updatedLabel(DateTime(2026, 10, 10, 12, 5), now: now),
          'Updated today 12:05 pm');
      expect(updatedLabel(DateTime(2026, 10, 10, 0, 7), now: now),
          'Updated today 12:07 am');
      expect(updatedLabel(DateTime(2026, 10, 10, 9, 30), now: now),
          'Updated today 9:30 am');
    });

    test('yesterday (even under 24 h) and earlier: the date', () {
      expect(updatedLabel(DateTime(2026, 10, 9, 23, 59), now: now),
          'Updated 9 Oct');
      expect(updatedLabel(DateTime(2026, 1, 3, 8), now: now), 'Updated 3 Jan');
      // Under an hour old still reads in minutes across midnight.
      expect(
          updatedLabel(DateTime(2026, 10, 9, 23, 50),
              now: DateTime(2026, 10, 10, 0, 10)),
          'Updated 20 min ago');
    });

    test('an earlier year carries the year', () {
      expect(updatedLabel(DateTime(2025, 12, 31, 22), now: now),
          'Updated 31 Dec 2025');
    });
  });

  // -------------------------------------------------------------------------
  group('newest last_synced on the summaries', () {
    PlaidAccount acct(String id, DateTime? synced) => PlaidAccount(
        accountId: id,
        itemId: 'i',
        accountName: id,
        accountType: 'depository',
        lastSynced: synced);

    test('SavingsSummary.lastSynced is the newest, null when none', () {
      final newest = DateTime.utc(2026, 10, 10, 8);
      expect(
          SavingsSummary(accounts: [
            acct('a', DateTime.utc(2026, 10, 9)),
            acct('b', newest),
            acct('c', null),
          ], transactionsByAccount: const {})
              .lastSynced,
          newest);
      expect(
          SavingsSummary(
                  accounts: [acct('a', null)], transactionsByAccount: const {})
              .lastSynced,
          isNull);
    });

    test('InvestmentAccount reads last_synced; PortfolioSummary takes newest',
        () {
      final older = DateTime.utc(2026, 10, 1);
      final newer = DateTime.utc(2026, 10, 10, 8);
      final a = InvestmentAccount.fromFirestore(
          'a', {'last_synced': Timestamp.fromDate(older)});
      final b = InvestmentAccount.fromFirestore(
          'b', {'last_synced': Timestamp.fromDate(newer)});
      final c = InvestmentAccount.fromFirestore('c', {});
      expect(a.lastSynced, older.toLocal());
      expect(c.lastSynced, isNull);
      expect(PortfolioSummary(accounts: [a, b, c]).lastSynced,
          newer.toLocal());
      expect(PortfolioSummary(accounts: [c]).lastSynced, isNull);
    });
  });

  // -------------------------------------------------------------------------
  group('AccountsRefreshIndicator (shared by Home, Savings, Investments)', () {
    testWidgets('a pull calls refresh once and shows the message',
        (tester) async {
      final fake = _FakePlaidRefresh();
      await tester.pumpWidget(_pullHost(fake.call));
      await _pull(tester);
      await tester.pumpAndSettle();
      expect(fake.calls, 1);
      expect(find.text('Updated'), findsOneWidget);
    });

    for (final (status, message) in [
      ('skipped_recent', 'Already up to date'),
      ('no_connections', 'No connected accounts to refresh'),
      ('failed', "Couldn't refresh your accounts. Try again later."),
    ]) {
      testWidgets('status $status shows "$message"', (tester) async {
        final fake = _FakePlaidRefresh()..result = _result(status);
        await tester.pumpWidget(_pullHost(fake.call));
        await _pull(tester);
        await tester.pumpAndSettle();
        expect(find.text(message), findsOneWidget);
      });
    }

    testWidgets('partial names the failed bank', (tester) async {
      final fake = _FakePlaidRefresh()
        ..result = _result('partial', [_failedItem('Tartan Bank')]);
      await tester.pumpWidget(_pullHost(fake.call));
      await _pull(tester);
      await tester.pumpAndSettle();
      expect(find.text("Couldn't refresh Tartan Bank. In Settings, disconnect it, then connect it again."),
          findsOneWidget);
    });

    testWidgets('a timeout shows the delay message', (tester) async {
      final fake = _FakePlaidRefresh()
        ..error = PlaidServiceException('timed out', isNetwork: true);
      await tester.pumpWidget(_pullHost(fake.call));
      await _pull(tester);
      await tester.pumpAndSettle();
      expect(
          find.text('Taking longer than expected. Pull to refresh again shortly.'),
          findsOneWidget);
    });

    testWidgets('a second pull while one is in flight does nothing',
        (tester) async {
      final fake = _FakePlaidRefresh()..gate = Completer();
      await tester.pumpWidget(_pullHost(fake.call));
      await _pull(tester);
      expect(fake.calls, 1);
      await _pull(tester);
      await _pull(tester);
      expect(fake.calls, 1);
      fake.gate!.complete(_result('refreshed'));
      await tester.pumpAndSettle();
      expect(fake.calls, 1);
      expect(find.text('Updated'), findsOneWidget);
    });

    testWidgets('a null result (caller declined) shows nothing',
        (tester) async {
      var calls = 0;
      await tester.pumpWidget(_pullHost(() async {
        calls++;
        return null;
      }));
      await _pull(tester);
      await tester.pumpAndSettle();
      expect(calls, 1);
      expect(find.byType(SnackBar), findsNothing);
    });
  });

  // -------------------------------------------------------------------------
  group('Home: pull vs Re-run Score', () {
    testWidgets('pull calls the account refresh once, never /score/refresh',
        (tester) async {
      final score = _FakeScoreService();
      final plaid = _FakePlaidRefresh();
      final controller = HomeScoreController(
          uid: 'u1', service: score, plaidRefresh: plaid.call);
      addTearDown(controller.dispose);

      await tester.pumpWidget(_pullHost(controller.refreshAll));
      await _pull(tester);
      await tester.pumpAndSettle();
      expect(plaid.calls, 1);
      expect(score.refreshCalls, 0);
      expect(find.text('Updated'), findsOneWidget);
      // The pill's own state is untouched by a pull.
      expect(controller.state.refreshing, isFalse);
    });

    testWidgets('a second refreshAll while one runs returns null, no call',
        (tester) async {
      final plaid = _FakePlaidRefresh()..gate = Completer();
      final controller = HomeScoreController(
          uid: 'u1', service: _FakeScoreService(), plaidRefresh: plaid.call);
      addTearDown(controller.dispose);

      final first = controller.refreshAll();
      expect(await controller.refreshAll(), isNull);
      expect(plaid.calls, 1);
      plaid.gate!.complete(_result('refreshed'));
      expect((await first)!.status, PlaidRefreshStatus.refreshed);
      // Free again once finished.
      plaid.gate = null;
      await controller.refreshAll();
      expect(plaid.calls, 2);
    });

    testWidgets('Re-run Score (refresh) never calls PlaidService.refresh',
        (tester) async {
      final score = _FakeScoreService();
      final plaid = _FakePlaidRefresh();
      final controller = HomeScoreController(
          uid: 'u1', service: score, plaidRefresh: plaid.call);
      addTearDown(controller.dispose);

      final rerun = controller.refresh();
      // No stream echo arrives in this test; let the 10 s echo wait expire.
      await tester.pump(const Duration(seconds: 11));
      await rerun;
      expect(score.refreshCalls, 1);
      expect(plaid.calls, 0);
    });

    test('the screens are wired to the shared helper (source check)', () {
      // The screens read FirebaseService.instance directly, so they cannot be
      // pumped without Firebase; this pins the wiring the widget tests above
      // exercise through the shared pieces.
      String src(String f) => File('lib/screens/$f').readAsStringSync();
      final home = src('home_screen.dart');
      expect(home, contains('onRefresh: controller.refreshAll'));
      expect(home, contains('onRerun: controller.refresh,'));
      expect(home, isNot(contains('onRefresh: controller.refresh,')));
      for (final f in ['savings_screen.dart', 'investments_screen.dart']) {
        final s = src(f);
        expect(s, contains('AccountsRefreshIndicator('), reason: f);
        expect(s, contains('onRefresh: PlaidService.instance.refresh'), reason: f);
        expect(s, contains('physics: const AlwaysScrollableScrollPhysics()'),
            reason: f);
        expect(s, contains('UpdatedLabel('), reason: f);
      }
      // Nothing in lib/ calls the sync routes or rescores around a refresh.
      for (final entity in Directory('lib').listSync(recursive: true)) {
        if (entity is! File || !entity.path.endsWith('.dart')) continue;
        final text = entity.readAsStringSync();
        expect(text, isNot(contains('/plaid/sync-')), reason: entity.path);
      }
    });
  });

  // -------------------------------------------------------------------------
  group('UpdatedLabel', () {
    testWidgets('shows, updates and hides', (tester) async {
      var clock = DateTime(2026, 10, 10, 15, 42);
      Widget host(DateTime? synced) => MaterialApp(
          home: Scaffold(
              body: UpdatedLabel(synced: synced, now: () => clock)));

      await tester.pumpWidget(host(DateTime(2026, 10, 10, 15, 37)));
      expect(find.text('Updated 5 min ago'), findsOneWidget);

      // A new snapshot from the stream (a refresh landed).
      await tester.pumpWidget(host(DateTime(2026, 10, 10, 15, 42)));
      expect(find.text('Updated just now'), findsOneWidget);

      // Time passing on its own moves it, with no new snapshot.
      clock = DateTime(2026, 10, 10, 15, 45);
      await tester.pump(const Duration(seconds: 31));
      expect(find.text('Updated 3 min ago'), findsOneWidget);

      // Never synced: nothing rendered.
      await tester.pumpWidget(host(null));
      expect(find.textContaining('Updated'), findsNothing);
    });

    testWidgets('fits a 360 dp hero row without overflow', (tester) async {
      tester.view.physicalSize = const Size(360, 640);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.reset);
      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: Padding(
            // The hero card's outer (10) and inner (16) horizontal padding.
            padding: const EdgeInsets.symmetric(horizontal: 26),
            child: Row(children: [
              const Text('Total Portfolio Value',
                  style: TextStyle(fontSize: 13)),
              const SizedBox(width: 12),
              Expanded(
                child: Align(
                  alignment: Alignment.centerRight,
                  child: UpdatedLabel(
                    synced: DateTime(2026, 10, 10, 12, 42),
                    now: () => DateTime(2026, 10, 10, 15, 42),
                    style: const TextStyle(fontSize: 11),
                  ),
                ),
              ),
            ]),
          ),
        ),
      ));
      expect(tester.takeException(), isNull);
      expect(find.text('Updated today 12:42 pm'), findsOneWidget);
    });
  });
}
