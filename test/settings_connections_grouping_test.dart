import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:unify/models/investment_holding.dart';
import 'package:unify/models/plaid_account.dart';
import 'package:unify/screens/settings_screen.dart';

/// Connections' institution grouping, and the disconnect disambiguation that
/// grouping makes necessary.
///
/// The screen itself is Firestore-stream-backed and not widget-tested, so the
/// grouping is exercised directly through `ConnectionsSummary.from` — which is
/// exactly the function the card calls with the documents its two snapshots
/// return.
PlaidAccount bank(
  String id, {
  required String itemId,
  String? institution,
  DateTime? synced,
}) =>
    PlaidAccount(
      accountId: id,
      itemId: itemId,
      accountName: 'Account $id',
      accountType: 'depository',
      institutionName: institution,
      lastSynced: synced,
    );

InvestmentAccount invest(String id, {String? itemId, String name = 'Brokerage'}) =>
    InvestmentAccount(
      accountId: id,
      accountName: name,
      accountType: 'brokerage',
      itemId: itemId,
      totalValue: 1000,
      holdings: const [],
      assetAllocation: const {},
    );

void main() {
  group('grouping by institution name', () {
    test('one Item per institution gives one row per Item, as before', () {
      final summary = ConnectionsSummary.from(
        banking: [
          bank('a1', itemId: 'item-1', institution: 'Tartan Bank'),
          bank('a2', itemId: 'item-1', institution: 'Tartan Bank'),
          bank('b1', itemId: 'item-2', institution: 'Houndstooth Bank'),
        ],
        investments: [invest('i1', itemId: 'item-1')],
      );

      expect(summary.groups.length, 2);
      expect(summary.items.length, 2, reason: 'two underlying Items');

      final tartan = summary.groups.first;
      expect(tartan.institutionName, 'Tartan Bank');
      expect(tartan.breakdown, '2 banking accounts · 1 investment account');
      expect(tartan.needsDisambiguation, isFalse);
    });

    test('two Items at the same institution MERGE into one row with summed '
        'counts', () {
      final summary = ConnectionsSummary.from(
        banking: [
          bank('a1', itemId: 'item-1', institution: 'Tartan Bank'),
          bank('a2', itemId: 'item-1', institution: 'Tartan Bank'),
          bank('b1', itemId: 'item-2', institution: 'Tartan Bank'),
        ],
        investments: [
          invest('i1', itemId: 'item-1'),
          invest('i2', itemId: 'item-2'),
        ],
      );

      expect(summary.groups.length, 1, reason: 'one institution, one row');
      final tartan = summary.groups.single;
      // Summed across BOTH Items, not one Item's breakdown.
      expect(tartan.bankingCount, 3);
      expect(tartan.investmentCount, 2);
      expect(tartan.breakdown, '3 banking accounts · 2 investment accounts');

      // ...but the Items underneath are still separate, which is what keeps
      // disconnect per-Item.
      expect(tartan.items.length, 2);
      expect(tartan.needsDisambiguation, isTrue);
      expect(
        tartan.items.map((i) => i.itemId),
        containsAll(<String>['item-1', 'item-2']),
      );
    });

    test('an investment-only Item keeps the account-name fallback', () {
      final summary = ConnectionsSummary.from(
        banking: [bank('a1', itemId: 'item-1', institution: 'Tartan Bank')],
        investments: [invest('i9', itemId: 'item-9', name: 'Vanguard Brokerage')],
      );

      expect(summary.groups.length, 2);
      final brokerage =
          summary.groups.firstWhere((g) => g.institutionName != 'Tartan Bank');
      // No institution_name exists for such an Item anywhere the client can
      // read, so the account name stands in — unchanged from before the merge.
      expect(brokerage.institutionName, 'Vanguard Brokerage');
      expect(brokerage.breakdown, '1 investment account');
      expect(brokerage.needsDisambiguation, isFalse);
      expect(brokerage.disconnectableItems.single.itemId, 'item-9');
    });

    test('a mixed Item is named by its BANK, never by its brokerage account',
        () {
      // The ordering guarantee: investments are attached to an Item that
      // banking already named. If this regressed, the row would read
      // "Brokerage" instead of "Tartan Bank".
      final summary = ConnectionsSummary.from(
        banking: [bank('a1', itemId: 'item-1', institution: 'Tartan Bank')],
        investments: [invest('i1', itemId: 'item-1', name: 'Some Brokerage')],
      );
      expect(summary.groups.single.institutionName, 'Tartan Bank');
    });

    test('a manual investment entry (no item_id) is shown but not '
        'disconnectable', () {
      final summary = ConnectionsSummary.from(
        banking: const [],
        investments: [invest('m1', itemId: null, name: 'Manual Holdings')],
      );
      final group = summary.groups.single;
      expect(group.institutionName, 'Manual Holdings');
      expect(group.items.single.itemId, isNull);
      // No Plaid Item behind it, so nothing to revoke.
      expect(group.disconnectableItems, isEmpty);
      expect(group.needsDisambiguation, isFalse);
    });

    test('a banking account with no institution_name falls back', () {
      final summary = ConnectionsSummary.from(
        banking: [bank('a1', itemId: 'item-1')],
        investments: const [],
      );
      expect(summary.groups.single.institutionName, 'Connected bank');
    });

    test('nothing connected produces no groups and no items', () {
      final summary =
          ConnectionsSummary.from(banking: const [], investments: const []);
      expect(summary.groups, isEmpty);
      expect(summary.items, isEmpty);
    });

    test('the Active badge counts CONNECTIONS, not merged rows', () {
      final summary = ConnectionsSummary.from(
        banking: [
          bank('a1', itemId: 'item-1', institution: 'Tartan Bank'),
          bank('b1', itemId: 'item-2', institution: 'Tartan Bank'),
          bank('c1', itemId: 'item-3', institution: 'Houndstooth Bank'),
        ],
        investments: const [],
      );
      // 2 rows, but 3 live Plaid Items — the badge reports the Items, so the
      // app never under-reports how many connections it actually holds.
      expect(summary.groups.length, 2);
      expect(summary.items.length, 3);
    });

    test('last synced per Item is the LATEST across that Item\'s accounts', () {
      final early = DateTime.utc(2026, 7, 30, 12);
      final late_ = DateTime.utc(2026, 8, 2, 3);
      final summary = ConnectionsSummary.from(
        banking: [
          bank('a1', itemId: 'item-1', institution: 'Tartan Bank', synced: early),
          bank('a2', itemId: 'item-1', institution: 'Tartan Bank', synced: late_),
        ],
        investments: const [],
      );
      expect(summary.groups.single.items.single.lastSynced, late_);
    });
  });

  group('disconnect picker', () {
    /// Pumps the picker and returns which item the caller received.
    Future<ConnectedItem?> pick(
      WidgetTester tester,
      List<ConnectedItem> items, {
      required Future<void> Function(WidgetTester) act,
    }) async {
      ConnectedItem? chosen;
      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: Builder(
            builder: (context) => ElevatedButton(
              onPressed: () async {
                chosen = await showDisconnectItemPicker(
                    context, 'Tartan Bank', items);
              },
              child: const Text('open'),
            ),
          ),
        ),
      ));
      await tester.tap(find.text('open'));
      await tester.pumpAndSettle();
      await act(tester);
      await tester.pumpAndSettle();
      return chosen;
    }

    List<ConnectedItem> twoItems() {
      final a = ConnectedItem('Tartan Bank', 'item-1')
        ..bankingCount = 10
        ..investmentCount = 2
        ..lastSynced = DateTime.utc(2026, 7, 30, 12);
      final b = ConnectedItem('Tartan Bank', 'item-2')
        ..bankingCount = 10
        ..investmentCount = 2
        ..lastSynced = DateTime.utc(2026, 8, 2, 3);
      return [a, b];
    }

    testWidgets('lists every connection with distinguishing detail',
        (tester) async {
      await pick(tester, twoItems(), act: (t) async {
        expect(find.text('Which Tartan Bank connection?'), findsOneWidget);
        // Identical account counts are why the date and the ordinal matter —
        // real Sandbox Items from one institution are otherwise identical.
        expect(find.textContaining('Connection 1'), findsOneWidget);
        expect(find.textContaining('Connection 2'), findsOneWidget);
        expect(find.textContaining('30 Jul 2026'), findsOneWidget);
        expect(find.textContaining('2 Aug 2026'), findsOneWidget);
        expect(
          find.text('10 banking accounts · 2 investment accounts'),
          findsNWidgets(2),
        );
        // Reassures the user the others survive.
        expect(find.textContaining('The others stay connected'), findsOneWidget);
        await t.tapAt(const Offset(10, 10)); // dismiss
      });
    });

    testWidgets('returns the specific connection the user picked',
        (tester) async {
      final chosen = await pick(tester, twoItems(),
          act: (t) async => t.tap(find.textContaining('Connection 2')));
      expect(chosen, isNotNull);
      expect(chosen!.itemId, 'item-2',
          reason: 'the picked Item, not simply the first in the group');
    });

    testWidgets('dismissing picks nothing, so no disconnect follows',
        (tester) async {
      final chosen = await pick(tester, twoItems(),
          act: (t) async => t.tapAt(const Offset(10, 10)));
      expect(chosen, isNull);
    });
  });
}
