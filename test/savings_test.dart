import 'package:flutter_test/flutter_test.dart';

import 'package:unify/models/plaid_account.dart';

PlaidAccount _account({
  String id = 'acc1',
  String type = 'depository',
  String? subtype = 'savings',
  String? institution = 'Bank A',
  String? mask = '4821',
  double? current = 1000,
  double? available = 900,
}) {
  return PlaidAccount(
    accountId: id,
    itemId: 'item1',
    accountName: 'Test Account',
    accountType: type,
    accountSubtype: subtype,
    mask: mask,
    institutionName: institution,
    balanceCurrent: current,
    balanceAvailable: available,
  );
}

PlaidTransaction _txn({
  String id = 't1',
  String account = 'acc1',
  required double amount,
  required DateTime date,
}) {
  return PlaidTransaction(
    transactionId: id,
    accountId: account,
    amount: amount,
    date: date,
    name: 'Test',
  );
}

void main() {
  final now = DateTime(2026, 7, 5);

  group('SavingsSummary', () {
    test('totalSavings sums balance_current across depository only', () {
      final summary = SavingsSummary(
        accounts: [
          _account(id: 'a', current: 1200.50),
          _account(id: 'b', current: 800.25),
          _account(id: 'c', type: 'credit', current: 400),
          _account(id: 'd', type: 'loan', current: 9000),
        ],
        transactionsByAccount: const {},
        now: now,
      );
      expect(summary.totalSavings, closeTo(2000.75, 0.001));
    });

    test('bankCount counts distinct institutions', () {
      final summary = SavingsSummary(
        accounts: [
          _account(id: 'a', institution: 'Bank A'),
          _account(id: 'b', institution: 'Bank A'),
          _account(id: 'c', institution: 'Bank B'),
        ],
        transactionsByAccount: const {},
        now: now,
      );
      expect(summary.bankCount, 2);
    });

    test('avgMonthlySpent sums depository outflows in last 30 days only',
        () {
      final summary = SavingsSummary(
        accounts: [
          _account(id: 'dep'),
          _account(id: 'card', type: 'credit'),
        ],
        transactionsByAccount: {
          'dep': [
            _txn(amount: 50, date: now.subtract(const Duration(days: 5))),
            _txn(amount: 30, date: now.subtract(const Duration(days: 29))),
            // inflow — not spending
            _txn(amount: -500, date: now.subtract(const Duration(days: 3))),
            // outside the window
            _txn(amount: 999, date: now.subtract(const Duration(days: 45))),
          ],
          // credit-card spending excluded to avoid double counting
          'card': [
            _txn(amount: 80, date: now.subtract(const Duration(days: 2))),
          ],
        },
        now: now,
      );
      expect(summary.avgMonthlySpent, 80);
    });

    test('monthChange is net inflow (negated Plaid sign) and null with no '
        'data in window', () {
      final withData = SavingsSummary(
        accounts: [_account(id: 'dep')],
        transactionsByAccount: {
          'dep': [
            _txn(amount: -1800, date: now.subtract(const Duration(days: 4))),
            _txn(amount: 300, date: now.subtract(const Duration(days: 2))),
          ],
        },
        now: now,
      );
      expect(withData.monthChange, 1500);

      final noData = SavingsSummary(
        accounts: [_account(id: 'dep')],
        transactionsByAccount: {
          'dep': [
            _txn(amount: 10, date: now.subtract(const Duration(days: 90))),
          ],
        },
        now: now,
      );
      expect(noData.monthChange, isNull);
    });

    test('emergency fund need/progress derive from spend and savings', () {
      final summary = SavingsSummary(
        accounts: [_account(id: 'dep', current: 5000)],
        transactionsByAccount: {
          'dep': [
            _txn(amount: 2380, date: now.subtract(const Duration(days: 1))),
          ],
        },
        now: now,
      );
      expect(summary.emergencyFundNeed(6), 14280);
      expect(summary.emergencyFundProgress(6), closeTo(0.35, 0.001));
      // no spending data -> no meaningful progress
      final empty = SavingsSummary(
        accounts: [_account(id: 'dep')],
        transactionsByAccount: const {},
        now: now,
      );
      expect(empty.emergencyFundProgress(6), isNull);
    });

    test('recentTwo takes the first two (already date-descending)', () {
      final summary = SavingsSummary(
        accounts: [_account(id: 'dep')],
        transactionsByAccount: {
          'dep': [
            _txn(id: 'newest', amount: 1, date: now),
            _txn(id: 'mid', amount: 2, date: now.subtract(const Duration(days: 1))),
            _txn(id: 'old', amount: 3, date: now.subtract(const Duration(days: 2))),
          ],
        },
        now: now,
      );
      expect(summary.recentTwo('dep').map((t) => t.transactionId),
          ['newest', 'mid']);
      expect(summary.recentTwo('missing'), isEmpty);
    });

    test('byInstitution groups and sorts accounts', () {
      final summary = SavingsSummary(
        accounts: [
          _account(id: 'b1', institution: 'Bank B'),
          _account(id: 'a1', institution: 'Bank A'),
          _account(id: 'a2', institution: 'Bank A'),
        ],
        transactionsByAccount: const {},
        now: now,
      );
      final grouped = summary.byInstitution;
      expect(grouped.keys.toList(), ['Bank A', 'Bank B']);
      expect(grouped['Bank A']!.length, 2);
    });

    test('type predicates follow account_type', () {
      expect(_account(type: 'depository').isDepository, isTrue);
      expect(_account(type: 'depository').isDebt, isFalse);
      expect(_account(type: 'credit').isCredit, isTrue);
      expect(_account(type: 'credit').isDebt, isTrue);
      expect(_account(type: 'loan').isLoan, isTrue);
      expect(_account(type: 'loan').isDebt, isTrue);
      expect(_account(type: 'credit').isLoan, isFalse);
    });

    test('balanceOwed reads balance_current, never balance_available', () {
      // The real Sandbox shape: two cards on one Item, one with available
      // null and one with available populated. Both must report what is
      // owed, not whichever field happened to be set.
      final noAvailable =
          _account(type: 'credit', current: 410, available: null);
      final withAvailable =
          _account(type: 'credit', current: 410, available: 4980);
      expect(noAvailable.balanceOwed, 410);
      expect(withAvailable.balanceOwed, 410);
      expect(noAvailable.balanceOwed, withAvailable.balanceOwed);
      // displayBalance is exactly the trap being avoided.
      expect(withAvailable.displayBalance, 4980);
      // Depository accounts owe nothing.
      expect(_account(current: 1000).balanceOwed, 0);
    });

    test('rowBalance is owed for debt, available-then-current otherwise', () {
      expect(_account(current: 1000, available: 900).rowBalance, 900);
      expect(_account(current: 1000, available: null).rowBalance, 1000);
      expect(
          _account(type: 'credit', current: 410, available: 4980).rowBalance,
          410);
    });

    test('card totals split available from owed', () {
      // The real O7oEGbN3 test user's single-institution card.
      final accounts = [
        _account(id: 'd1', current: 1000, available: null), // cd
        _account(id: 'd2', current: 110, available: 100), // checking
        _account(id: 'd3', current: 43200, available: 43200), // money market
        _account(id: 'd4', current: 210, available: 200), // savings
        _account(id: 'd5', current: 12060, available: 12060), // cash mgmt
        _account(id: 'd6', current: 6009, available: 6009), // hsa
        _account(id: 'c1', type: 'credit', current: 410, available: null),
        _account(id: 'c2', type: 'credit', current: 5020, available: 4980),
        _account(id: 'l1', type: 'loan', current: 65262, available: null),
        _account(id: 'l2', type: 'loan', current: 56302.06, available: null),
      ];
      // Available = depository displayBalance only. Distinct from the Total
      // Savings hero card's 62,589.00, which sums balance_current instead —
      // two different quantities by design, both correct for their own card.
      expect(accounts.availableTotal, closeTo(62569.0, 0.001));
      // Owed = balance_current across credit + loan.
      expect(accounts.owedTotal, closeTo(126994.06, 0.001));
      expect(accounts.hasDebt, isTrue);
      // The old merged fold — the defect this replaces.
      final oldMerged =
          accounts.fold(0.0, (acc, a) => acc + a.displayBalance);
      expect(oldMerged, closeTo(189523.06, 0.001));
      expect(accounts.availableTotal, lessThan(oldMerged));
    });

    test('depository-only institution reports no debt', () {
      final accounts = [
        _account(id: 'a', current: 2000, available: 2000),
        _account(id: 'b', current: 500, available: 500),
      ];
      expect(accounts.hasDebt, isFalse);
      expect(accounts.owedTotal, 0);
      expect(accounts.availableTotal, closeTo(2500, 0.001));
    });

    test('depositoryFirst orders debt accounts last, keeping all', () {
      final accounts = [
        _account(id: 'c1', type: 'credit'),
        _account(id: 'd1', type: 'depository'),
        _account(id: 'l1', type: 'loan'),
        _account(id: 'd2', type: 'depository'),
      ];
      expect(accounts.depositoryFirst.map((a) => a.accountId),
          ['d1', 'd2', 'c1', 'l1']);
      expect(accounts.depositoryFirst.length, accounts.length);
    });

    test('maskedLabel formats mask + subtype', () {
      expect(_account().maskedLabel, '•••• 4821 · Savings');
      expect(_account(mask: null).maskedLabel, 'Savings');
      expect(
        _account(subtype: 'credit card', mask: '3333').maskedLabel,
        '•••• 3333 · Credit card',
      );
    });
  });
}
