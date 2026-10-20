import 'package:cloud_firestore/cloud_firestore.dart';

/// One users/{uid}/plaidAccounts/{accountId} document (brief Section 8,
/// banking only — depository | credit | loan). Written exclusively by the
/// backend; the client only reads.
class PlaidAccount {
  const PlaidAccount({
    required this.accountId,
    required this.itemId,
    required this.accountName,
    required this.accountType,
    this.accountSubtype,
    this.mask,
    this.institutionName,
    this.balanceCurrent,
    this.balanceAvailable,
    this.isoCurrencyCode,
    this.lastSynced,
  });

  final String accountId;
  final String itemId;
  final String accountName;
  final String accountType;
  final String? accountSubtype;
  final String? mask;
  final String? institutionName;
  final double? balanceCurrent;
  final double? balanceAvailable;
  final String? isoCurrencyCode;

  /// When the backend last wrote this account. Read by Settings' disconnect
  /// picker, which needs something that actually differs between two Items
  /// from the SAME institution — account counts and Plaid's Sandbox masks are
  /// identical across such Items, so this is the only distinguishing field the
  /// client can see (plaidItems, where last_synced canonically lives, is not
  /// client-readable).
  final DateTime? lastSynced;

  factory PlaidAccount.fromFirestore(String id, Map<String, dynamic> data) {
    return PlaidAccount(
      accountId: id,
      itemId: (data['item_id'] as String?) ?? '',
      accountName: (data['account_name'] as String?) ?? 'Account',
      accountType: (data['account_type'] as String?) ?? '',
      accountSubtype: data['account_subtype'] as String?,
      mask: data['mask'] as String?,
      institutionName: data['institution_name'] as String?,
      balanceCurrent: (data['balance_current'] as num?)?.toDouble(),
      balanceAvailable: (data['balance_available'] as num?)?.toDouble(),
      isoCurrencyCode: data['iso_currency_code'] as String?,
      lastSynced: (data['last_synced'] as Timestamp?)?.toDate(),
    );
  }

  bool get isDepository => accountType == 'depository';
  bool get isCredit => accountType == 'credit';
  bool get isLoan => accountType == 'loan';

  /// Credit cards and loans — the two banking types (brief Section 8) that
  /// hold money owed rather than money available to spend.
  bool get isDebt => isCredit || isLoan;

  /// Balance shown on account rows: available, falling back to current.
  double get displayBalance => balanceAvailable ?? balanceCurrent ?? 0;

  /// What a credit or loan account owes, read from `balance_current` ONLY.
  ///
  /// [displayBalance]'s available-then-current fallback is actively wrong
  /// here: on a credit card Plaid's `available` is the REMAINING CREDIT LINE,
  /// not the debt, so the fallback reports two different quantities under one
  /// label depending on which field the institution happened to populate.
  /// Real Sandbox data carries both shapes on the same Item — one card with
  /// `available` null (so the fallback reached `current`, the amount owed) and
  /// one with `available` set (so it reported remaining credit instead).
  ///
  /// Zero for depository accounts, which owe nothing.
  double get balanceOwed => isDebt ? (balanceCurrent ?? 0) : 0;

  /// The balance for this account's own row: what is available on a
  /// depository account, what is owed on a credit or loan one.
  double get rowBalance => isDebt ? balanceOwed : displayBalance;

  /// "•••• 4821 · Savings" — mask may be absent for some institutions.
  String get maskedLabel {
    final subtype = accountSubtype == null || accountSubtype!.isEmpty
        ? accountType
        : accountSubtype!;
    final pretty = subtype.isEmpty
        ? ''
        : subtype[0].toUpperCase() + subtype.substring(1);
    return mask == null || mask!.isEmpty ? pretty : '•••• $mask · $pretty';
  }
}

/// Totals for one Connected Accounts card. Scoped to a single institution's
/// own account list rather than the whole connected set, which is why these
/// live here and not on [SavingsSummary].
extension PlaidAccountTotals on Iterable<PlaidAccount> {
  /// Spendable cash: depository accounts only.
  ///
  /// Credit and loan balances are debt and are never added in. Summing all
  /// three types into one "Available" figure was a real defect — on a test
  /// user it read $189,523.06 available against $62,589.00 of actual cash,
  /// with $126,994.06 of mortgage, student loan and card debt counted as
  /// though it were money to spend.
  double get availableTotal => where((a) => a.isDepository)
      .fold(0.0, (acc, a) => acc + a.displayBalance);

  /// Money owed across this card's credit and loan accounts, each read via
  /// [PlaidAccount.balanceOwed] so the available-vs-current trap can't return.
  double get owedTotal =>
      where((a) => a.isDebt).fold(0.0, (acc, a) => acc + a.balanceOwed);

  /// Whether this card has anything to show an "Owed" figure for. An
  /// institution with only depository accounts shows "Available" alone.
  bool get hasDebt => any((a) => a.isDebt);

  /// Depository accounts first, debt accounts last, so the card can rule a
  /// single sub-divider between the two groups. Anything that is neither
  /// (no such type exists today — the backend writes only the three banking
  /// types) stays in the first group rather than disappearing.
  List<PlaidAccount> get depositoryFirst =>
      [...where((a) => !a.isDebt), ...where((a) => a.isDebt)];
}

/// One transactions subcollection document. Plaid sign convention (brief
/// Section 8): positive amount = money out, negative = money in.
class PlaidTransaction {
  const PlaidTransaction({
    required this.transactionId,
    required this.accountId,
    required this.amount,
    this.date,
    this.name,
    this.category = const [],
    this.pending = false,
  });

  final String transactionId;
  final String accountId;
  final double amount;
  final DateTime? date;
  final String? name;
  final List<String> category;
  final bool pending;

  factory PlaidTransaction.fromFirestore(
      String id, Map<String, dynamic> data) {
    return PlaidTransaction(
      transactionId: id,
      accountId: (data['account_id'] as String?) ?? '',
      amount: (data['amount'] as num?)?.toDouble() ?? 0,
      date: (data['date'] as Timestamp?)?.toDate(),
      name: data['name'] as String?,
      category: [
        for (final c in (data['category'] as List?) ?? []) c.toString(),
      ],
      pending: (data['pending'] as bool?) ?? false,
    );
  }

  bool get isOutflow => amount > 0;
}

/// Pure computations behind the Total Savings card and the Emergency Fund
/// estimator — kept free of Flutter/Firestore imports so they unit-test
/// directly (same pattern as CoverageSummary on the Insurance screen).
class SavingsSummary {
  SavingsSummary({
    required List<PlaidAccount> accounts,
    required Map<String, List<PlaidTransaction>> transactionsByAccount,
    DateTime? now,
  })  : _accounts = accounts,
        _txns = transactionsByAccount,
        _now = now ?? DateTime.now();

  final List<PlaidAccount> _accounts;
  final Map<String, List<PlaidTransaction>> _txns;
  final DateTime _now;

  List<PlaidAccount> get depositoryAccounts =>
      _accounts.where((a) => a.isDepository).toList();

  /// Section 1 headline: sum of balance_current across depository accounts.
  double get totalSavings => depositoryAccounts.fold(
      0.0, (acc, a) => acc + (a.balanceCurrent ?? 0));

  /// The newest sync time across every connected account, for the hero card's
  /// "Updated ..." label. Null when nothing has ever synced.
  DateTime? get lastSynced {
    DateTime? newest;
    for (final a in _accounts) {
      final t = a.lastSynced;
      if (t != null && (newest == null || t.isAfter(newest))) newest = t;
    }
    return newest;
  }

  /// "N Banks" — distinct institutions across all connected accounts.
  int get bankCount => _accounts
      .map((a) => a.institutionName ?? a.itemId)
      .toSet()
      .length;

  Iterable<PlaidTransaction> _depositoryTxnsSince(Duration window) sync* {
    final cutoff = _now.subtract(window);
    final depositoryIds =
        depositoryAccounts.map((a) => a.accountId).toSet();
    for (final entry in _txns.entries) {
      if (!depositoryIds.contains(entry.key)) continue;
      for (final txn in entry.value) {
        if (txn.date != null && !txn.date!.isBefore(cutoff)) yield txn;
      }
    }
  }

  /// Sum of outflows from depository accounts over the last 30 days.
  /// Credit-card spending is deliberately excluded: card repayments leave
  /// the depository accounts, so counting both would double-count.
  double get avgMonthlySpent => _depositoryTxnsSince(
        const Duration(days: 30),
      ).where((t) => t.isOutflow).fold(0.0, (acc, t) => acc + t.amount);

  /// Net movement of depository balances over the last 30 days
  /// (inflows − outflows; Plaid amounts are positive-out, so negated sum).
  /// Null when there are no transactions in the window — the card omits
  /// the change indicator rather than showing a fabricated $0.
  double? get monthChange {
    final txns = _depositoryTxnsSince(const Duration(days: 30)).toList();
    if (txns.isEmpty) return null;
    return -txns.fold(0.0, (acc, t) => acc + t.amount);
  }

  /// Emergency-fund target for [months] of expenses.
  double emergencyFundNeed(int months) => avgMonthlySpent * months;

  /// 0..1 progress toward the target; null when spending data is missing
  /// (no meaningful percentage can be shown).
  double? emergencyFundProgress(int months) {
    final need = emergencyFundNeed(months);
    if (need <= 0) return null;
    return (totalSavings / need).clamp(0.0, 1.0);
  }

  /// Two most recent transactions for one account (input lists are already
  /// date-descending from Firestore).
  List<PlaidTransaction> recentTwo(String accountId) =>
      (_txns[accountId] ?? const []).take(2).toList();

  /// Accounts grouped by institution for the Connected Accounts section,
  /// insertion-ordered by institution name.
  Map<String, List<PlaidAccount>> get byInstitution {
    final grouped = <String, List<PlaidAccount>>{};
    final sorted = [..._accounts]..sort((a, b) =>
        (a.institutionName ?? '').compareTo(b.institutionName ?? ''));
    for (final account in sorted) {
      grouped.putIfAbsent(account.institutionName ?? 'Bank', () => [])
          .add(account);
    }
    return grouped;
  }
}
