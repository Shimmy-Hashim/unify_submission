import 'dart:developer' as developer;

import 'package:cloud_firestore/cloud_firestore.dart';

/// Return figures outside this range are treated as bad source data (the
/// Sandbox cost_basis fixtures are semantically inconsistent — sometimes
/// per-share, sometimes total) and displayed as "—" instead of an absurd
/// number. The calculation itself stays correct for trustworthy data.
const double kMinPlausibleReturn = -0.95; // −95%
const double kMaxPlausibleReturn = 5.0; // +500%

/// Avg-purchase-price guard: a derived per-share cost more than 10× away
/// from the current per-share market price is shown as "—", same reasoning.
const double kMaxAvgPriceFactor = 10.0;

/// Any account whose single largest holding exceeds this share of the
/// account is flagged Overweight (matches the design's "Limit is 30%").
const double kConcentrationLimit = 0.30;

void _logGuard(String what, String holding) {
  developer.log(
    'sanity guard: $what for "$holding" outside plausible range — '
    'showing "—" (known Sandbox cost_basis inconsistency)',
    name: 'investments',
  );
}

/// One entry of the holdings array on
/// users/{uid}/investmentAccounts/{accountId} (brief Section 8).
///
/// cost_basis is parsed when present but is NOT currently written by the
/// Phase 3 backend sync (Section 8's holdings schema stops at asset_type),
/// so every return/avg-price figure renders "—" until the backend adds it —
/// flagged as a Phase 3 follow-up, deliberately not worked around here.
class InvestmentHolding {
  const InvestmentHolding({
    this.ticker,
    this.name,
    this.quantity,
    this.marketValue,
    this.assetType,
    this.costBasis,
  });

  final String? ticker;
  final String? name;
  final double? quantity;
  final double? marketValue;
  final String? assetType;
  final double? costBasis;

  factory InvestmentHolding.fromMap(Map<String, dynamic> data) {
    return InvestmentHolding(
      ticker: data['ticker'] as String?,
      name: data['name'] as String?,
      quantity: (data['quantity'] as num?)?.toDouble(),
      marketValue: (data['market_value'] as num?)?.toDouble(),
      assetType: data['asset_type'] as String?,
      costBasis: (data['cost_basis'] as num?)?.toDouble(),
    );
  }

  String get displayName => (name?.isNotEmpty ?? false)
      ? name!
      : (ticker?.isNotEmpty ?? false)
          ? ticker!
          : 'Holding';

  /// Short label for tight spots (legend): ticker when available.
  String get shortLabel =>
      (ticker?.isNotEmpty ?? false) ? ticker! : displayName;

  /// Total return since purchase as a fraction (0.12 = +12%), or null when
  /// cost basis is missing or the result trips the sanity guard.
  double? get totalReturnSincePurchase {
    final cb = costBasis;
    final mv = marketValue;
    if (cb == null || cb <= 0 || mv == null) return null;
    final r = (mv - cb) / cb;
    if (r < kMinPlausibleReturn || r > kMaxPlausibleReturn) {
      _logGuard('total return ${(r * 100).toStringAsFixed(0)}%', displayName);
      return null;
    }
    return r;
  }

  /// cost_basis / quantity, or null when unavailable or implausible next to
  /// the current market per-share price.
  double? get avgPurchasePrice {
    final cb = costBasis;
    final qty = quantity;
    if (cb == null || cb <= 0 || qty == null || qty <= 0) return null;
    final avg = cb / qty;
    final mv = marketValue;
    if (mv != null && mv > 0) {
      final marketPerShare = mv / qty;
      if (avg > marketPerShare * kMaxAvgPriceFactor ||
          avg < marketPerShare / kMaxAvgPriceFactor) {
        _logGuard('avg price \$${avg.toStringAsFixed(2)} vs market '
            '\$${marketPerShare.toStringAsFixed(2)}', displayName);
        return null;
      }
    }
    return avg;
  }
}

/// One users/{uid}/investmentAccounts/{accountId} document (brief Section
/// 8). Written exclusively by the backend Phase 3 sync; client read-only.
class InvestmentAccount {
  const InvestmentAccount({
    required this.accountId,
    required this.accountName,
    required this.accountType,
    this.accountSubtypeRaw,
    this.itemId,
    this.source,
    this.totalValue = 0,
    this.holdings = const [],
    this.assetAllocation = const {},
    this.concentrationRisk,
    this.isoCurrencyCode,
    this.lastSynced,
  });

  final String accountId;
  final String accountName;
  final String accountType; // "brokerage" | "srs" | "cpfis" | "other"
  final String? accountSubtypeRaw; // Plaid's raw subtype, e.g. "401k"
  final String? itemId;
  final String? source;
  final double totalValue;
  final List<InvestmentHolding> holdings;
  final Map<String, double> assetAllocation;
  final double? concentrationRisk; // fraction 0–1, cached by the backend
  final String? isoCurrencyCode;

  /// When the backend last wrote this account (holdings sync).
  final DateTime? lastSynced;

  factory InvestmentAccount.fromFirestore(
      String id, Map<String, dynamic> data) {
    return InvestmentAccount(
      accountId: id,
      accountName: (data['account_name'] as String?) ?? 'Investment account',
      accountType: (data['account_type'] as String?) ?? 'other',
      accountSubtypeRaw: data['account_subtype_raw'] as String?,
      itemId: data['item_id'] as String?,
      source: data['source'] as String?,
      totalValue: (data['total_value_sgd'] as num?)?.toDouble() ?? 0,
      holdings: [
        for (final h in (data['holdings'] as List?) ?? [])
          if (h is Map) InvestmentHolding.fromMap(Map<String, dynamic>.from(h)),
      ],
      assetAllocation: {
        for (final e
            in ((data['asset_allocation'] as Map?) ?? {}).entries)
          e.key.toString(): (e.value as num?)?.toDouble() ?? 0,
      },
      concentrationRisk: (data['concentration_risk'] as num?)?.toDouble(),
      isoCurrencyCode: data['iso_currency_code'] as String?,
      lastSynced: (data['last_synced'] as Timestamp?)?.toDate(),
    );
  }

  InvestmentHolding? get largestHolding {
    InvestmentHolding? largest;
    for (final h in holdings) {
      if ((h.marketValue ?? 0) > (largest?.marketValue ?? -1)) largest = h;
    }
    return largest;
  }
}

/// One investmentAccounts/{accountId}/transactions document (Section 8).
/// Amounts keep Plaid's sign convention: positive = cash debited (a buy),
/// negative = cash credited (deposit / sale proceeds / dividend).
class InvestmentTransaction {
  const InvestmentTransaction({
    required this.transactionId,
    required this.accountId,
    required this.type,
    required this.amount,
    this.quantity,
    this.price,
    this.ticker,
    this.date,
  });

  final String transactionId;
  final String accountId;
  final String type; // "buy" | "sell" | "dividend" | "deposit" | "withdrawal"
  final double amount;
  final double? quantity;
  final double? price;
  final String? ticker;
  final DateTime? date;

  factory InvestmentTransaction.fromFirestore(
      String id, Map<String, dynamic> data) {
    return InvestmentTransaction(
      transactionId: id,
      accountId: (data['account_id'] as String?) ?? '',
      type: (data['type'] as String?) ?? '',
      amount: (data['amount'] as num?)?.toDouble() ?? 0,
      quantity: (data['quantity'] as num?)?.toDouble(),
      price: (data['price'] as num?)?.toDouble(),
      ticker: data['ticker'] as String?,
      date: (data['date'] as Timestamp?)?.toDate(),
    );
  }

  bool get isCashOut => amount > 0;
}

/// One holding with its owning-account context, so per-position rows can
/// show share-of-portfolio and the Overweight badge.
class PortfolioPosition {
  const PortfolioPosition({
    required this.account,
    required this.holding,
    required this.share,
    required this.overweight,
  });

  final InvestmentAccount account;
  final InvestmentHolding holding;

  /// market_value / total portfolio value, 0–1.
  final double share;

  /// True when this is the largest holding of an account whose cached
  /// concentration_risk exceeds [kConcentrationLimit].
  final bool overweight;
}

/// Pure computations behind the Investments screen — no Flutter/Firestore
/// imports so it unit-tests directly (same pattern as SavingsSummary).
class PortfolioSummary {
  PortfolioSummary({required this.accounts}) {
    final total = accounts.fold(0.0, (acc, a) => acc + a.totalValue);
    final list = <PortfolioPosition>[];
    for (final account in accounts) {
      final concentrated =
          (account.concentrationRisk ?? 0) > kConcentrationLimit;
      final largest = account.largestHolding;
      for (final holding in account.holdings) {
        list.add(PortfolioPosition(
          account: account,
          holding: holding,
          share: total > 0 ? (holding.marketValue ?? 0) / total : 0,
          overweight: concentrated && identical(holding, largest),
        ));
      }
    }
    list.sort((a, b) => (b.holding.marketValue ?? 0)
        .compareTo(a.holding.marketValue ?? 0));
    totalValue = total;
    positions = List.unmodifiable(list);
  }

  final List<InvestmentAccount> accounts;
  late final double totalValue;

  /// Every holding across all accounts, sorted by market value descending.
  late final List<PortfolioPosition> positions;

  int get holdingsCount => positions.length;
  bool get isEmpty => accounts.isEmpty;

  /// The newest sync time across every investment account, for the hero
  /// card's "Updated ..." label. Null when nothing has ever synced.
  DateTime? get lastSynced {
    DateTime? newest;
    for (final a in accounts) {
      final t = a.lastSynced;
      if (t != null && (newest == null || t.isAfter(newest))) newest = t;
    }
    return newest;
  }

  /// Accounts flagged by the backend-cached concentration_risk (fraction of
  /// the ACCOUNT in its single largest holding) above the 30% limit.
  List<InvestmentAccount> get overweightAccounts => [
        for (final a in accounts)
          if ((a.concentrationRisk ?? 0) > kConcentrationLimit &&
              a.largestHolding != null)
            a,
      ];

  /// Highest total-return-since-purchase holding (all-time, not monthly —
  /// deliberate substitution, labeled as such in the UI). Null when no
  /// holding has usable cost-basis data.
  PortfolioPosition? get bestPerformer {
    PortfolioPosition? best;
    double bestReturn = double.negativeInfinity;
    for (final p in positions) {
      final r = p.holding.totalReturnSincePurchase;
      if (r != null && r > bestReturn) {
        bestReturn = r;
        best = p;
      }
    }
    return best;
  }

  /// Largest NEGATIVE total-return-since-purchase holding, for the
  /// "declining position" alert. Null when nothing is down (or no data).
  PortfolioPosition? get worstDecliner {
    PortfolioPosition? worst;
    double worstReturn = 0;
    for (final p in positions) {
      final r = p.holding.totalReturnSincePurchase;
      if (r != null && r < worstReturn) {
        worstReturn = r;
        worst = p;
      }
    }
    return worst;
  }

  /// Portfolio-wide since-purchase return: (Σ market_value − Σ cost_basis)
  /// / Σ cost_basis over the holdings that carry cost basis, guarded like
  /// the per-holding figure. Null when no cost-basis data exists at all.
  double? get portfolioReturnSincePurchase {
    var sumValue = 0.0, sumCost = 0.0;
    for (final p in positions) {
      final cb = p.holding.costBasis;
      final mv = p.holding.marketValue;
      if (cb == null || cb <= 0 || mv == null) continue;
      sumValue += mv;
      sumCost += cb;
    }
    if (sumCost <= 0) return null;
    final r = (sumValue - sumCost) / sumCost;
    if (r < kMinPlausibleReturn || r > kMaxPlausibleReturn) {
      _logGuard('portfolio return ${(r * 100).toStringAsFixed(0)}%',
          'portfolio');
      return null;
    }
    return r;
  }
}
