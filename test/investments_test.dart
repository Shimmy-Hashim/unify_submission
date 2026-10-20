import 'package:flutter_test/flutter_test.dart';

import 'package:unify/models/investment_holding.dart';

InvestmentHolding _holding({
  String? ticker,
  String? name,
  double? quantity,
  double? marketValue,
  String? assetType = 'equity',
  double? costBasis,
}) {
  return InvestmentHolding(
    ticker: ticker,
    name: name,
    quantity: quantity,
    marketValue: marketValue,
    assetType: assetType,
    costBasis: costBasis,
  );
}

InvestmentAccount _account({
  String id = 'acc1',
  String name = 'Brokerage',
  double totalValue = 0,
  List<InvestmentHolding> holdings = const [],
  double? concentrationRisk,
}) {
  return InvestmentAccount(
    accountId: id,
    accountName: name,
    accountType: 'brokerage',
    totalValue: totalValue,
    holdings: holdings,
    concentrationRisk: concentrationRisk,
  );
}

void main() {
  group('InvestmentHolding sanity guards', () {
    // Real First Platypus Bank fixture values (2026-07-06 investigation):
    // SBSI holds 213 shares at $34.73 ($7,397.49) with cost_basis 30.0 —
    // read as a total cost that is a +24,558% return, the known-absurd
    // Sandbox case. Both derived figures must refuse to render.
    test('known-bad Sandbox SBSI cost_basis is guarded to null', () {
      final sbsi = _holding(
        ticker: 'SBSI',
        name: 'Southside Bancshares Inc.',
        quantity: 213,
        marketValue: 7397.49,
        costBasis: 30.0,
      );
      // Raw would be (7397.49 - 30) / 30 ≈ +245.58 (i.e. +24,558%).
      expect(sbsi.totalReturnSincePurchase, isNull);
      // Raw avg would be $0.14 vs $34.73 market — >10x apart.
      expect(sbsi.avgPurchasePrice, isNull);
    });

    test('plausible returns pass through unguarded (ACHN fixture)', () {
      final achn = _holding(
        ticker: 'ACHN',
        quantity: 1,
        marketValue: 2.11,
        costBasis: 1.5,
      );
      expect(achn.totalReturnSincePurchase, closeTo(0.4067, 0.001));
      expect(achn.avgPurchasePrice, closeTo(1.5, 0.001));
    });

    test('plausible negative return passes (BTC fixture)', () {
      final btc = _holding(
        ticker: 'BTC',
        quantity: 0.00293644,
        marketValue: 115.57268,
        costBasis: 120.03,
      );
      expect(btc.totalReturnSincePurchase, closeTo(-0.0371, 0.001));
    });

    test('missing cost_basis (current backend state) yields null, not 0',
        () {
      final h = _holding(quantity: 10, marketValue: 500);
      expect(h.totalReturnSincePurchase, isNull);
      expect(h.avgPurchasePrice, isNull);
    });

    test('return below −95% is guarded', () {
      final h = _holding(quantity: 1, marketValue: 1, costBasis: 100);
      expect(h.totalReturnSincePurchase, isNull);
    });

    test('fromMap parses holdings with and without cost_basis', () {
      final withCb = InvestmentHolding.fromMap(const {
        'ticker': 'EWZ',
        'name': 'iShares Inc MSCI Brazil',
        'quantity': 5,
        'market_value': 210.75,
        'asset_type': 'etf',
        'cost_basis': 200.0,
      });
      expect(withCb.costBasis, 200.0);
      expect(withCb.totalReturnSincePurchase, closeTo(0.0538, 0.001));

      final withoutCb = InvestmentHolding.fromMap(const {
        'ticker': 'EWZ',
        'quantity': 5,
        'market_value': 210.75,
        'asset_type': 'etf',
      });
      expect(withoutCb.costBasis, isNull);
      expect(withoutCb.totalReturnSincePurchase, isNull);
    });
  });

  group('PortfolioSummary', () {
    test('totalValue and holdingsCount aggregate across accounts', () {
      final summary = PortfolioSummary(accounts: [
        _account(
          id: 'a',
          totalValue: 320.76,
          holdings: [
            _holding(ticker: 'EWZ', marketValue: 210.75),
            _holding(ticker: 'NFLX', marketValue: 110.0),
            _holding(name: 'U S Dollar', marketValue: 0.01),
          ],
        ),
        _account(
          id: 'b',
          totalValue: 100,
          holdings: [_holding(ticker: 'X', marketValue: 100)],
        ),
      ]);
      expect(summary.totalValue, closeTo(420.76, 0.001));
      expect(summary.holdingsCount, 4);
      expect(summary.isEmpty, isFalse);
    });

    test('positions sort by value desc and carry share of portfolio', () {
      final summary = PortfolioSummary(accounts: [
        _account(totalValue: 1000, holdings: [
          _holding(ticker: 'SMALL', marketValue: 250),
          _holding(ticker: 'BIG', marketValue: 750),
        ]),
      ]);
      expect(summary.positions.first.holding.ticker, 'BIG');
      expect(summary.positions.first.share, closeTo(0.75, 0.001));
      expect(summary.positions.last.share, closeTo(0.25, 0.001));
    });

    test('overweight flags the largest holding of a concentrated account',
        () {
      final summary = PortfolioSummary(accounts: [
        _account(
          id: 'ira',
          totalValue: 320.76,
          concentrationRisk: 0.657, // real IRA fixture value
          holdings: [
            _holding(ticker: 'EWZ', marketValue: 210.75),
            _holding(ticker: 'NFLX', marketValue: 110.0),
          ],
        ),
        _account(
          id: 'ok',
          totalValue: 100,
          concentrationRisk: 0.25, // under the 30% limit
          holdings: [_holding(ticker: 'X', marketValue: 25)],
        ),
      ]);
      expect(summary.overweightAccounts.map((a) => a.accountId), ['ira']);
      final flagged = summary.positions
          .where((p) => p.overweight)
          .map((p) => p.holding.ticker);
      expect(flagged, ['EWZ']);
    });

    test('bestPerformer skips guarded holdings and picks the valid max',
        () {
      final summary = PortfolioSummary(accounts: [
        _account(totalValue: 7399.6 + 115.57, holdings: [
          // Guarded (the SBSI case) — must NOT win despite +24,558% raw.
          _holding(
              ticker: 'SBSI',
              quantity: 213,
              marketValue: 7397.49,
              costBasis: 30.0),
          _holding(ticker: 'ACHN', quantity: 1, marketValue: 2.11, costBasis: 1.5),
          _holding(
              ticker: 'BTC',
              quantity: 0.00293644,
              marketValue: 115.57268,
              costBasis: 120.03),
        ]),
      ]);
      expect(summary.bestPerformer?.holding.ticker, 'ACHN');
      expect(summary.worstDecliner?.holding.ticker, 'BTC');
    });

    test('portfolio-wide return is guarded when the aggregate is absurd',
        () {
      // The bad SBSI cost basis dominates the aggregate: (7397.49 + 2.11 −
      // 31.5) / 31.5 ≈ +23,290% — the portfolio figure must refuse too.
      final summary = PortfolioSummary(accounts: [
        _account(totalValue: 7399.6, holdings: [
          _holding(
              ticker: 'SBSI',
              quantity: 213,
              marketValue: 7397.49,
              costBasis: 30.0),
          _holding(ticker: 'ACHN', quantity: 1, marketValue: 2.11, costBasis: 1.5),
        ]),
      ]);
      expect(summary.portfolioReturnSincePurchase, isNull);
    });

    test('portfolio-wide return computes when data is plausible', () {
      final summary = PortfolioSummary(accounts: [
        _account(totalValue: 330, holdings: [
          _holding(ticker: 'A', quantity: 10, marketValue: 220, costBasis: 200),
          _holding(ticker: 'B', quantity: 5, marketValue: 110, costBasis: 100),
        ]),
      ]);
      expect(summary.portfolioReturnSincePurchase, closeTo(0.1, 0.001));
    });

    test('no cost_basis anywhere (current backend state) → all return '
        'stats null, structure stats still real', () {
      final summary = PortfolioSummary(accounts: [
        _account(totalValue: 500, holdings: [
          _holding(ticker: 'A', quantity: 10, marketValue: 300),
          _holding(ticker: 'B', quantity: 5, marketValue: 200),
        ]),
      ]);
      expect(summary.bestPerformer, isNull);
      expect(summary.worstDecliner, isNull);
      expect(summary.portfolioReturnSincePurchase, isNull);
      expect(summary.totalValue, 500);
      expect(summary.holdingsCount, 2);
    });

    test('empty portfolio', () {
      final summary = PortfolioSummary(accounts: const []);
      expect(summary.isEmpty, isTrue);
      expect(summary.totalValue, 0);
      expect(summary.positions, isEmpty);
      expect(summary.bestPerformer, isNull);
    });
  });
}
