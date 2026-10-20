import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../models/investment_holding.dart';
import '../services/firebase_service.dart';
import '../utils/app_styles.dart';
import '../utils/formatters.dart';

// Same two-pair scheme as savings_screen.dart: these amounts sit on a light
// card, where the darker halves are what clear WCAG AA as 12px bold text
// (#C62828 5.61:1, #2E7D32 5.13:1 on white). On a dark card those darks fall
// to about 3:1, so the dark branch keeps the brighter originals.
const _kOnDarkCashOutRed = Color(0xFFE53935);
const _kOnDarkCashInGreen = Color(0xFF4CAF50);

Color _cashOutRed(BuildContext context) =>
    AppTheme.isDark(context) ? _kOnDarkCashOutRed : const Color(0xFFC62828);
Color _cashInGreen(BuildContext context) =>
    AppTheme.isDark(context) ? _kOnDarkCashInGreen : const Color(0xFF2E7D32);

/// Secondary text, matching the token of the same name on savings_screen and
/// account_transactions_screen. [AppTheme.faintText]'s light value (#BBBBBB)
/// measures about 1.9:1 on the card — the least readable text on the screen.
Color _muted(BuildContext context) =>
    AppTheme.isDark(context) ? AppTheme.muted : const Color(0xFF6B6B6B);

/// Plaid returns unrounded float quantities (e.g. 0.7388014749727547) —
/// trim to at most 4 decimals for display.
String _formatQuantity(double qty) {
  if (qty == qty.roundToDouble()) return qty.round().toString();
  var s = qty.toStringAsFixed(4);
  while (s.endsWith('0')) {
    s = s.substring(0, s.length - 1);
  }
  return s.endsWith('.') ? s.substring(0, s.length - 1) : s;
}

/// Basic functional list of synced investment transactions across all of
/// the user's investment accounts — the "See all transactions →" target.
/// Deliberately simple, same treatment as the Savings equivalent: a
/// fully-designed transactions-detail screen is a follow-up (no Figma
/// design exists for it).
class InvestmentTransactionsScreen extends StatelessWidget {
  const InvestmentTransactionsScreen({
    super.key,
    required this.uid,
    required this.accounts,
  });

  final String uid;
  final List<InvestmentAccount> accounts;

  Future<List<(InvestmentAccount, InvestmentTransaction)>> _fetch() async {
    final all = <(InvestmentAccount, InvestmentTransaction)>[];
    await Future.wait(accounts.map((account) async {
      final snap = await FirebaseService.instance
          .investmentTransactions(uid, account.accountId)
          .orderBy('date', descending: true)
          .limit(200)
          .get();
      for (final d in snap.docs) {
        all.add(
            (account, InvestmentTransaction.fromFirestore(d.id, d.data())));
      }
    }));
    all.sort((a, b) {
      final da = a.$2.date, db = b.$2.date;
      if (da == null || db == null) return da == null ? 1 : -1;
      return db.compareTo(da);
    });
    return all;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background(context),
      appBar: AppBar(
        backgroundColor: AppTheme.background(context),
        title: Text(
          'Investment Transactions',
          style: GoogleFonts.spaceGrotesk(
            fontWeight: FontWeight.w700,
            fontSize: 18,
            color: AppTheme.primaryText(context),
          ),
        ),
      ),
      body: FutureBuilder<List<(InvestmentAccount, InvestmentTransaction)>>(
        future: _fetch(),
        builder: (context, snap) {
          if (snap.hasError) {
            return Center(
                child: Text('Could not load transactions: ${snap.error}'));
          }
          if (!snap.hasData) {
            return const Center(child: CircularProgressIndicator());
          }
          final rows = snap.data!;
          if (rows.isEmpty) {
            return const Center(
                child: Text('No investment transactions synced yet.'));
          }
          return ListView.separated(
            padding: const EdgeInsets.all(10),
            itemCount: rows.length,
            separatorBuilder: (_, index) => const SizedBox(height: 6),
            itemBuilder: (context, i) =>
                _TransactionTile(account: rows[i].$1, txn: rows[i].$2),
          );
        },
      ),
    );
  }
}

class _TransactionTile extends StatelessWidget {
  const _TransactionTile({required this.account, required this.txn});

  final InvestmentAccount account;
  final InvestmentTransaction txn;

  @override
  Widget build(BuildContext context) {
    final type = txn.type.isEmpty
        ? 'Transaction'
        : txn.type[0].toUpperCase() + txn.type.substring(1);
    final title = [
      type,
      if (txn.ticker != null && txn.ticker!.isNotEmpty) txn.ticker!,
    ].join(' · ');
    final subtitle = [
      if (txn.date != null) formatShortDate(txn.date!),
      account.accountName,
      if (txn.quantity != null && txn.price != null)
        '${_formatQuantity(txn.quantity!)} @ ${formatSgdExact(txn.price!)}',
    ].join(' · ');
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: AppTheme.surface(context),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          // Direction is carried by the glyph as well as by the colour, the
          // same pairing the Savings and Account transaction rows use.
          //
          // Driven by isCashOut — the SAME predicate the amount's colour and
          // sign already use — so the icon can never disagree with the number
          // beside it. That makes this screen's arrow mean CASH direction,
          // not portfolio direction: a Buy debits cash and points out (red)
          // even though it increases holdings, and a Sell or Deposit credits
          // cash and points in (green). Matching the existing colour logic
          // was the point; introducing a second, conflicting notion of
          // "incoming" here would make the row contradict itself.
          Icon(
            txn.isCashOut ? Icons.call_made : Icons.call_received,
            size: 14,
            color:
                txn.isCashOut ? _cashOutRed(context) : _cashInGreen(context),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.inter(
                    fontWeight: FontWeight.w500,
                    fontSize: 12,
                    color: AppTheme.secondaryText(context),
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  subtitle,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.inter(
                    fontSize: 10,
                    color: _muted(context),
                  ),
                ),
              ],
            ),
          ),
          Text(
            // Plaid investments sign convention (Section 8): positive =
            // cash debited (e.g. a buy), negative = cash credited.
            formatSignedSgdExact(txn.amount, negative: txn.isCashOut),
            style: GoogleFonts.spaceGrotesk(
              fontWeight: FontWeight.w700,
              fontSize: 12,
              color:
                  txn.isCashOut ? _cashOutRed(context) : _cashInGreen(context),
            ),
          ),
        ],
      ),
    );
  }
}
