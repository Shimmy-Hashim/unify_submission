import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../models/plaid_account.dart';
import '../services/firebase_service.dart';
import '../utils/app_styles.dart';
import '../utils/formatters.dart';

// Same two-pair scheme as savings_screen.dart, and the same reason: the rows
// sit on a light card, where the darker halves are what clear WCAG AA as 12px
// bold text (#C62828 5.61:1, #2E7D32 5.13:1), while on a dark card those same
// darks fall to about 3:1 — so the dark branch keeps the brighter originals.
//
// Duplicated rather than shared: this screen's row is its own private widget
// (_TransactionTile), not savings_screen's _TransactionRow. Unifying the two
// is a real refactor and is not this task.
const _kOnDarkRed = Color(0xFFE53935);
const _kOnDarkGreen = Color(0xFF4CAF50);

Color _moneyRed(BuildContext context) =>
    AppTheme.isDark(context) ? _kOnDarkRed : const Color(0xFFC62828);
Color _moneyGreen(BuildContext context) =>
    AppTheme.isDark(context) ? _kOnDarkGreen : const Color(0xFF2E7D32);

/// Secondary text, matching savings_screen's token of the same name.
/// AppTheme.faintText's light value (#BBBBBB) sits at about 1.9:1 on the card.
Color _muted(BuildContext context) =>
    AppTheme.isDark(context) ? AppTheme.muted : const Color(0xFF6B6B6B);

/// Basic functional list of every synced transaction for one account —
/// the "See all transactions →" target. Deliberately simple: a dedicated,
/// fully-designed transactions-detail screen is a follow-up (no Figma
/// design exists for it yet).
class AccountTransactionsScreen extends StatelessWidget {
  const AccountTransactionsScreen({
    super.key,
    required this.uid,
    required this.account,
  });

  final String uid;
  final PlaidAccount account;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background(context),
      appBar: AppBar(
        backgroundColor: AppTheme.background(context),
        title: Text(
          account.institutionName ?? account.accountName,
          style: GoogleFonts.spaceGrotesk(
            fontWeight: FontWeight.w700,
            fontSize: 18,
            color: AppTheme.primaryText(context),
          ),
        ),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(20),
          child: Padding(
            padding: const EdgeInsets.only(bottom: 6),
            child: Text(
              account.maskedLabel,
              style: GoogleFonts.inter(
                fontSize: 11,
                color: AppTheme.muted,
              ),
            ),
          ),
        ),
      ),
      body: StreamBuilder<QuerySnapshot<Map<String, dynamic>>>(
        stream: FirebaseService.instance
            .plaidTransactions(uid, account.accountId)
            .orderBy('date', descending: true)
            .snapshots(),
        builder: (context, snap) {
          if (snap.hasError) {
            return Center(
                child: Text('Could not load transactions: ${snap.error}'));
          }
          if (!snap.hasData) {
            return const Center(child: CircularProgressIndicator());
          }
          final txns = snap.data!.docs
              .map((d) => PlaidTransaction.fromFirestore(d.id, d.data()))
              .toList();
          if (txns.isEmpty) {
            return const Center(child: Text('No transactions synced yet.'));
          }
          return ListView.separated(
            padding: const EdgeInsets.all(10),
            itemCount: txns.length,
            separatorBuilder: (_, index) => const SizedBox(height: 6),
            itemBuilder: (context, i) => _TransactionTile(txn: txns[i]),
          );
        },
      ),
    );
  }
}

class _TransactionTile extends StatelessWidget {
  const _TransactionTile({required this.txn});

  final PlaidTransaction txn;

  @override
  Widget build(BuildContext context) {
    final subtitle = [
      if (txn.date != null) formatShortDate(txn.date!),
      if (txn.category.isNotEmpty) txn.category.first,
      if (txn.pending) 'Pending',
    ].join(' · ');
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: AppTheme.surface(context),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          // Direction is carried by the glyph as well as by the colour, so
          // inflow and outflow stay distinguishable without relying on hue.
          Icon(
            txn.isOutflow ? Icons.call_made : Icons.call_received,
            size: 14,
            color: txn.isOutflow ? _moneyRed(context) : _moneyGreen(context),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  txn.name ?? 'Transaction',
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
                  style: GoogleFonts.inter(
                    fontSize: 10,
                    color: _muted(context),
                  ),
                ),
              ],
            ),
          ),
          Text(
            // Plaid sign convention: positive = money out.
            formatSignedSgdExact(txn.amount, negative: txn.isOutflow),
            style: GoogleFonts.spaceGrotesk(
              fontWeight: FontWeight.w700,
              fontSize: 12,
              color: txn.isOutflow ? _moneyRed(context) : _moneyGreen(context),
            ),
          ),
        ],
      ),
    );
  }
}
