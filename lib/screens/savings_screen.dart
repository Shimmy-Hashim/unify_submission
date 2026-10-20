import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../models/plaid_account.dart';
import '../services/firebase_service.dart';
import '../services/plaid_link_service.dart';
import '../services/plaid_service.dart';
import '../utils/app_styles.dart';
import '../utils/formatters.dart';
import '../widgets/accounts_refresh.dart';
import '../widgets/dashed_border.dart';
import 'account_transactions_screen.dart';

// ---- Audit palette, scoped to this screen -------------------------------
// These three shadow shared AppTheme tokens rather than editing them, the
// same containment Settings and Investments used: the shared values are read
// by Home and Insurance too, and those screens have their own audit tasks.

/// Secondary text — section labels, captions, "Available", the masked account
/// line. Replaces this file's former flat muted constant.
///
/// Light is the WCAG AA fix: [AppTheme.muted] (#AAAAAA) measures about 2.3:1
/// on white, against the 4.5:1 minimum. #6B6B6B clears it at 5.3:1, and is
/// the value the audit frame uses for every one of these labels.
///
/// Dark deliberately does not follow it down — #AAAAAA already clears 6.9:1
/// on the #1E1E1E card, and #6B6B6B there would fall to roughly 2.9:1,
/// breaking dark mode to fix light. A flat constant could not express that,
/// which is why it had to become a pair.
Color _muted(BuildContext context) =>
    AppTheme.isDark(context) ? AppTheme.muted : const Color(0xFF6B6B6B);

/// Primary ink. [AppTheme.primaryText]'s light value is pure `Colors.black`;
/// the brand value is #202020, sampled from the audit frame.
Color _primaryText(BuildContext context) =>
    AppTheme.isDark(context) ? Colors.white : AppColors.brandBlack;

/// The Total Savings card at the top. [AppTheme.heroSurface] is pure black in
/// light mode; the frame's card is #202020. Dark mode keeps heroSurface's own
/// #2A2A2A for the documented reason — #202020 on the #121212 background sits
/// under the #1E1E1E of the ordinary cards around it, so the one card meant to
/// stand out would recede instead.
Color _heroSurface(BuildContext context) =>
    AppTheme.isDark(context) ? const Color(0xFF2A2A2A) : AppColors.brandBlack;

// ---- Signed-money colours ------------------------------------------------
//
// Two pairs, because the discriminator is the SURFACE, not the theme. On a
// light card the darker pair is needed to clear WCAG AA as 12px bold text
// (#C62828 5.61:1, #2E7D32 5.13:1 on white — the old #E53935/#4CAF50 measured
// 4.23:1 and 2.78:1 there). On anything dark those same darks collapse to
// about 2.9-3.2:1, so a flat swap would have fixed the cards by breaking the
// hero card and dark mode — the trap AppTheme.muted's doc comment already
// warns about.

/// Signed money on a dark surface: the hero card in either theme, and every
/// card once the app is in dark mode. Unchanged values — they already clear
/// about 5.9:1 (green) on #202020 / #1E1E1E.
const _kOnDarkRed = Color(0xFFE53935);
const _kOnDarkGreen = Color(0xFF4CAF50);

/// Signed money on a light card. Both halves are already in this codebase as
/// chip foregrounds, so this introduces no new colour.
Color _moneyRed(BuildContext context) =>
    AppTheme.isDark(context) ? _kOnDarkRed : const Color(0xFFC62828);
Color _moneyGreen(BuildContext context) =>
    AppTheme.isDark(context) ? _kOnDarkGreen : const Color(0xFF2E7D32);

/// Savings tab — Figma node 53:147.
///
/// Real data: Total Savings summary, Emergency Fund estimator and Connected
/// Accounts (all computed from users/{uid}/plaidAccounts + transactions,
/// streamed so a new bank connection updates the UI automatically), plus the
/// functional Connect-to-Bank Plaid Link flow. Statement upload is
/// Insurance-only; this screen has no upload shell.
class SavingsScreen extends StatefulWidget {
  const SavingsScreen({super.key});

  @override
  State<SavingsScreen> createState() => _SavingsScreenState();
}

class _SavingsScreenState extends State<SavingsScreen> {
  int _months = 6;
  bool _connecting = false;

  // Transactions are refetched only when the accounts stream emits a new
  // snapshot (or after an exchange completes) — not on every rebuild.
  QuerySnapshot<Map<String, dynamic>>? _lastAccountsSnap;
  Future<Map<String, List<PlaidTransaction>>>? _txnsFuture;

  // ---- Connect Bank flow (Section 4) ----

  /// PlaidLinkService owns the Link lifecycle; the result comes back only to
  /// the screen that started it, so Investments and Settings stay silent even
  /// though they are alive in the IndexedStack.
  Future<void> _connectBank() async {
    if (_connecting) return;
    setState(() => _connecting = true);
    final result = await PlaidLinkService.instance.connect();
    if (!mounted) return;
    setState(() {
      _connecting = false;
      // Force a transactions refetch on the next accounts-stream build.
      if (result.status == PlaidConnectStatus.connected) {
        _lastAccountsSnap = null;
      }
    });
    if (result.status == PlaidConnectStatus.connected) {
      final institution = result.institutionName ?? 'your bank';
      _showMessage(
          'Connected $institution, ${result.accountCount} accounts synced.');
    } else if (result.status == PlaidConnectStatus.failed) {
      _showMessage(result.message ??
          'Bank connection was interrupted. Please try again.');
    }
    // cancelled: the user backed out. busy: another screen owns the session.
  }

  void _showMessage(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message)));
  }

  // ---- Data assembly ----

  Future<Map<String, List<PlaidTransaction>>> _fetchTransactions(
      String uid, List<PlaidAccount> accounts) async {
    final entries = await Future.wait(accounts.map((account) async {
      final snap = await FirebaseService.instance
          .plaidTransactions(uid, account.accountId)
          .orderBy('date', descending: true)
          .limit(100)
          .get();
      final txns = snap.docs
          .map((d) => PlaidTransaction.fromFirestore(d.id, d.data()))
          .toList();
      return MapEntry(account.accountId, txns);
    }));
    return Map.fromEntries(entries);
  }

  @override
  Widget build(BuildContext context) {
    final uid = FirebaseService.instance.currentUser?.uid;
    return Scaffold(
      backgroundColor: AppTheme.background(context),
      body: uid == null
          ? const Center(child: Text('Not signed in'))
          : SafeArea(
              child: StreamBuilder<QuerySnapshot<Map<String, dynamic>>>(
                stream:
                    FirebaseService.instance.plaidAccounts(uid).snapshots(),
                builder: (context, snap) {
                  if (snap.hasError) {
                    return Center(
                        child: Text('Could not load accounts: ${snap.error}'));
                  }
                  if (!snap.hasData) {
                    return const Center(child: CircularProgressIndicator());
                  }
                  final accounts = snap.data!.docs
                      .map((d) => PlaidAccount.fromFirestore(d.id, d.data()))
                      .toList();
                  if (!identical(_lastAccountsSnap, snap.data)) {
                    _lastAccountsSnap = snap.data;
                    _txnsFuture = _fetchTransactions(uid, accounts);
                  }
                  return FutureBuilder<Map<String, List<PlaidTransaction>>>(
                    future: _txnsFuture,
                    builder: (context, txnSnap) {
                      final summary = SavingsSummary(
                        accounts: accounts,
                        transactionsByAccount: txnSnap.data ?? const {},
                      );
                      return _buildContent(uid, summary);
                    },
                  );
                },
              ),
            ),
    );
  }

  Widget _buildContent(String uid, SavingsSummary summary) {
    // New balances and transactions land through the accounts stream above;
    // the transactions refetch follows the new snapshot on its own.
    return AccountsRefreshIndicator(
      onRefresh: PlaidService.instance.refresh,
      child: _buildList(uid, summary),
    );
  }

  Widget _buildList(String uid, SavingsSummary summary) {
    return ListView(
      // Always scrollable so a short list (no banks yet) still pulls.
      physics: const AlwaysScrollableScrollPhysics(),
      padding: const EdgeInsets.fromLTRB(10, 12, 10, 24),
      children: [
        _TotalSavingsCard(summary: summary),
        const SizedBox(height: 20),
        const _SectionLabel('EMERGENCY FUND ESTIMATOR'),
        const SizedBox(height: 8),
        _EmergencyFundCard(
          summary: summary,
          months: _months,
          onMonthsChanged: (m) => setState(() => _months = m),
        ),
        const SizedBox(height: 20),
        const _SectionLabel('CONNECTED ACCOUNTS'),
        const SizedBox(height: 8),
        if (summary.byInstitution.isEmpty)
          const _NoAccountsCard()
        else
          for (final entry in summary.byInstitution.entries) ...[
            _InstitutionCard(
              uid: uid,
              institutionName: entry.key,
              accounts: entry.value,
              summary: summary,
            ),
            const SizedBox(height: 14),
          ],
        const SizedBox(height: 6),
        _ConnectBankCard(connecting: _connecting, onTap: _connectBank),
      ],
    );
  }
}

class _SectionLabel extends StatelessWidget {
  const _SectionLabel(this.text);

  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(left: 4),
      child: Text(
        text,
        style: GoogleFonts.inter(
          fontWeight: FontWeight.w600,
          fontSize: 11,
          letterSpacing: 0.4,
          color: _muted(context),
        ),
      ),
    );
  }
}

/// Section 1 — black summary card, Figma node 53:184.
class _TotalSavingsCard extends StatelessWidget {
  const _TotalSavingsCard({required this.summary});

  final SavingsSummary summary;

  @override
  Widget build(BuildContext context) {
    final change = summary.monthChange;
    final n = summary.bankCount;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
      decoration: BoxDecoration(
        color: _heroSurface(context),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                'Total Savings',
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w500,
                  fontSize: 13,
                  color: Colors.white.withValues(alpha: 0.55),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Align(
                  alignment: Alignment.centerRight,
                  child: UpdatedLabel(
                    synced: summary.lastSynced,
                    style: GoogleFonts.inter(
                      fontSize: 11,
                      color: Colors.white.withValues(alpha: 0.55),
                    ),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 2),
          Text(
            formatSgd(summary.totalSavings),
            style: GoogleFonts.spaceGrotesk(
              fontWeight: FontWeight.w700,
              fontSize: 40,
              height: 1.1,
              color: Colors.white,
            ),
          ),
          // Net depository movement over the last 30 days of synced
          // transactions; omitted entirely when no data is in the window.
          if (change != null) ...[
            const SizedBox(height: 2),
            Text(
              '${formatSgdChange(change)} this month',
              style: GoogleFonts.spaceGrotesk(
                fontWeight: FontWeight.w700,
                fontSize: 12,
                color: change >= 0 ? _kOnDarkGreen : _kOnDarkRed,
              ),
            ),
          ],
          const SizedBox(height: 12),
          Divider(
            height: 1,
            thickness: 1,
            color: Colors.white.withValues(alpha: 0.12),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              _StatColumn(
                label: 'Accounts',
                value: '$n ${n == 1 ? 'Bank' : 'Banks'}',
              ),
              // Plaid's Transactions/Liabilities products expose no deposit
              // interest rates — shown as an em dash rather than invented.
              const _StatColumn(label: 'Avg. Interest', value: '-'),
              _StatColumn(
                label: 'Monthly Spent',
                value: formatSgd(summary.avgMonthlySpent),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _StatColumn extends StatelessWidget {
  const _StatColumn({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Column(
        children: [
          Text(
            label,
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w500,
              fontSize: 10,
              color: Colors.white.withValues(alpha: 0.45),
            ),
          ),
          const SizedBox(height: 4),
          FittedBox(
            fit: BoxFit.scaleDown,
            child: Text(
              value,
              style: GoogleFonts.spaceGrotesk(
                fontWeight: FontWeight.w700,
                fontSize: 13,
                color: Colors.white,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Section 2 — Emergency Fund estimator, Figma node 53:201. Fully real:
/// need = avg monthly spend × months, have = current total savings.
class _EmergencyFundCard extends StatelessWidget {
  const _EmergencyFundCard({
    required this.summary,
    required this.months,
    required this.onMonthsChanged,
  });

  final SavingsSummary summary;
  final int months;
  final ValueChanged<int> onMonthsChanged;

  @override
  Widget build(BuildContext context) {
    final need = summary.emergencyFundNeed(months);
    final have = summary.totalSavings;
    final progress = summary.emergencyFundProgress(months);
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
      decoration: BoxDecoration(
        color: AppTheme.surface(context),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [AppTheme.cardShadow(context)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'How many months do you want covered?',
            style: GoogleFonts.spaceGrotesk(
              fontWeight: FontWeight.w700,
              fontSize: 14,
              color: _primaryText(context),
            ),
          ),
          Text(
            summary.avgMonthlySpent > 0
                ? 'Based on your avg. monthly spending of '
                    '${formatSgd(summary.avgMonthlySpent)}'
                : 'Connect a bank so we can estimate your monthly spending',
            style: GoogleFonts.spaceGrotesk(
              fontWeight: FontWeight.w700,
              fontSize: 11,
              color: _muted(context),
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              for (final m in const [3, 6, 9, 12]) ...[
                Expanded(
                  child: _MonthPill(
                    months: m,
                    selected: m == months,
                    onTap: () => onMonthsChanged(m),
                  ),
                ),
                if (m != 12) const SizedBox(width: 8),
              ],
            ],
          ),
          const SizedBox(height: 12),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(vertical: 12),
            decoration: BoxDecoration(
              color: AppTheme.inset(context),
              borderRadius: BorderRadius.circular(14),
            ),
            child: IntrinsicHeight(
              child: Row(
                children: [
                  Expanded(
                    child: _NeedHaveColumn(
                      label: 'You need',
                      value: need > 0 ? formatSgd(need) : '-',
                      valueColor: need > 0 && need > have
                          ? _moneyRed(context)
                          : _primaryText(context),
                      caption: '$months months of expenses',
                    ),
                  ),
                  VerticalDivider(
                    width: 1,
                    thickness: 1,
                    // Its own value rather than AppTheme.divider: this rule
                    // sits on the recessed #F8F8F8 box, not on the card, so
                    // Figma gives it a slightly darker grey than the card
                    // dividers use.
                    color: AppTheme.isDark(context)
                        ? const Color(0xFF3A3A3A)
                        : const Color(0xFFE5E5E5),
                  ),
                  Expanded(
                    child: _NeedHaveColumn(
                      label: 'You have',
                      value: formatSgd(have),
                      valueColor: _primaryText(context),
                      caption: 'Current Savings',
                    ),
                  ),
                ],
              ),
            ),
          ),
          if (progress != null) ...[
            const SizedBox(height: 14),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: progress,
                minHeight: 7,
                backgroundColor: AppTheme.track(context),
                color:
                    progress >= 1 ? _moneyGreen(context) : _moneyRed(context),
              ),
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  '${(progress * 100).round()}% Funded',
                  style: GoogleFonts.inter(
                    fontWeight: FontWeight.w600,
                    fontSize: 11,
                    color: _muted(context),
                  ),
                ),
                Text(
                  progress >= 1
                      ? 'Fully funded'
                      : '${formatSgd(need - have)} to go',
                  style: GoogleFonts.inter(
                    fontWeight: FontWeight.w600,
                    fontSize: 11,
                    color: progress >= 1
                        ? _moneyGreen(context)
                        : _moneyRed(context),
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }
}

class _MonthPill extends StatelessWidget {
  const _MonthPill({
    required this.months,
    required this.selected,
    required this.onTap,
  });

  final int months;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        height: 44,
        decoration: BoxDecoration(
          // Lime marks the one pill the user has actively chosen, the same
          // convention Settings' toggles follow. It replaces an inverted
          // black/white fill, which read as a disabled or locked control
          // rather than a selection. Lime is fixed in both modes — it is a
          // brand accent, not a surface — so its ink is fixed with it.
          color: selected ? AppColors.limeAccent : AppTheme.iconTile(context),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              '$months',
              style: GoogleFonts.spaceGrotesk(
                fontWeight: FontWeight.w700,
                fontSize: 14,
                height: 1.1,
                // One ink for both lines of an unselected pill. They used to
                // carry two different greys (#888888 and a 53%-alpha
                // #888888 ≈ #BFBFBF, about 3.5:1 and 1.8:1 on the tile); the
                // frame draws them as one #6B6B6B.
                color: selected ? AppColors.brandBlack : _muted(context),
              ),
            ),
            Text(
              'months',
              style: GoogleFonts.inter(
                fontWeight: FontWeight.w700,
                fontSize: 9,
                height: 1.2,
                color: selected ? AppColors.brandBlack : _muted(context),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _NeedHaveColumn extends StatelessWidget {
  const _NeedHaveColumn({
    required this.label,
    required this.value,
    required this.valueColor,
    required this.caption,
  });

  final String label;
  final String value;
  final Color valueColor;
  final String caption;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          label,
          style: GoogleFonts.inter(
            fontWeight: FontWeight.w700,
            fontSize: 10,
            color: _muted(context),
          ),
        ),
        const SizedBox(height: 2),
        FittedBox(
          fit: BoxFit.scaleDown,
          child: Text(
            value,
            style: GoogleFonts.spaceGrotesk(
              fontWeight: FontWeight.w700,
              fontSize: 20,
              color: valueColor,
            ),
          ),
        ),
        const SizedBox(height: 2),
        Text(
          caption,
          style: GoogleFonts.inter(
            fontWeight: FontWeight.w700,
            fontSize: 10,
            color: _muted(context),
          ),
        ),
      ],
    );
  }
}

/// Section 3 empty state — shown instead of institution cards when the user
/// has not connected any bank yet.
class _NoAccountsCard extends StatelessWidget {
  const _NoAccountsCard();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
      decoration: BoxDecoration(
        color: AppTheme.surface(context),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Column(
        children: [
          // Material's own bank glyph, so the empty state uses the same
          // icon system as the nav bar and every other icon on this
          // screen. Outlined and muted because this is a decorative
          // illustration — the heading below it carries the message.
          Icon(Icons.account_balance_outlined,
              size: 28, color: _muted(context)),
          const SizedBox(height: 8),
          Text(
            'No banks connected yet',
            style: GoogleFonts.spaceGrotesk(
              fontWeight: FontWeight.w700,
              fontSize: 14,
              color: _primaryText(context),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Link your first account below to see balances and '
            'transactions here.',
            textAlign: TextAlign.center,
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w500,
              fontSize: 11,
              height: 1.5,
              color: _muted(context),
            ),
          ),
        ],
      ),
    );
  }
}

/// Section 3 — one card per institution, Figma nodes 53:331/239/353.
///
/// No icon badge on the row. The original design gave each bank its own
/// emoji, which Plaid has no per-institution source for, so every card ended
/// up showing the same generic 🏦 in the same light-blue tile — a column of
/// identical marks carrying no information. The audit frame drops it and lets
/// the institution name start at the card's own padding instead.
class _InstitutionCard extends StatelessWidget {
  const _InstitutionCard({
    required this.uid,
    required this.institutionName,
    required this.accounts,
    required this.summary,
  });

  final String uid;
  final String institutionName;
  final List<PlaidAccount> accounts;
  final SavingsSummary summary;

  @override
  Widget build(BuildContext context) {
    // Two separate figures, never one merged total: credit and loan balances
    // are debt, and adding them to "Available" told a test user they had
    // $189,523 to spend when $126,994 of that was mortgage, student loan and
    // card debt. See PlaidAccountTotals.
    final available = accounts.availableTotal;
    final ordered = accounts.depositoryFirst;
    // Where the debt group starts, so one sub-divider can be ruled between
    // the two groups. -1 when the card has no debt accounts at all.
    final firstDebt = ordered.indexWhere((a) => a.isDebt);
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 10, 16, 12),
      decoration: BoxDecoration(
        color: AppTheme.surface(context),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  institutionName,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.spaceGrotesk(
                    fontWeight: FontWeight.w700,
                    fontSize: 15,
                    color: _primaryText(context),
                  ),
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  _HeaderStat(
                    amount: available,
                    label: 'Available',
                    amountColor: _primaryText(context),
                    labelColor: _muted(context),
                  ),
                  // Omitted entirely when the institution has no credit or
                  // loan account, per the reference frame — Bank B and Bank C
                  // are depository-only and show "Available" alone rather
                  // than a $0.00 Owed cluttering the header.
                  if (accounts.hasDebt) ...[
                    const SizedBox(height: 4),
                    _HeaderStat(
                      amount: accounts.owedTotal,
                      label: 'Owed',
                      // Both halves red in the reference, not just the
                      // figure — #C62828 on a light card, which is what
                      // _moneyRed already resolves to.
                      amountColor: _moneyRed(context),
                      labelColor: _moneyRed(context),
                    ),
                  ],
                ],
              ),
            ],
          ),
          const SizedBox(height: 8),
          Divider(height: 1, thickness: 1, color: AppTheme.divider(context)),
          for (var i = 0; i < ordered.length; i++) ...[
            // Sub-divider between the depository group and the debt group.
            // Skipped when the debt group starts at index 0 (a loan- or
            // card-only institution), where it would sit directly under the
            // header rule and read as a doubled line.
            if (i == firstDebt && i != 0) ...[
              const SizedBox(height: 8),
              Divider(
                  height: 1, thickness: 1, color: AppTheme.divider(context)),
            ],
            const SizedBox(height: 8),
            _AccountBlock(uid: uid, account: ordered[i], summary: summary),
          ],
        ],
      ),
    );
  }
}

/// One amount-over-label cluster in an institution card's header. Two of
/// these stack when the institution carries debt: "Available" then "Owed".
class _HeaderStat extends StatelessWidget {
  const _HeaderStat({
    required this.amount,
    required this.label,
    required this.amountColor,
    required this.labelColor,
  });

  final double amount;
  final String label;
  final Color amountColor;
  final Color labelColor;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        Text(
          formatSgd(amount),
          style: GoogleFonts.spaceGrotesk(
            fontWeight: FontWeight.w700,
            fontSize: 17,
            color: amountColor,
          ),
        ),
        Text(
          label,
          style: GoogleFonts.inter(
            fontWeight: FontWeight.w500,
            fontSize: 10,
            color: labelColor,
          ),
        ),
      ],
    );
  }
}

class _AccountBlock extends StatelessWidget {
  const _AccountBlock({
    required this.uid,
    required this.account,
    required this.summary,
  });

  final String uid;
  final PlaidAccount account;
  final SavingsSummary summary;

  @override
  Widget build(BuildContext context) {
    final recent = summary.recentTwo(account.accountId);
    final isDebt = account.isDebt;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                account.maskedLabel,
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w500,
                  fontSize: 11,
                  // A debt account names itself in red, so "•••• 7734 ·
                  // Credit card" cannot be mistaken for another cash row.
                  color: isDebt ? _moneyRed(context) : _muted(context),
                ),
              ),
            ),
            Text(
              // rowBalance, not displayBalance: for a card this is the
              // amount owed, never the remaining credit line.
              formatSgd(account.rowBalance),
              style: GoogleFonts.spaceGrotesk(
                fontWeight: FontWeight.w700,
                fontSize: 12,
                color: _primaryText(context),
              ),
            ),
          ],
        ),
        for (final txn in recent) ...[
          const SizedBox(height: 6),
          _TransactionRow(txn: txn),
        ],
        const SizedBox(height: 6),
        Center(
          child: InkWell(
            onTap: () => Navigator.of(context).push(
              MaterialPageRoute(
                builder: (_) =>
                    AccountTransactionsScreen(uid: uid, account: account),
              ),
            ),
            child: Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              child: Text(
                'See all transactions →',
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w600,
                  fontSize: 11,
                  color: _muted(context),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _TransactionRow extends StatelessWidget {
  const _TransactionRow({required this.txn});

  final PlaidTransaction txn;

  @override
  Widget build(BuildContext context) {
    // Plaid sign convention (brief Section 8): positive = money out.
    final color =
        txn.isOutflow ? _moneyRed(context) : _moneyGreen(context);
    final amount = formatSignedSgdExact(txn.amount, negative: txn.isOutflow);
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Direction is carried by the glyph as well as by the colour, so
        // inflow and outflow stay distinguishable without relying on hue —
        // the 6px dot this replaces was colour-only, which is exactly the
        // signal a red/green colour-blind user cannot read.
        Padding(
          padding: const EdgeInsets.only(top: 1),
          child: Icon(
            txn.isOutflow ? Icons.call_made : Icons.call_received,
            size: 14,
            color: color,
          ),
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
              Text(
                [
                  if (txn.date != null) formatShortDate(txn.date!),
                  if (txn.category.isNotEmpty) txn.category.first,
                ].join(' · '),
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w500,
                  fontSize: 10,
                  // Was AppTheme.faintText, whose light value (#BBBBBB) sits
                  // at about 1.9:1 on the card — the least readable text on
                  // the screen. _muted is the same #6B6B6B every other
                  // secondary label on this screen now uses.
                  color: _muted(context),
                ),
              ),
            ],
          ),
        ),
        Text(
          amount,
          style: GoogleFonts.spaceGrotesk(
            fontWeight: FontWeight.w700,
            fontSize: 12,
            color: color,
          ),
        ),
      ],
    );
  }
}

/// Section 4 — Connect to Bank(s), Figma node 53:373. The dashed card
/// doubles as the progress indicator while the link-token fetch, Plaid Link
/// session and server-side exchange + initial sync are running.
class _ConnectBankCard extends StatelessWidget {
  const _ConnectBankCard({required this.connecting, required this.onTap});

  final bool connecting;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: connecting ? null : onTap,
      borderRadius: BorderRadius.circular(20),
      child: CustomPaint(
        painter: DashedBorderPainter(
          color: AppTheme.dashedBorder(context),
          radius: 20,
          strokeWidth: 1.5,
        ),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 12),
          decoration: BoxDecoration(
            color: AppTheme.surface(context),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Row(
            children: [
              Container(
                width: 36,
                height: 36,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: AppTheme.iconTile(context),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: connecting
                    ? SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: _primaryText(context),
                        ),
                      )
                    : Icon(Icons.add,
                        size: 20, color: _primaryText(context)),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Connect to Bank(s)',
                      style: GoogleFonts.spaceGrotesk(
                        fontWeight: FontWeight.w700,
                        fontSize: 14,
                        color: _primaryText(context),
                      ),
                    ),
                    Text(
                      connecting
                          ? 'Connecting your bank. This can take a few '
                              'seconds…'
                          : 'Link via Plaid · Secure & encrypted',
                      style: GoogleFonts.inter(
                        fontWeight: FontWeight.w700,
                        fontSize: 11,
                        color: _muted(context),
                      ),
                    ),
                  ],
                ),
              ),
              Icon(Icons.chevron_right, color: _muted(context)),
            ],
          ),
        ),
      ),
    );
  }
}

