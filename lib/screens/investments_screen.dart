import 'dart:math' as math;

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../models/investment_holding.dart';
import '../services/firebase_service.dart';
import '../services/plaid_link_service.dart';
import '../services/plaid_service.dart';
import '../utils/app_styles.dart';
import '../utils/formatters.dart';
import '../widgets/accounts_refresh.dart';
import '../widgets/dashed_border.dart';
import 'investment_transactions_screen.dart';

// ---- Audit palette, scoped to this screen -------------------------------
// These three shadow shared AppTheme tokens rather than editing them. The
// shared values are consumed by Home, Savings and Insurance too, and those
// screens have their own audit tasks; changing the tokens here would restyle
// them silently. Same containment Settings used.

/// Secondary text — section labels, holding subtitles, "% of portfolio".
///
/// Light is the WCAG AA fix: [AppTheme.muted] (#AAAAAA) measures about 2.3:1
/// on white, against the 4.5:1 minimum. #6B6B6B clears it at 5.3:1, and is
/// what the audit frame uses.
///
/// Dark deliberately does not follow it down — #AAAAAA already clears 6.9:1
/// on the #1E1E1E card, and #6B6B6B there would fall to roughly 2.9:1,
/// breaking dark mode to fix light. The old flat constant could not express
/// that, which is why it had to become a pair.
Color _muted(BuildContext context) =>
    AppTheme.isDark(context) ? AppTheme.muted : const Color(0xFF6B6B6B);

/// Primary ink. [AppTheme.primaryText]'s light value is pure `Colors.black`;
/// the brand value is #202020, confirmed against the audit frame.
Color _primaryText(BuildContext context) =>
    AppTheme.isDark(context) ? Colors.white : AppColors.brandBlack;

/// The dark hero card at the top. [AppTheme.heroSurface] is pure black in
/// light mode; the frame's card is #202020. Dark mode keeps heroSurface's own
/// #2A2A2A for the documented reason — #202020 on the #121212 background sits
/// under the #1E1E1E of the ordinary cards around it, so the one card meant to
/// stand out would recede instead.
Color _heroSurface(BuildContext context) =>
    AppTheme.isDark(context) ? const Color(0xFF2A2A2A) : AppColors.brandBlack;

// Gain/loss on the HERO CARD keep their hues in both modes: that card is dark
// either way, and these two clear ~5.9:1 (green) on it. They are deliberately
// NOT the table's colours — see below.
const _kGainGreen = Color(0xFF4CAF50);
const _kLossRed = Color(0xFFE53935);

/// Signed return inside the Holdings table, which sits on a light card.
///
/// Same two-pair scheme the transaction rows use, for the same reason: on
/// white, #2E7D32/#C62828 clear WCAG AA as small bold text (5.13:1 / 5.61:1)
/// where the hero pair above measures 2.78:1 and 4.23:1; on a #1E1E1E card
/// those darks collapse to about 3:1, so the dark branch keeps the brighter
/// originals. The audit frame draws the table's positive returns as #2E7D32.
Color _moneyGreen(BuildContext context) =>
    AppTheme.isDark(context) ? _kGainGreen : const Color(0xFF2E7D32);
Color _moneyRed(BuildContext context) =>
    AppTheme.isDark(context) ? _kLossRed : const Color(0xFFC62828);

// Badge palettes lifted from the Figma risk pills. These are the LIGHT pairs;
// every use goes through AppTheme.chip, which inverts them for dark mode.
const _kBadgeRedBg = Color(0xFFFFEBEE);
const _kBadgeRedFg = Color(0xFFC62828);
const _kBadgeAmberBg = Color(0xFFFFF3E0);
const _kBadgeAmberFg = Color(0xFFE65100);
const _kBadgeGreyBg = Color(0xFFF2F3F5);
const _kBadgeGreyFg = Color(0xFF555555);

// The seven-colour categorical palette (_kHoldingColors / _kOtherGrey /
// _holdingColor) lived here. It had exactly two consumers, the donut's slices
// and the per-row swatch, and the audit removed both: the colours were
// assigned by list position, so they encoded nothing about the stock itself.
// Holdings are plain text columns now, matching the moomoo reference.

String _formatPct(double fraction, {bool signed = false}) {
  final pct = (fraction * 100).toStringAsFixed(1);
  return signed && fraction >= 0 ? '+$pct%' : '$pct%';
}

String _formatQuantity(double qty) {
  if (qty == qty.roundToDouble()) return qty.round().toString();
  var s = qty.toStringAsFixed(4);
  while (s.endsWith('0')) {
    s = s.substring(0, s.length - 1);
  }
  return s.endsWith('.') ? s.substring(0, s.length - 1) : s;
}

/// Investments tab — Figma node 78:178.
///
/// Real data: portfolio value/holdings/allocation/overweight alerts (all
/// from users/{uid}/investmentAccounts, streamed) and the Connect Broker
/// Plaid Link flow. Since-purchase return figures are a deliberate,
/// labeled substitution for the design's month-over-month stats (no daily
/// price history exists) and render "—" until the backend syncs
/// cost_basis into the holdings array (flagged Phase 3 follow-up). Sector
/// and geographic cards are honest placeholders — Plaid provides no such
/// data (confirmed empirically). Statement upload is Insurance-only; this
/// screen has no upload shell.
class InvestmentsScreen extends StatefulWidget {
  const InvestmentsScreen({super.key});

  @override
  State<InvestmentsScreen> createState() => _InvestmentsScreenState();
}

class _InvestmentsScreenState extends State<InvestmentsScreen> {
  bool _connecting = false;

  // ---- Connect Broker flow (Section 5) — same chain as Savings' Connect
  // Bank; the link token already requests the investments product and the
  // exchange endpoint runs the initial investments sync server-side.

  /// PlaidLinkService owns the Link lifecycle; the result comes back only to
  /// the screen that started it, so Savings and Settings stay silent even
  /// though they are alive in the IndexedStack.
  Future<void> _connectBroker() async {
    if (_connecting) return;
    setState(() => _connecting = true);
    final result = await PlaidLinkService.instance.connect();
    if (!mounted) return;
    setState(() => _connecting = false);
    if (result.status == PlaidConnectStatus.connected) {
      final institution = result.institutionName ?? 'your broker';
      _showMessage('Connected $institution, investment accounts synced.');
    } else if (result.status == PlaidConnectStatus.failed) {
      _showMessage(result.message ??
          'Broker connection was interrupted. Please try again.');
    }
    // cancelled: the user backed out. busy: another screen owns the session.
  }

  void _showMessage(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message)));
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
                stream: FirebaseService.instance
                    .investmentAccounts(uid)
                    .snapshots(),
                builder: (context, snap) {
                  if (snap.hasError) {
                    return Center(
                        child: Text('Could not load investments: '
                            '${snap.error}'));
                  }
                  if (!snap.hasData) {
                    return const Center(child: CircularProgressIndicator());
                  }
                  final summary = PortfolioSummary(
                    accounts: [
                      for (final d in snap.data!.docs)
                        InvestmentAccount.fromFirestore(d.id, d.data()),
                    ],
                  );
                  return _buildContent(uid, summary);
                },
              ),
            ),
    );
  }

  Widget _buildContent(String uid, PortfolioSummary summary) {
    // Refreshed holdings land through the investmentAccounts stream above.
    return AccountsRefreshIndicator(
      onRefresh: PlaidService.instance.refresh,
      child: _buildList(uid, summary),
    );
  }

  Widget _buildList(String uid, PortfolioSummary summary) {
    return ListView(
      // Always scrollable so a short list (no investments yet) still pulls.
      physics: const AlwaysScrollableScrollPhysics(),
      padding: const EdgeInsets.fromLTRB(10, 12, 10, 24),
      children: [
        _PortfolioValueCard(summary: summary),
        const SizedBox(height: 20),
        const _SectionLabel('RISK ALERTS'),
        const SizedBox(height: 8),
        if (summary.isEmpty)
          const _NoInvestmentsCard()
        else ...[
          _RiskAlertsStrip(summary: summary),
          // The Holdings Breakdown donut sat here. Removed by the audit: every
          // slice's percentage is already printed as exact text on the holding
          // card below, so the chart restated the same numbers less precisely.
          // HOLDINGS now follows Risk Alerts directly, on the same 20pt rhythm
          // every other section break uses.
          //
          // 8, not 20, and only here: the strip above carries 12pt of internal
          // bottom padding so its cards' shadows are not clipped (see
          // _RiskAlertsStrip). 12 + 8 keeps the card-bottom-to-label distance
          // at the same 20 as every other section break — measure this gap
          // from the card, not from the SizedBox.
          const SizedBox(height: 8),
          const _SectionLabel('HOLDINGS'),
          const SizedBox(height: 8),
          _HoldingsTable(positions: summary.positions),
          const SizedBox(height: 10),
          // Relocated out of the deleted donut card, not dropped. This is the
          // only route to InvestmentTransactionsScreen anywhere in the app —
          // deleting the card wholesale would have stranded that screen with
          // no entry point and no compile error to reveal it.
          _SeeAllTransactions(uid: uid, summary: summary),
        ],
        const SizedBox(height: 6),
        _ConnectBrokerCard(connecting: _connecting, onTap: _connectBroker),
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

/// Section 1 — black summary card, Figma node 78:280.
///
/// The design's "+$2,340 (5.5%) this month" indicator is omitted entirely:
/// month-over-month market movement needs daily price history Plaid does
/// not provide, and approximating it from transactions would fabricate a
/// number. Best Performer / Total Return are since-purchase figures
/// (labeled "all-time"), not monthly.
class _PortfolioValueCard extends StatelessWidget {
  const _PortfolioValueCard({required this.summary});

  final PortfolioSummary summary;

  @override
  Widget build(BuildContext context) {
    final best = summary.bestPerformer;
    final totalReturn = summary.portfolioReturnSincePurchase;
    final n = summary.holdingsCount;
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
                'Total Portfolio Value',
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
            formatSgd(summary.totalValue),
            style: GoogleFonts.spaceGrotesk(
              fontWeight: FontWeight.w700,
              fontSize: 40,
              height: 1.1,
              color: Colors.white,
            ),
          ),
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
                label: 'Holdings',
                value: '$n ${n == 1 ? 'Holding' : 'Holdings'}',
              ),
              _StatColumn(
                label: 'Best Performer (all-time)',
                value: best?.holding.shortLabel ?? '-',
              ),
              _StatColumn(
                label: 'Total Return (all-time)',
                // 'n/a', not '-': a bare hyphen reads as a minus sign in
                // a column of signed returns.
                value: totalReturn == null
                    ? 'n/a'
                    : _formatPct(totalReturn, signed: true),
                valueColor: totalReturn == null
                    ? Colors.white
                    : totalReturn >= 0
                        ? _kGainGreen
                        : _kLossRed,
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _StatColumn extends StatelessWidget {
  const _StatColumn({
    required this.label,
    required this.value,
    this.valueColor = Colors.white,
  });

  final String label;
  final String value;
  final Color valueColor;

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Column(
        children: [
          FittedBox(
            fit: BoxFit.scaleDown,
            child: Text(
              label,
              style: GoogleFonts.inter(
                fontWeight: FontWeight.w500,
                fontSize: 10,
                color: Colors.white.withValues(alpha: 0.45),
              ),
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
                color: valueColor,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Section 2 — horizontally scrolling alert cards, Figma node 82:332.
/// Built dynamically from what is actually computable: overweight cards
/// from the backend-cached concentration_risk, a since-purchase decliner
/// (labeled substitution), and two honest no-data placeholders for the
/// sector/geography categories Plaid has no data for.
class _RiskAlertsStrip extends StatelessWidget {
  const _RiskAlertsStrip({required this.summary});

  final PortfolioSummary summary;

  @override
  Widget build(BuildContext context) {
    final cards = <Widget>[
      for (final account in summary.overweightAccounts)
        _AlertCard(
          emoji: '⚠️',
          title: '${account.largestHolding!.shortLabel} Overweight',
          subtitle:
              '${_formatPct(account.concentrationRisk ?? 0)} of '
              '${account.accountName}. Limit is '
              '${(kConcentrationLimit * 100).round()}%',
          badgeText: 'High Risk',
          badgeBg: _kBadgeRedBg,
          badgeFg: _kBadgeRedFg,
        ),
      if (summary.worstDecliner != null)
        _AlertCard(
          emoji: '📉',
          title: '${summary.worstDecliner!.holding.shortLabel} Declining',
          subtitle: 'Down '
              '${(summary.worstDecliner!.holding.totalReturnSincePurchase!.abs() * 100).toStringAsFixed(1)}% '
              'since purchase, worst performer',
          badgeText: 'Since purchase',
          badgeBg: _kBadgeAmberBg,
          badgeFg: _kBadgeAmberFg,
        ),
      // Plaid's security data carries no sector/industry values (confirmed
      // empirically — fields exist but are null), and nothing geographic at
      // all. Neutral no-data cards, same honest-gap treatment as Savings'
      // "Avg. Interest —".
      const _AlertCard(
        emoji: '📊',
        title: 'Sector Mix',
        subtitle: 'Sector data unavailable from your broker connection',
        badgeText: 'No data',
        badgeBg: _kBadgeGreyBg,
        badgeFg: _kBadgeGreyFg,
      ),
      const _AlertCard(
        emoji: '🌍',
        title: 'Geographic Spread',
        subtitle: 'Geographic data unavailable from your broker connection',
        badgeText: 'No data',
        badgeBg: _kBadgeGreyBg,
        badgeFg: _kBadgeGreyFg,
      ),
    ];
    // The 12pt of bottom padding is load-bearing, not a margin.
    //
    // This box used to be exactly 104 — the card's own height — so the
    // viewport's edge landed precisely on the card's edge and clipped
    // AppTheme.cardShadow (offset 0,4 + blur 4) out of existence. Measured
    // against a Holdings card, which fades through a ~25px gradient: these
    // cards went straight from #FFFFFF to the #F2F3F5 page background with no
    // gradient at all, so they read as flat panels pressed into a band rather
    // than the floating cards every other section uses.
    //
    // The padding gives the shadow somewhere to paint; the card itself is
    // still 104 (children are laid out against the viewport height minus
    // padding). No top padding is needed — the shadow is offset downward by
    // exactly its blur radius, so it starts at the card's top edge and never
    // paints above it. Horizontal clipping stays at 2: cards SHOULD be cut off
    // at the sides, that is the scroll affordance.
    return SizedBox(
      height: 104 + 12,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.fromLTRB(2, 0, 2, 12),
        itemCount: cards.length,
        separatorBuilder: (_, index) => const SizedBox(width: 8),
        itemBuilder: (context, i) => cards[i],
      ),
    );
  }
}

class _AlertCard extends StatelessWidget {
  const _AlertCard({
    required this.emoji,
    required this.title,
    required this.subtitle,
    required this.badgeText,
    required this.badgeBg,
    required this.badgeFg,
  });

  final String emoji;
  final String title;
  final String subtitle;
  final String badgeText;
  final Color badgeBg;
  final Color badgeFg;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 172,
      padding: const EdgeInsets.fromLTRB(14, 8, 14, 10),
      decoration: BoxDecoration(
        color: AppTheme.surface(context),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [AppTheme.cardShadow(context)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '$emoji $title',
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w600,
              fontSize: 11,
              color: _primaryText(context),
            ),
          ),
          const SizedBox(height: 4),
          Expanded(
            child: Text(
              subtitle,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.inter(
                fontWeight: FontWeight.w600,
                fontSize: 10,
                height: 1.35,
                color: AppTheme.subtleBody(context),
              ),
            ),
          ),
          const SizedBox(height: 6),
          _Badge(text: badgeText, background: badgeBg, foreground: badgeFg),
        ],
      ),
    );
  }
}

/// Takes the Figma LIGHT pair and resolves it for the active theme — the pale
/// fill / dark ink combination the mock uses is unreadable as-is on a dark
/// card, so [AppTheme.chip] inverts it while keeping the hue.
class _Badge extends StatelessWidget {
  const _Badge({
    required this.text,
    required this.background,
    required this.foreground,
  });

  final String text;
  final Color background;
  final Color foreground;

  @override
  Widget build(BuildContext context) {
    final chip = AppTheme.chip(
      context,
      background: background,
      foreground: foreground,
    );
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
      decoration: BoxDecoration(
        color: chip.background,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(
        text,
        style: GoogleFonts.inter(
          fontWeight: FontWeight.w600,
          fontSize: 9,
          color: chip.foreground,
        ),
      ),
    );
  }
}

/// The "See all transactions →" action, lifted out of the deleted Holdings
/// Breakdown card so it survives the donut's removal.
///
/// It is the app's ONLY entry point to [InvestmentTransactionsScreen] — no
/// other widget constructs that route — so deleting its former host card
/// without rehoming it would have quietly orphaned a whole screen. It now
/// closes the Holdings list rather than the allocation card.
class _SeeAllTransactions extends StatelessWidget {
  const _SeeAllTransactions({required this.uid, required this.summary});

  final String uid;
  final PortfolioSummary summary;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: InkWell(
        onTap: () => Navigator.of(context).push(
          MaterialPageRoute(
            builder: (_) => InvestmentTransactionsScreen(
              uid: uid,
              accounts: summary.accounts,
            ),
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          // Was AppTheme.linkText — 0x88888888 resolves to about #B4B4B4 on
          // white, roughly 2.2:1. It is secondary text in the same family the
          // audit targeted, so it moves to the screen's #6B6B6B with the rest.
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
    );
  }
}

/// Section 4 — the Holdings table, Figma node 374:710 ("Holdings Table Card").
///
/// Replaces the former card-per-holding list. The card list spent ~62pt of
/// height per holding to show four numbers; the table shows the same four in a
/// 40pt row and gains a comparison the list could not offer — you can read
/// down a column.
///
/// Still absent, unchanged by this redesign and for the same reason as before:
/// a day-change column and a sector column. Plaid exposes neither daily price
/// history nor sector data, and this screen has already declined to invent
/// both. RETURN here is return SINCE PURCHASE, not today's move.
class _HoldingsTable extends StatefulWidget {
  const _HoldingsTable({required this.positions});

  final List<PortfolioPosition> positions;

  @override
  State<_HoldingsTable> createState() => _HoldingsTableState();
}

/// Which column the table is ordered by.
enum _SortKey { symbol, value, returnPct, share }

class _HoldingsTableState extends State<_HoldingsTable> {
  // Value, descending — biggest position first, which is what the Risk Alerts
  // strip above is talking about.
  _SortKey _sort = _SortKey.value;
  bool _descending = true;

  void _onHeaderTap(_SortKey key) {
    setState(() {
      if (_sort == key) {
        _descending = !_descending;
      } else {
        _sort = key;
        // A fresh column starts descending for the three numeric ones (bigger
        // first reads as "top of the table"), ascending for SYMBOL, where
        // descending alphabetical would be a strange default.
        _descending = key != _SortKey.symbol;
      }
    });
  }

  /// A holding with no usable return sorts to the BOTTOM in both directions
  /// rather than flipping between the two ends of the table. `null` here means
  /// "not computable" (cost basis missing, or the model's plausibility guard
  /// rejected it), not "zero" — ordering it among real numbers would invent a
  /// rank for a value the app deliberately refuses to state.
  List<PortfolioPosition> get _sorted {
    final rows = [...widget.positions];
    int cmp(PortfolioPosition a, PortfolioPosition b) {
      switch (_sort) {
        case _SortKey.symbol:
          return a.holding.shortLabel.toLowerCase().compareTo(
            b.holding.shortLabel.toLowerCase(),
          );
        case _SortKey.value:
          return (a.holding.marketValue ?? 0).compareTo(
            b.holding.marketValue ?? 0,
          );
        case _SortKey.share:
          return a.share.compareTo(b.share);
        case _SortKey.returnPct:
          final ra = a.holding.totalReturnSincePurchase;
          final rb = b.holding.totalReturnSincePurchase;
          if (ra == null && rb == null) return 0;
          if (ra == null) return _descending ? -1 : 1;
          if (rb == null) return _descending ? 1 : -1;
          return ra.compareTo(rb);
      }
    }

    rows.sort((a, b) => _descending ? cmp(b, a) : cmp(a, b));
    return rows;
  }

  @override
  Widget build(BuildContext context) {
    final rows = _sorted;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
      decoration: BoxDecoration(
        color: AppTheme.surface(context),
        // 12, not the 20 every other card on this screen uses. A data table is
        // not a soft content block, and the tighter corner is what the audit
        // frame draws.
        borderRadius: BorderRadius.circular(12),
        boxShadow: [AppTheme.cardShadow(context)],
      ),
      child: LayoutBuilder(
        builder: (context, constraints) {
          // Fixed numeric columns, flexible SYMBOL. With today's four columns
          // the total lands exactly on the available width, so the scroll view
          // below has nothing to scroll — add a fifth column and it starts
          // scrolling instead of overflowing, with no restructuring here.
          const numeric = _kValueW + _kReturnW + _kShareW;
          final symbolW = math.max(112.0, constraints.maxWidth - numeric);
          return SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: SizedBox(
              width: symbolW + numeric,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _HoldingsHeaderRow(
                    symbolWidth: symbolW,
                    sort: _sort,
                    descending: _descending,
                    onTap: _onHeaderTap,
                  ),
                  Container(
                    height: 1,
                    color: AppTheme.isDark(context)
                        ? const Color(0xFF3A3A3A)
                        : const Color(0xFFDEDEDE),
                  ),
                  for (final position in rows)
                    _HoldingsDataRow(position: position, symbolWidth: symbolW),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}

// Column widths. The three numeric ones are fixed so the header and every row
// stay on the same grid; SYMBOL absorbs whatever is left.
const _kValueW = 80.0;
const _kReturnW = 70.0;
// Widest label of the four, and it has to hold "% PORTFOLIO" plus the sort
// caret without clipping — at 74 the caret fell off the right edge and the
// active column lost its direction indicator.
const _kShareW = 88.0;

class _HoldingsHeaderRow extends StatelessWidget {
  const _HoldingsHeaderRow({
    required this.symbolWidth,
    required this.sort,
    required this.descending,
    required this.onTap,
  });

  final double symbolWidth;
  final _SortKey sort;
  final bool descending;
  final ValueChanged<_SortKey> onTap;

  @override
  Widget build(BuildContext context) {
    Widget cell(String label, _SortKey key, double width) {
      final active = sort == key;
      final caret = descending ? '▾' : '▴';
      return SizedBox(
        width: width,
        child: InkWell(
          onTap: () => onTap(key),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Text(
              active ? '$label $caret' : label,
              maxLines: 1,
              overflow: TextOverflow.clip,
              style: GoogleFonts.inter(
                // Weight and colour carry the active column, never the lime
                // accent: #C9F158 measures about 1.5:1 on white, so as small
                // foreground text it would be the least readable thing in the
                // table. Lime stays a fill colour in this app.
                fontWeight: active ? FontWeight.w700 : FontWeight.w600,
                fontSize: 10,
                letterSpacing: 0.4,
                color: active ? _primaryText(context) : _muted(context),
              ),
            ),
          ),
        ),
      );
    }

    return Row(
      children: [
        cell('SYMBOL', _SortKey.symbol, symbolWidth),
        cell('VALUE', _SortKey.value, _kValueW),
        cell('RETURN', _SortKey.returnPct, _kReturnW),
        cell('% PORTFOLIO', _SortKey.share, _kShareW),
      ],
    );
  }
}

class _HoldingsDataRow extends StatelessWidget {
  const _HoldingsDataRow({required this.position, required this.symbolWidth});

  final PortfolioPosition position;
  final double symbolWidth;

  @override
  Widget build(BuildContext context) {
    final holding = position.holding;
    final qty = holding.quantity;
    final avg = holding.avgPurchasePrice;
    final subtitle = [
      if (qty != null) '${_formatQuantity(qty)} sh',
      if (avg != null) formatSgdExact(avg),
    ].join(' · ');

    // Already on the model, plausibility guard and all — not re-derived here.
    // It is (marketValue - costBasis) / costBasis, the same quantity as the
    // spec's (value - shares x avgPrice) / (shares x avgPrice), since
    // avgPurchasePrice is itself costBasis / quantity. Going through costBasis
    // directly avoids inheriting avgPurchasePrice's separate guard, which
    // blanks a different set of holdings.
    final ret = holding.totalReturnSincePurchase;

    Widget numeric(String text, TextStyle style, double width) => SizedBox(
      width: width,
      child: Text(text, maxLines: 1, style: style),
    );

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 9),
      child: Row(
        children: [
          SizedBox(
            width: symbolWidth,
            // Right gutter so an ellipsized symbol stops short of the VALUE
            // column instead of touching it — several real tickers are long
            // enough to run the full width of this cell.
            child: Padding(
              padding: const EdgeInsets.only(right: 10),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    holding.shortLabel,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.spaceGrotesk(
                      fontWeight: FontWeight.w700,
                      fontSize: 14,
                      height: 1.15,
                      color: _primaryText(context),
                    ),
                  ),
                  if (subtitle.isNotEmpty)
                    Text(
                      subtitle,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.inter(
                        fontWeight: FontWeight.w500,
                        fontSize: 10,
                        height: 1.3,
                        color: _muted(context),
                      ),
                    ),
                ],
              ),
            ),
          ),
          numeric(
            formatSgd(holding.marketValue ?? 0),
            GoogleFonts.spaceGrotesk(
              fontWeight: FontWeight.w700,
              fontSize: 12,
              color: _primaryText(context),
            ),
            _kValueW,
          ),
          numeric(
            // 'n/a', not '-', for the same reason as Total Return above.
            ret == null ? 'n/a' : _formatPct(ret, signed: true),
            GoogleFonts.spaceGrotesk(
              fontWeight: FontWeight.w700,
              fontSize: 12,
              color: ret == null
                  ? _muted(context)
                  : ret >= 0
                  ? _moneyGreen(context)
                  : _moneyRed(context),
            ),
            _kReturnW,
          ),
          numeric(
            _formatPct(position.share),
            GoogleFonts.inter(
              fontWeight: FontWeight.w500,
              fontSize: 12,
              color: _muted(context),
            ),
            _kShareW,
          ),
        ],
      ),
    );
  }
}

/// Empty state — replaces Sections 2–4 when nothing is connected yet.
class _NoInvestmentsCard extends StatelessWidget {
  const _NoInvestmentsCard();

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
          const Text('📈', style: TextStyle(fontSize: 26)),
          const SizedBox(height: 8),
          Text(
            'No investments connected yet',
            style: GoogleFonts.spaceGrotesk(
              fontWeight: FontWeight.w700,
              fontSize: 14,
              color: _primaryText(context),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Link your broker below to see portfolio value, risk alerts '
            'and holdings here.',
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

/// Section 5 — Connect to Broker / Exchange, Figma node 78:200. Identical
/// flow to Savings' Connect Bank; the dashed card doubles as the progress
/// indicator while link-token fetch, Plaid Link and the server-side
/// exchange + investments sync run.
class _ConnectBrokerCard extends StatelessWidget {
  const _ConnectBrokerCard({required this.connecting, required this.onTap});

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
                      'Connect to Broker / Exchange',
                      style: GoogleFonts.spaceGrotesk(
                        fontWeight: FontWeight.w700,
                        fontSize: 14,
                        color: _primaryText(context),
                      ),
                    ),
                    Text(
                      connecting
                          ? 'Connecting your broker. This can take a few '
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

