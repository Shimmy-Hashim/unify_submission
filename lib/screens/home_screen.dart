import 'dart:async';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../models/home_score_state.dart';
import '../models/insurance_policy.dart';
import '../models/investment_holding.dart';
import '../models/plaid_account.dart';
import '../models/score_history.dart';
import '../models/score_profile.dart';
import '../models/user_financial_profile.dart';
import '../services/app_preferences.dart';
import '../services/firebase_service.dart';
import '../services/health_score_service.dart';
import '../services/home_score_controller.dart';
import '../utils/app_styles.dart';
import '../utils/formatters.dart';
import '../widgets/accounts_refresh.dart';
import '../widgets/collapsible_card.dart';
import '../widgets/insight_card.dart';
import '../widgets/score_history_chart.dart';
import '../widgets/tier_segment_bar.dart';

/// Home — Figma 134:283 (Full Assessment) and 143:379 (Partial Assessment).
///
/// Reads the score through Slice 1's [HomeScoreController]: a live Firestore
/// stream of financialProfile/profile. Pull-to-refresh calls POST
/// /plaid/refresh (accounts, then a server-side rescore); the Re-run pill
/// calls POST /score/refresh alone. Nothing on this screen recomputes a
/// tier, a threshold or a reason code — every one of them is read back from
/// the cached pipeline result.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, this.onOpenInsurance, this.onOpenInvestments});

  /// Switches the app shell to the Insurance tab. Owned by MainShell, which
  /// holds the selected tab index; null leaves the capped-score message
  /// untappable.
  final VoidCallback? onOpenInsurance;

  /// Switches the app shell to the Investments tab, the same way — for
  /// Wealth Breakdown's "See all holdings" link. Null leaves it untappable.
  final VoidCallback? onOpenInvestments;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

/// Legible on both the white and the #1E1E1E card, so it stays a constant —
/// the same call Settings makes with its own `_kMuted`.
const _kMuted = AppTheme.muted;

typedef _Docs = List<QueryDocumentSnapshot<Map<String, dynamic>>>;

class _HomeScreenState extends State<HomeScreen> {
  HomeScoreController? _controller;

  // One listener per collection, owned here rather than inside Wealth
  // Breakdown, because the score card needs the same data: whether any bank or
  // investment account exists is what separates "Not connected" from "Can't
  // score yet". Firestore snapshot streams are single-subscription, so the
  // two consumers share the latest docs instead of each opening a stream.
  // Null until the first snapshot, and again after a listen error.
  final _subscriptions = <StreamSubscription<Object?>>[];
  _Docs? _bankDocs;
  _Docs? _investmentDocs;
  _Docs? _policyDocs;

  /// Recent daily snapshots, for the score card's trend line. Null until the
  /// first snapshot and after a listen error — either way the trend is simply
  /// not shown, since the score itself is still valid without it.
  List<ScoreHistoryPoint>? _history;

  @override
  void initState() {
    super.initState();
    final uid = FirebaseService.instance.currentUser?.uid;
    if (uid == null) return;
    _controller = HomeScoreController(uid: uid);
    final service = FirebaseService.instance;
    _listen(service.plaidAccounts(uid), (docs) => _bankDocs = docs);
    _listen(service.investmentAccounts(uid), (docs) => _investmentDocs = docs);
    _listen(service.insurancePolicies(uid), (docs) => _policyDocs = docs);
    _subscriptions.add(HealthScoreService.instance.watchHistory(uid).listen(
      (points) {
        if (mounted) setState(() => _history = points);
      },
      onError: (Object error) {
        debugPrint('home: scoreHistory listen failed ($error)');
        if (mounted) setState(() => _history = null);
      },
    ));
  }

  void _listen(
    CollectionReference<Map<String, dynamic>> collection,
    void Function(_Docs?) store,
  ) {
    _subscriptions.add(collection.snapshots().listen(
      (snap) {
        if (mounted) setState(() => store(snap.docs));
      },
      onError: (Object error) {
        debugPrint('home: ${collection.id} listen failed ($error)');
        if (mounted) setState(() => store(null));
      },
    ));
  }

  @override
  void dispose() {
    for (final sub in _subscriptions) {
      sub.cancel();
    }
    _controller?.dispose();
    super.dispose();
  }

  /// A failed manual refresh is a snackbar, never a replacement for the card —
  /// the data on screen is still valid, only the recompute request failed.
  void _showRefreshError(HomeScoreController controller, String message) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text(message)));
      controller.acknowledgeRefreshError();
    });
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    return Scaffold(
      backgroundColor: AppTheme.background(context),
      body: controller == null
          ? const Center(child: Text('Not signed in'))
          : SafeArea(
              child: ListenableBuilder(
                listenable: controller,
                builder: (context, _) {
                  final state = controller.state;
                  final error = state.refreshError;
                  if (error != null) _showRefreshError(controller, error);
                  // A pull refreshes the connected accounts (POST
                  // /plaid/refresh), which rescores server-side; the Re-run
                  // pill below stays score-only.
                  return AccountsRefreshIndicator(
                    onRefresh: controller.refreshAll,
                    child: ListView(
                      // Always scrollable so pull-to-refresh still works on
                      // the short states (Unavailable, Failed).
                      physics: const AlwaysScrollableScrollPhysics(),
                      padding: const EdgeInsets.fromLTRB(10, 12, 10, 24),
                      children: [
                        // Deliberately NOT const: MainShell rebuilds this
                        // screen through a ListenableBuilder when Hide
                        // Balances is toggled, and a const child in the same
                        // slot is skipped by the element tree (see main.dart's
                        // note on _screens). Every money figure on these two
                        // cards goes through formatSgd, so const here means
                        // the mask never applies.
                        _NetWorthCard(),
                        const SizedBox(height: 12),
                        HealthScoreCard(
                          state: state.score,
                          presence: AccountPresence(
                            hasBankAccounts: _bankDocs?.isNotEmpty,
                            hasInvestmentAccounts:
                                _investmentDocs?.isNotEmpty,
                          ),
                          history: _history,
                          refreshing: state.refreshing,
                          // Score-only recompute (POST /score/refresh), with
                          // its own in-flight guard and error snackbar.
                          onRerun: controller.refresh,
                          onOpenInsurance: widget.onOpenInsurance,
                        ),
                        const SizedBox(height: 18),
                        const _SectionLabel('INSIGHTS'),
                        const SizedBox(height: 6),
                        _InsightsSection(state: state.score),
                        const SizedBox(height: 18),
                        // Figma 231:485: between Insights and Wealth
                        // Breakdown, as a card with no section label.
                        ScoreHistoryCard(
                          key: const ValueKey('score-history'),
                          state: state.score,
                          history: _history,
                        ),
                        const SizedBox(height: 18),
                        const _SectionLabel('WEALTH BREAKDOWN'),
                        const SizedBox(height: 8),
                        WealthBreakdownCard(
                          bankDocs: _bankDocs,
                          investmentDocs: _investmentDocs,
                          policyDocs: _policyDocs,
                          score: state.score,
                          onOpenInvestments: widget.onOpenInvestments,
                        ),
                      ],
                    ),
                  );
                },
              ),
            ),
    );
  }
}

/// Same style as the Savings/Investments section headings.
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
          color: _kMuted,
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// 1. Net Worth / Total Assets — Figma 231:597 (Net Worth tab) and
//    384:722 (Total Assets tab selected)
// ---------------------------------------------------------------------------

/// Which figure the card shows. Net Worth is the default: it is the honest
/// headline, since total assets alone flatters anyone carrying debt.
enum WealthFigure { netWorth, totalAssets }

/// Streams financialProfile/profile and owns the selected tab. The figures are
/// all backend-owned — net_worth_sgd, total_assets_sgd and total_debts_sgd are
/// written together by `compute_balance_totals()`, so assets − debts == net
/// worth by construction and nothing here adds or subtracts anything.
///
/// The stream is created once, in initState, not in build: the tab tap
/// rebuilds this widget, and a fresh `snapshots()` stream per build would
/// reset the StreamBuilder and flash "—" on every tap.
class _NetWorthCard extends StatefulWidget {
  const _NetWorthCard();

  @override
  State<_NetWorthCard> createState() => _NetWorthCardState();
}

class _NetWorthCardState extends State<_NetWorthCard> {
  late final Stream<DocumentSnapshot<Map<String, dynamic>>> _profile;
  WealthFigure _selected = WealthFigure.netWorth;

  @override
  void initState() {
    super.initState();
    final uid = FirebaseService.instance.currentUser?.uid;
    _profile = uid == null
        ? const Stream.empty()
        : FirebaseService.instance.financialProfileDoc(uid).snapshots();
  }

  /// [UserFinancialProfile.fromMap] casts every field it knows about, so a
  /// malformed value in one this card never reads — a corrupted `ml_profile`,
  /// say — would otherwise throw during build and take the whole screen down
  /// with it. The score card already surfaces that corruption properly as
  /// ScoreFailed; the figures just degrade to a dash.
  static UserFinancialProfile? _readProfile(
      DocumentSnapshot<Map<String, dynamic>>? doc) {
    if (doc == null || !doc.exists) return null;
    try {
      return UserFinancialProfile.fromDoc(doc);
    } catch (error) {
      debugPrint('net worth: unreadable financialProfile ($error)');
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<DocumentSnapshot<Map<String, dynamic>>>(
      stream: _profile,
      builder: (context, snap) => NetWorthCardBody(
        profile: _readProfile(snap.data),
        selected: _selected,
        onSelect: (figure) => setState(() => _selected = figure),
      ),
    );
  }
}

/// The card itself, public and Firebase-free so the tab behaviour can be
/// widget-tested directly.
///
/// NO "this month" CHANGE LINE on either tab, deliberately — unchanged from
/// before the tabs. Both Figma frames carry the same static "↑ +$1,200 this
/// month" placeholder. A real Net Worth trend would now be derivable from
/// scoreHistory.net_worth_sgd, but that is a new feature with its own
/// decisions (comparison window, gap days), not part of surfacing these two
/// fields; Total Assets has no stored history at all, so a trend there could
/// only be invented. Each tab's slot says instead what the figure is.
class NetWorthCardBody extends StatelessWidget {
  const NetWorthCardBody({
    super.key,
    required this.profile,
    required this.selected,
    required this.onSelect,
  });

  /// Null while loading, when no document exists, or when it is unreadable.
  final UserFinancialProfile? profile;
  final WealthFigure selected;
  final ValueChanged<WealthFigure> onSelect;

  /// The displayed figure. Net worth keeps its existing `?? 0` for a profile
  /// that exists (onboarding seeds it at 0.0). Total assets does NOT default:
  /// the client never seeds it, so its absence means the backend has not
  /// refreshed this profile since the field was added — unknown, shown as a
  /// dash rather than a false $0.
  static String figureText(UserFinancialProfile? profile, WealthFigure figure) {
    if (profile == null) return '-';
    return switch (figure) {
      WealthFigure.netWorth => formatSgd(profile.netWorthSgd ?? 0),
      WealthFigure.totalAssets => profile.totalAssetsSgd == null
          ? '-'
          : formatSgd(profile.totalAssetsSgd!),
    };
  }

  /// What the figure is. Total Assets names the debts it excludes, which is
  /// what ties the two tabs together: assets − debts = net worth.
  static String captionText(
      UserFinancialProfile? profile, WealthFigure figure) {
    return switch (figure) {
      WealthFigure.netWorth => 'Savings + investments − debts',
      WealthFigure.totalAssets => profile?.totalDebtsSgd == null
          ? 'Savings + investments, before debts'
          : 'Savings + investments, before '
              '${formatSgd(profile!.totalDebtsSgd!)} in debts',
    };
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(17, 10, 14, 14),
      decoration: BoxDecoration(
        color: AppTheme.surface(context),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [AppTheme.cardShadow(context)],
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Figma: the tabs replace the old "Total Net Worth" title.
                Row(
                  children: [
                    _FigureTab(
                      label: 'Net Worth',
                      selected: selected == WealthFigure.netWorth,
                      onTap: () => onSelect(WealthFigure.netWorth),
                    ),
                    const SizedBox(width: 4),
                    _FigureTab(
                      label: 'Total Assets',
                      selected: selected == WealthFigure.totalAssets,
                      onTap: () => onSelect(WealthFigure.totalAssets),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                FittedBox(
                  fit: BoxFit.scaleDown,
                  alignment: Alignment.centerLeft,
                  child: Text(
                    figureText(profile, selected),
                    style: GoogleFonts.inter(
                      fontWeight: FontWeight.w700,
                      fontSize: 40,
                      height: 1.1,
                      color: AppTheme.primaryText(context),
                    ),
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  captionText(profile, selected),
                  style: GoogleFonts.inter(
                    fontWeight: FontWeight.w600,
                    fontSize: 12,
                    color: _kMuted,
                  ),
                ),
              ],
            ),
          ),
          // The design's eye icon, wired to the existing Hide Balances
          // preference — every figure on the card, on either tab, goes
          // through formatSgd, so one toggle masks all of them.
          IconButton(
            iconSize: 24,
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(),
            visualDensity: VisualDensity.compact,
            tooltip: AppPreferences.instance.hideBalances
                ? 'Show balances'
                : 'Hide balances',
            onPressed: () => AppPreferences.instance
                .setHideBalances(!AppPreferences.instance.hideBalances),
            icon: Icon(
              AppPreferences.instance.hideBalances
                  ? Icons.visibility_off_outlined
                  : Icons.remove_red_eye,
              color: AppTheme.primaryText(context),
            ),
          ),
        ],
      ),
    );
  }
}

/// Figma 384:718-729: a 24px pill, Inter SemiBold 11. Lime marks the one the
/// user has chosen — the selection convention Savings' month pills and
/// Settings' toggles already follow — with the unselected pill on the same
/// [AppTheme.iconTile] fill (#F2F3F5 light) and muted ink those pills use.
class _FigureTab extends StatelessWidget {
  const _FigureTab({
    required this.label,
    required this.selected,
    required this.onTap,
  });

  final String label;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      selected: selected,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(20),
        child: Container(
          height: 24,
          padding: const EdgeInsets.symmetric(horizontal: 11),
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: selected ? AppColors.limeAccent : AppTheme.iconTile(context),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Text(
            label,
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w600,
              fontSize: 11,
              // Lime is fixed in both modes, so its ink is fixed with it.
              color: selected
                  ? AppColors.brandBlack
                  : AppTheme.tertiaryText(context),
            ),
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// 2. Financial Health Score — Figma 134:291 / 143:379
// ---------------------------------------------------------------------------

/// Display metadata for one domain. The two tier labels are the artifact's own
/// per-domain labels (ml/layer3.py's locked constants) — they have to be
/// static rather than read off the result, because an unassessed domain still
/// shows both of them under a neutral bar.
class _DomainMeta {
  const _DomainMeta(this.emoji, this.name, this.lowLabel, this.highLabel);

  final String emoji;
  final String name;
  final String lowLabel;
  final String highLabel;
}

// Spelled exactly as the artifacts' tier_labels ("On Track", "Getting
// Started", "Not Covered"), so the sub-labels match the tier value the cache
// shows above them.
const _domainMeta = {
  ScoreDomain.banking: _DomainMeta('💧', 'Liquidity', 'Building', 'Stable'),
  ScoreDomain.investment:
      _DomainMeta('📈', 'Growth', 'Getting Started', 'On Track'),
  ScoreDomain.insurance:
      _DomainMeta('🛡️', 'Protection', 'Not Covered', 'Covered'),
};

/// The capped-score explanation, built from the cached breakdown — or null
/// when the score is not capped, in which case the card shows nothing in its
/// place (Figma has no uncapped copy, and "nothing is holding this back" needs
/// no sentence).
///
/// WHY "INSURANCE" IS NOT A GUESS
/// ml/overall_score.py derives the ceiling from exactly one input:
/// CEILING_BY_INDICATORS_CLEARED[insurance_indicators_cleared] — the three
/// protection indicators (insurance tier, income protection, hospitalisation).
/// `capped` is `option_b_score > ceiling`. So whenever `capped` is true, the
/// cap is by construction a protection cap; there is no other kind.
///
/// The two detail lines use only cached numbers: how many of the three
/// indicators are met, and `option_b_score`, the score before the ceiling was
/// applied.
///
/// TWO LINES, NOT ONE WRAPPED STRING. The summary column is 126px wide; the
/// old single line ("0 of 3 protection checks met · uncapped 67", 198px in
/// Inter 500 at 9.5) wrapped wherever the layout happened to break it. Each
/// line is now its own Text, measured against the real font at its widest
/// ("0 of 3 checks met", 81.5px).
({String headline, String checks, String uncapped})? cappedScoreMessage(
    ScoreBreakdown breakdown) {
  if (!breakdown.capped) return null;
  return (
    headline: 'Score capped by your insurance gap',
    checks: '${breakdown.insuranceIndicatorsCleared} of 3 checks met',
    uncapped: 'Uncapped ${breakdown.optionBScore.round()}',
  );
}

/// ("↑ +4 pts", "since 2 Sep") / ("↓ -3 pts", "since 2 Sep") /
/// ("No change", "since 2 Sep").
///
/// "since `date`" rather than the mock's "this month": the comparison point is
/// the last day that actually has a score, which may be yesterday or several
/// weeks back, and labelling either as "this month" would misstate it. The
/// year is added only when the comparison crosses into a different one.
///
/// Returned as two lines for the same reason as [cappedScoreMessage]: the
/// widest possible single line ("↓ -100 pts since 30 May 2025") is 162px in
/// Inter 600 at 11 against a 126px column. Split, the widest halves are
/// "↑ +100 pts" (60px) and "since 30 May 2025" (102px), so no reachable
/// delta or date wraps at the default text scale.
({String change, String since}) scoreTrendLabel(ScoreTrend trend,
    {required DateTime now}) {
  final since = trend.since;
  final date = since.year == now.year
      ? '${since.day} ${_monthShort(since.month)}'
      : formatShortDate(since);
  final delta = trend.deltaPoints;
  final unit = delta.abs() == 1 ? 'pt' : 'pts';
  return (
    change: delta == 0
        ? 'No change'
        : delta > 0
            ? '↑ +$delta $unit'
            : '↓ -${delta.abs()} $unit',
    since: 'since $date',
  );
}

String _monthShort(int month) => const [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    ][month - 1];

/// "Last calculated today" / "yesterday" / "on 2 Sep 2026", in the device's
/// local calendar — this is the user's own clock, unlike the UTC snapshot IDs.
String lastCalculatedLabel(DateTime computedAt, {required DateTime now}) {
  final local = computedAt.toLocal();
  final day = DateTime(local.year, local.month, local.day);
  final today = DateTime(now.year, now.month, now.day);
  final days = today.difference(day).inDays;
  if (days <= 0) return 'Last calculated today';
  if (days == 1) return 'Last calculated yesterday';
  return 'Last calculated on ${formatShortDate(local)}';
}

/// Public so its copy can be exercised in widget tests without Firebase — it
/// reads nothing but its arguments.
class HealthScoreCard extends StatelessWidget {
  const HealthScoreCard({
    super.key,
    required this.state,
    required this.presence,
    this.history,
    this.refreshing = false,
    this.onRerun,
    this.onOpenInsurance,
    this.now,
  });

  final ScoreState state;
  final AccountPresence presence;

  /// Recent scoreHistory snapshots for the trend line; null while loading.
  final List<ScoreHistoryPoint>? history;

  /// A refresh is in flight — the Re-run action shows progress and disables.
  final bool refreshing;

  /// HomeScoreController.refresh, the pull-to-refresh trigger. Null hides the
  /// Re-run action.
  final Future<void> Function()? onRerun;

  /// Opens the Insurance tab from the capped-score message.
  final VoidCallback? onOpenInsurance;

  /// Clock override for tests; defaults to DateTime.now().
  final DateTime? now;

  /// When the cached run on screen was computed, if there is one.
  DateTime? get _computedAt => switch (state) {
        ScoreComplete(:final profile) => profile.computedAt,
        ScorePartial(:final profile) => profile.computedAt,
        ScoreUnavailable(:final profile) => profile?.computedAt,
        ScoreLoading() || ScoreNeverComputed() || ScoreFailed() => null,
      };

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(12, 12, 14, 14),
      decoration: BoxDecoration(
        color: AppTheme.surface(context),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [AppTheme.cardShadow(context)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _header(context),
          const SizedBox(height: 6),
          _body(context),
        ],
      ),
    );
  }

  Widget _header(BuildContext context) {
    final computedAt = _computedAt;
    final rerun = onRerun;
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(left: 4, top: 4),
          child: Row(
            children: [
              Text(
                'Financial Health Score',
                style: GoogleFonts.spaceGrotesk(
                  fontWeight: FontWeight.w700,
                  fontSize: 13,
                  color: AppTheme.tertiaryText(context),
                ),
              ),
              const SizedBox(width: 6),
              _InfoButton(onTap: () => _showExplainer(context)),
            ],
          ),
        ),
        const Spacer(),
        if (rerun != null || computedAt != null)
          Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              if (rerun != null)
                _RerunButton(refreshing: refreshing, onPressed: rerun),
              if (computedAt != null) ...[
                const SizedBox(height: 4),
                Text(
                  lastCalculatedLabel(computedAt,
                      now: now ?? DateTime.now()),
                  style: GoogleFonts.inter(
                    fontWeight: FontWeight.w500,
                    fontSize: 9,
                    color: AppTheme.tertiaryText(context),
                  ),
                ),
              ],
            ],
          ),
      ],
    );
  }

  /// The card body per state. Every branch of the sealed hierarchy is handled
  /// here so a state added later fails to compile rather than rendering blank.
  Widget _body(BuildContext context) {
    return switch (state) {
      ScoreLoading() => const _ScoreCardLoading(),
      ScoreFailed(:final message) => _ScoreCardFailed(message: message),
      ScoreUnavailable(:final reason) => _ScoreCardUnavailable(reason: reason),
      // Never computed carries no payload at all (Slice 1: `const
      // ScoreNeverComputed()`), so there is no tier to show — all three rows
      // render unassessed, and account presence decides whether each one is
      // "Not connected" or merely not calculated yet.
      ScoreNeverComputed() => _ScoreCardRows(
          summary: const _ScoreSummary(
            score: null,
            caption: 'Not calculated yet',
          ),
          presence: presence,
          profile: null,
          footnote: 'Connect a bank, add investments or upload a policy, '
              'then tap Re-run Score.',
        ),
      ScorePartial(:final profile) => _ScoreCardRows(
          summary: _ScoreSummary(
            score: null,
            caption: '${profile.assessedDomains.length} of 3 areas assessed',
          ),
          presence: presence,
          profile: profile,
        ),
      ScoreComplete(:final profile, :final overallScore, :final breakdown) =>
        _ScoreCardRows(
          summary: _ScoreSummary(
            score: overallScore,
            // No trend until history has loaded — and none at all when there
            // is no earlier real score to compare with (computeScoreTrend).
            trend: history == null
                ? null
                : computeScoreTrend(
                    currentScore: overallScore,
                    currentComputedAt: profile.computedAt,
                    history: history!,
                  ),
            now: now,
            capped: cappedScoreMessage(breakdown),
            onOpenInsurance: onOpenInsurance,
          ),
          presence: presence,
          profile: profile,
        ),
    };
  }

  void _showExplainer(BuildContext context) {
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Financial Health Score',
            style: GoogleFonts.spaceGrotesk(fontWeight: FontWeight.w700)),
        content: Text(
          'Your score combines three areas: Liquidity (savings, emergency '
          'fund, debt), Growth (investing) and Protection (insurance).\n\n'
          'Protection can raise your score and can also cap it: strong '
          'banking alone can never make up for a coverage gap.',
          style: GoogleFonts.inter(fontSize: 13, height: 1.5),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Got it'),
          ),
        ],
      ),
    );
  }
}

class _InfoButton extends StatelessWidget {
  const _InfoButton({required this.onTap});

  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      customBorder: const CircleBorder(),
      child: Container(
        width: 15,
        height: 15,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(color: AppTheme.tertiaryText(context)),
        ),
        child: Text(
          'i',
          style: GoogleFonts.spaceGrotesk(
            fontWeight: FontWeight.w700,
            fontSize: 10,
            height: 1.0,
            color: AppTheme.tertiaryText(context),
          ),
        ),
      ),
    );
  }
}

/// Figma 270:653 — the lime pill. Lime is the app's action accent
/// ([AppColors.limeAccent], "reserved for genuine actions"), with
/// [AppColors.brandBlack] ink as that token requires.
class _RerunButton extends StatelessWidget {
  const _RerunButton({required this.refreshing, required this.onPressed});

  final bool refreshing;
  final Future<void> Function() onPressed;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: AppColors.limeAccent,
      borderRadius: BorderRadius.circular(20),
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        // Disabled while in flight. HomeScoreController.refresh is already a
        // no-op then; disabling says so visibly.
        onTap: refreshing ? null : onPressed,
        child: SizedBox(
          height: 28,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (refreshing) ...[
                  const SizedBox(
                    width: 10,
                    height: 10,
                    child: CircularProgressIndicator(
                      strokeWidth: 1.6,
                      color: AppColors.brandBlack,
                    ),
                  ),
                  const SizedBox(width: 6),
                ],
                Text(
                  refreshing ? 'Re-running…' : 'Re-run Score',
                  style: GoogleFonts.inter(
                    fontWeight: FontWeight.w700,
                    fontSize: 11,
                    color: AppColors.brandBlack,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/// The score-plus-three-rows layout shared by Complete, Partial and
/// NeverComputed. One widget, because the three differ only in which rows have
/// a tier and what the summary column says — building them separately is how
/// the "assessed" and "not assessed" treatments drift apart.
class _ScoreCardRows extends StatelessWidget {
  const _ScoreCardRows({
    required this.summary,
    required this.presence,
    required this.profile,
    this.footnote,
  });

  final Widget summary;
  final AccountPresence presence;

  /// Null for NeverComputed — every row renders unassessed.
  final ScoreProfile? profile;
  final String? footnote;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 8),
        Row(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            SizedBox(width: _kSummaryWidth, child: summary),
            const SizedBox(width: 8),
            Expanded(
              child: Column(
                children: [
                  for (final domain in ScoreDomain.values) ...[
                    _TierRow(
                      domain: domain,
                      result: profile?.domains[domain],
                      presence: presence,
                    ),
                    if (domain != ScoreDomain.values.last)
                      const SizedBox(height: 12),
                  ],
                ],
              ),
            ),
          ],
        ),
        if (footnote != null) ...[
          const SizedBox(height: 12),
          Text(
            footnote!,
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w500,
              fontSize: 11,
              height: 1.4,
              color: AppTheme.secondaryText(context),
            ),
          ),
        ],
      ],
    );
  }
}

/// The left column's width — the slot the gauge used to occupy.
const _kSummaryWidth = 126.0;

/// Figma 266:653 / 266:655 / 231:490: the score numeral in its band colour,
/// the trend beneath it, then the capped explanation. No band word ("Stable",
/// "Good") — the colour carries the band.
///
/// With no score (Partial, NeverComputed) the numeral is a muted em dash and
/// [caption] says why; there is no trend or cap to show without a score.
class _ScoreSummary extends StatelessWidget {
  const _ScoreSummary({
    required this.score,
    this.caption,
    this.trend,
    this.now,
    this.capped,
    this.onOpenInsurance,
  });

  final double? score;
  final String? caption;
  final ScoreTrend? trend;
  final DateTime? now;
  final ({String headline, String checks, String uncapped})? capped;
  final VoidCallback? onOpenInsurance;

  @override
  Widget build(BuildContext context) {
    final value = score;
    final brightness = Theme.of(context).brightness;
    final trend = this.trend;
    final capped = this.capped;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        FittedBox(
          fit: BoxFit.scaleDown,
          child: Text(
            value == null ? '-' : value.round().toString(),
            style: GoogleFonts.spaceGrotesk(
              fontWeight: FontWeight.w700,
              fontSize: 40,
              height: 1.2,
              color: value == null
                  ? AppTheme.tertiaryText(context)
                  : TierSegmentBar.scoreColor(value),
            ),
          ),
        ),
        if (caption != null)
          Text(
            caption!,
            textAlign: TextAlign.center,
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w500,
              fontSize: 11,
              height: 1.35,
              color: AppTheme.tertiaryText(context),
            ),
          ),
        if (trend != null) ..._trendLines(context, trend, brightness),
        if (capped != null) ...[
          const SizedBox(height: 6),
          _CappedMessage(message: capped, onTap: onOpenInsurance),
        ],
      ],
    );
  }

  /// The change, then the comparison date, as two separate single-line Texts
  /// — see [scoreTrendLabel] for the widths this was measured against.
  List<Widget> _trendLines(
      BuildContext context, ScoreTrend trend, Brightness brightness) {
    final label = scoreTrendLabel(trend, now: now ?? DateTime.now());
    final style = GoogleFonts.inter(
      fontWeight: FontWeight.w600,
      fontSize: 11,
      height: 1.35,
      color: trend.deltaPoints == 0
          ? AppTheme.tertiaryText(context)
          : TierSegmentBar.changeColor(
              up: trend.deltaPoints > 0, brightness: brightness),
    );
    return [
      Text(label.change, textAlign: TextAlign.center, style: style),
      Text(label.since, textAlign: TextAlign.center, style: style),
    ];
  }
}

/// The tappable capped explanation. Red per Figma 231:490 (#E53935, resolved
/// through [AppTheme.alertRed] like the Insurance card's gap red), with a
/// chevron as the affordance that it leads somewhere.
class _CappedMessage extends StatelessWidget {
  const _CappedMessage({required this.message, required this.onTap});

  final ({String headline, String checks, String uncapped}) message;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final red = AppTheme.alertRed(context, const Color(0xFFE53935));
    return Semantics(
      button: onTap != null,
      hint: onTap == null ? null : 'Opens Insurance',
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(8),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 2, vertical: 3),
          child: Column(
            children: [
              Text.rich(
                TextSpan(
                  text: message.headline,
                  children: [
                    if (onTap != null)
                      const WidgetSpan(
                        alignment: PlaceholderAlignment.middle,
                        child: Icon(Icons.chevron_right, size: 13),
                      ),
                  ],
                ),
                textAlign: TextAlign.center,
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w600,
                  fontSize: 11,
                  height: 1.35,
                  color: red,
                ),
              ),
              const SizedBox(height: 2),
              // Two Texts, one per fact — see [cappedScoreMessage].
              for (final line in [message.checks, message.uncapped])
                Text(
                  line,
                  textAlign: TextAlign.center,
                  style: GoogleFonts.inter(
                    fontWeight: FontWeight.w500,
                    fontSize: 9.5,
                    height: 1.3,
                    color: AppTheme.tertiaryText(context),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

/// One domain row: icon + name, tier value, the two-segment bar and its two
/// sub-labels. An unassessed domain dims the whole row — the Partial
/// Assessment frame's Growth treatment, applied to whichever domains are
/// actually unassessed — and says why, per [domainRowStatus]: "Not connected"
/// only when no account of that type exists, "Can't score yet" when data
/// exists but no tier came out of it.
class _TierRow extends StatelessWidget {
  const _TierRow({
    required this.domain,
    required this.result,
    required this.presence,
  });

  final ScoreDomain domain;
  final DomainScore? result;
  final AccountPresence presence;

  @override
  Widget build(BuildContext context) {
    final meta = _domainMeta[domain]!;
    final status = domainRowStatus(domain, result, presence);
    final assessed = status == DomainRowStatus.assessed;
    final segment = !assessed
        ? TierSegment.notAssessed
        : result!.tierIndex == 0
            ? TierSegment.healthy
            : TierSegment.weak;

    final brightness = Theme.of(context).brightness;
    // The domain name is body ink, not a tier signal — it reads the same
    // whether the tier is healthy or weak. Sourced from the bar's table
    // anyway so the row's two inks stay in step across themes.
    final nameColor = assessed
        ? TierSegmentBar.valueColor(TierSegment.healthy, brightness)
        : TierSegmentBar.valueColor(TierSegment.notAssessed, brightness);

    final row = Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                '${meta.emoji} ${meta.name}',
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w600,
                  fontSize: 13,
                  color: nameColor,
                ),
              ),
            ),
            Text(
              switch (status) {
                DomainRowStatus.assessed => result!.tierLabel ?? '-',
                DomainRowStatus.notConnected => 'Not connected',
                DomainRowStatus.cannotCalculateYet => 'Can\'t score yet',
                // Presence still loading — claim neither.
                DomainRowStatus.unknown => '-',
              },
              style: GoogleFonts.inter(
                fontWeight: FontWeight.w600,
                fontSize: 13,
                color: TierSegmentBar.valueColor(segment, brightness),
              ),
            ),
          ],
        ),
        const SizedBox(height: 1),
        TierSegmentBar(
          segment: segment,
          lowLabel: meta.lowLabel,
          highLabel: meta.highLabel,
        ),
      ],
    );

    return assessed ? row : Opacity(opacity: 0.5, child: row);
  }
}

/// Loading — the app's established treatment (a centred
/// CircularProgressIndicator) held at the card's real height so the screen
/// does not jump when the first snapshot lands.
class _ScoreCardLoading extends StatelessWidget {
  const _ScoreCardLoading();

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 190,
      child: Center(
        child: SizedBox(
          width: 22,
          height: 22,
          child: CircularProgressIndicator(
              strokeWidth: 2, color: AppTheme.primaryText(context)),
        ),
      ),
    );
  }
}

/// Error — deliberately different in tone from every "not ready yet" state.
/// Something broke; the data is not merely missing.
class _ScoreCardFailed extends StatelessWidget {
  const _ScoreCardFailed({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.error_outline, color: AppColors.danger, size: 22),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Something went wrong',
                  style: GoogleFonts.spaceGrotesk(
                    fontWeight: FontWeight.w700,
                    fontSize: 14,
                    color: AppColors.danger,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  message,
                  style: GoogleFonts.inter(
                    fontWeight: FontWeight.w500,
                    fontSize: 11.5,
                    height: 1.45,
                    color: AppTheme.secondaryText(context),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// The four whole-pipeline unavailable reasons, each with its own message and
/// its own next step. They are genuinely different situations — one needs a
/// number the app never asks for twice, one needs onboarding finished, one the
/// user can do nothing about — so they are never collapsed into a shared
/// "score unavailable".
class _ScoreCardUnavailable extends StatelessWidget {
  const _ScoreCardUnavailable({required this.reason});

  final ScoreUnavailableReason reason;

  ({String title, String body, String? action}) get _copy => switch (reason) {
        ScoreUnavailableReason.incomeNotSet => (
            title: 'Your income isn\'t set',
            body: 'Every part of the score is a ratio against your monthly '
                'income (savings rate, coverage multiples, debt load), so '
                'none of them can be worked out without it.',
            // Income is written once at onboarding and has no edit screen
            // today, so there is no honest place to send the user.
            action: null,
          ),
        ScoreUnavailableReason.dateOfBirthNotSet => (
            title: 'Your date of birth is missing',
            body: 'Your score is measured against benchmarks for your age '
                'band, so we need your date of birth before we can work it '
                'out.',
            action: 'Add it in Settings → Edit Profile.',
          ),
        ScoreUnavailableReason.unsupportedAge => (
            title: 'Outside the scoring range',
            body: 'Unify\'s models are built and validated for ages 20 to 70. '
                'Yours falls outside that range, so we\'d rather show you '
                'nothing than a score the models were never fitted for.',
            action: 'Everything else in the app still works as normal.',
          ),
        ScoreUnavailableReason.noFinancialProfile => (
            title: 'Finish setting up your profile',
            body: 'Your financial profile hasn\'t been created yet, so there '
                'is nothing to score against.',
            action: 'Complete onboarding to get started.',
          ),
        ScoreUnavailableReason.unknown => (
            title: 'Score unavailable',
            body: 'Your score can\'t be worked out right now, for a reason '
                'this version of the app doesn\'t recognise.',
            action: 'Updating Unify may resolve it.',
          ),
      };

  @override
  Widget build(BuildContext context) {
    final copy = _copy;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // The same summary column as the scored states, with no number —
          // grey rather than any band colour, since nothing is on its way.
          const SizedBox(
            width: 110,
            child: _ScoreSummary(score: null, caption: 'Score unavailable'),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  copy.title,
                  style: GoogleFonts.spaceGrotesk(
                    fontWeight: FontWeight.w700,
                    fontSize: 14,
                    color: AppTheme.primaryText(context),
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  copy.body,
                  style: GoogleFonts.inter(
                    fontWeight: FontWeight.w500,
                    fontSize: 11.5,
                    height: 1.45,
                    color: AppTheme.secondaryText(context),
                  ),
                ),
                if (copy.action != null) ...[
                  const SizedBox(height: 6),
                  Text(
                    copy.action!,
                    style: GoogleFonts.inter(
                      fontWeight: FontWeight.w600,
                      fontSize: 11.5,
                      height: 1.4,
                      color: AppTheme.primaryText(context),
                    ),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// 3. Insights — Layer 3 cross-domain flags, Figma 134:397
// ---------------------------------------------------------------------------

class _InsightsSection extends StatelessWidget {
  const _InsightsSection({required this.state});

  final ScoreState state;

  @override
  Widget build(BuildContext context) {
    // Layer 3 only runs when all three tiers exist, so ScoreComplete is the
    // only state that can carry flags. Everything else is "not evaluated" —
    // which is a different message from "evaluated, nothing fired".
    final complete = state is ScoreComplete ? state as ScoreComplete : null;
    final flags = complete?.layer3Flags ?? const <String>[];
    if (flags.isEmpty) {
      return _InsightsEmpty(
        emptyInsightsMessage(allAssessed: complete != null),
      );
    }
    final insights = [
      for (final flag in flags)
        if (insightForFlag(flag) != null) insightForFlag(flag)!,
    ];
    return SizedBox(
      height: 116,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 8),
        itemCount: insights.length,
        separatorBuilder: (_, _) => const SizedBox(width: 8),
        itemBuilder: (context, i) => InsightCard(insight: insights[i]),
      ),
    );
  }
}

class _InsightsEmpty extends StatelessWidget {
  const _InsightsEmpty(this.message);

  final String message;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
      child: Text(
        message,
        style: GoogleFonts.inter(
          fontWeight: FontWeight.w500,
          fontSize: 12,
          height: 1.45,
          color: AppTheme.secondaryText(context),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// 3b. Score History — Figma 322:653 (collapsed row)
// ---------------------------------------------------------------------------

/// The collapsed row's one-line summary.
///
/// The trend is the score card's own — [computeScoreTrend] against the live
/// score, worded by [scoreTrendLabel] — so the row and the card can never
/// disagree. Only its first line is used; the row has room for the change,
/// not the date. Without a trend (no complete score, or no earlier scored
/// day) the row says how much history exists instead of inventing a change.
({String text, ScoreTrend? trend}) scoreHistorySummary({
  required ScoreState state,
  required List<ScoreHistoryPoint> history,
  required DateTime now,
}) {
  if (state case ScoreComplete(:final overallScore, :final profile)) {
    final trend = computeScoreTrend(
      currentScore: overallScore,
      currentComputedAt: profile.computedAt,
      history: history,
    );
    if (trend != null) {
      return (text: scoreTrendLabel(trend, now: now).change, trend: trend);
    }
  }
  final days = chronological(history).length;
  return (
    text: days == 0
        ? 'No history yet'
        : '$days day${days == 1 ? '' : 's'} recorded',
    trend: null,
  );
}

/// "Capped by your insurance gap since 2 Sep" — or "since at least 2 Sep" when
/// the streak reaches the oldest snapshot AND the window may be hiding older
/// ones (it is full). When the window holds fewer snapshots than its limit it
/// is the user's whole history, so the earliest capped day is the real start.
String cappedSinceLabel(
  CappedRun run, {
  required bool windowFull,
  required DateTime now,
}) {
  final since = run.since;
  final date = since.year == now.year
      ? '${since.day} ${_monthShort(since.month)}'
      : formatShortDate(since);
  final atLeast = run.startsAtEarliestLoaded && windowFull ? 'at least ' : '';
  return 'Capped by your insurance gap since $atLeast$date';
}

/// Collapsed by default, on the same [CollapsibleCard] as Wealth Breakdown's
/// rows, so the two read as one family and cannot drift apart.
///
/// THE WINDOW is everything [HealthScoreService.watchHistory] loads: up to
/// [HealthScoreService.historyLimit] (90) daily snapshots, GET /score/history's
/// own default. Most real accounts hold 1-3 days today, so a 30-day cut would
/// change nothing now while hiding history later for no gain; and charting
/// exactly what was loaded means the chart, the capped streak and the domain
/// arrows all describe the same days.
class ScoreHistoryCard extends StatelessWidget {
  const ScoreHistoryCard({
    super.key,
    required this.state,
    required this.history,
    this.now,
  });

  final ScoreState state;

  /// Null while loading, or after a listen error — the row stays untappable,
  /// as a Wealth Breakdown row does before its data lands.
  final List<ScoreHistoryPoint>? history;

  /// Clock override for tests.
  final DateTime? now;

  @override
  Widget build(BuildContext context) {
    final history = this.history;
    final now = this.now ?? DateTime.now();
    final summary = history == null
        ? null
        : scoreHistorySummary(state: state, history: history, now: now);
    final trend = summary?.trend;
    return CollapsibleCard(
      title: 'Score History',
      titleSize: 14,
      trailing: Text(
        summary?.text ?? '-',
        style: GoogleFonts.spaceGrotesk(
          fontWeight: FontWeight.w700,
          fontSize: 13,
          color: trend == null || trend.deltaPoints == 0
              ? AppTheme.tertiaryText(context)
              : TierSegmentBar.changeColor(
                  up: trend.deltaPoints > 0,
                  brightness: Theme.of(context).brightness),
        ),
      ),
      // summary is non-null exactly when history is.
      enabled: history != null,
      body: history == null
          ? const SizedBox(width: double.infinity)
          : _ScoreHistoryBody(history: history, now: now),
    );
  }
}

class _ScoreHistoryBody extends StatelessWidget {
  const _ScoreHistoryBody({required this.history, required this.now});

  final List<ScoreHistoryPoint> history;
  final DateTime now;

  @override
  Widget build(BuildContext context) {
    final points = chronological(history);
    final scoredDays = points.where((p) => p.overallScore != null).length;
    final run = currentCappedRun(points);
    final trends = domainTrends(points);

    // The short-history note. Most real accounts are here today, so it is
    // written as the ordinary state it is, not an error.
    final String? note = points.isEmpty
        ? 'Your first snapshot is saved the next time your score is '
            'calculated.'
        : scoredDays == 0
            ? 'No scored days yet. Days without a score show as gaps.'
            : scoredDays == 1
                ? 'One scored day so far. The line fills in as your score '
                    'is recalculated on later days.'
                : null;

    final muted = AppTheme.tertiaryText(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 0, 15, 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          ScoreHistoryChart(points: points),
          if (note != null) ...[
            const SizedBox(height: 6),
            Padding(
              padding: const EdgeInsets.only(left: 3),
              child: Text(
                note,
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w500,
                  fontSize: 11,
                  height: 1.4,
                  color: AppTheme.secondaryText(context),
                ),
              ),
            ),
          ],
          if (run != null) ...[
            const SizedBox(height: 10),
            _CappedSinceCallout(
              text: cappedSinceLabel(
                run,
                windowFull: history.length >= HealthScoreService.historyLimit,
                now: now,
              ),
            ),
          ],
          const SizedBox(height: 12),
          Row(
            children: [
              for (final domain in ScoreDomain.values)
                Expanded(
                  child: _DomainTrendChip(
                    meta: _domainMeta[domain]!,
                    trend: trends[domain]!,
                    muted: muted,
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }
}

/// Red text on an 8% tint of the same red, both resolved through
/// [AppTheme.alertRed] so the pair lightens together in dark mode.
class _CappedSinceCallout extends StatelessWidget {
  const _CappedSinceCallout({required this.text});

  final String text;

  @override
  Widget build(BuildContext context) {
    final red = AppTheme.alertRed(context, const Color(0xFFE53935));
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: red.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        children: [
          Icon(Icons.shield_outlined, size: 14, color: red),
          const SizedBox(width: 6),
          Expanded(
            child: Text(
              text,
              style: GoogleFonts.inter(
                fontWeight: FontWeight.w600,
                fontSize: 11.5,
                color: red,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// One domain's direction across the window: ↑ better tier, ↓ worse, → same.
/// With fewer than two assessed days there is no direction, so it shows a
/// muted dash rather than a flat arrow that would claim "unchanged".
class _DomainTrendChip extends StatelessWidget {
  const _DomainTrendChip({
    required this.meta,
    required this.trend,
    required this.muted,
  });

  final _DomainMeta meta;
  final DomainTrend trend;
  final Color muted;

  @override
  Widget build(BuildContext context) {
    final brightness = Theme.of(context).brightness;
    final (IconData? icon, String word, Color color) = switch (trend) {
      DomainTrend.improved => (
          Icons.arrow_upward,
          'Better',
          TierSegmentBar.changeColor(up: true, brightness: brightness),
        ),
      DomainTrend.worsened => (
          Icons.arrow_downward,
          'Worse',
          TierSegmentBar.changeColor(up: false, brightness: brightness),
        ),
      DomainTrend.unchanged => (Icons.arrow_forward, 'Same', muted),
      DomainTrend.notEnoughHistory => (null, '-', muted),
    };
    return Column(
      children: [
        Text(
          '${meta.emoji} ${meta.name}',
          style: GoogleFonts.inter(
            fontWeight: FontWeight.w600,
            fontSize: 11,
            color: AppTheme.secondaryText(context),
          ),
        ),
        const SizedBox(height: 3),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (icon != null) ...[
              Icon(icon, size: 13, color: color),
              const SizedBox(width: 2),
            ],
            Text(
              word,
              style: GoogleFonts.inter(
                fontWeight: FontWeight.w700,
                fontSize: 11,
                color: color,
              ),
            ),
          ],
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// 4. Wealth Breakdown — Figma 134:286; four rows incl. Debts, Figma 231:592
// ---------------------------------------------------------------------------

/// The red Figma gives Debts (label and figure). The same pair the Savings
/// screen's "Owed" cluster uses, so debt reads in one red across the app.
Color _debtRed(BuildContext context) => AppTheme.isDark(context)
    ? const Color(0xFFE53935)
    : const Color(0xFFC62828);

/// Every credit and loan account across ALL connected institutions, largest
/// balance owed first. Portfolio-wide by design — Debts is one figure for the
/// user, not a per-bank one.
List<PlaidAccount> debtAccountsByOwed(Iterable<PlaidAccount> accounts) =>
    [...accounts.where((a) => a.isDebt)]
      ..sort((a, b) => b.balanceOwed.compareTo(a.balanceOwed));

/// Savings' per-bank groups: [SavingsSummary.byInstitution]'s grouping and
/// order, keeping only depository accounts — debt accounts belong to the
/// Debts row, and an institution holding nothing but debt drops out here.
Map<String, List<PlaidAccount>> savingsByInstitution(SavingsSummary summary) => {
      for (final entry in summary.byInstitution.entries)
        if (entry.value.any((a) => a.isDepository))
          entry.key: [...entry.value.where((a) => a.isDepository)],
    };

/// The aggregate debt-split line, from the one cached ratio — never a
/// per-account label (that would need the backend's subtype tables in Dart).
///
/// Stated as a share, not dollars, on purpose: `healthy_debt_ratio` is
/// healthy ÷ CLASSIFIED debt, and scoring.py leaves unclassified subtypes
/// (business, construction, "other", anything new) out of both halves. Ratio
/// × [PlaidAccountTotals.owedTotal] would therefore print dollar figures that
/// quietly disagree with the balances listed above them whenever such an
/// account exists — and the client cannot tell when that is.
///
/// Null [profile] is a score state that carries none (Loading, NeverComputed,
/// Failed); a null ratio is "not computable". Both read "not available".
String debtSplitLabel(ScoreProfile? profile) {
  final ratio = profile?.healthyDebtRatio;
  if (ratio == null) {
    return 'Debt split not available yet. It is worked out when your '
        'score next refreshes';
  }
  final healthy = (ratio * 100).round();
  return 'Of the debt we can classify, $healthy% is structured (mortgage, '
      'student, auto, home equity) and ${100 - healthy}% is revolving or '
      'unsecured';
}

/// "12.3%" — one decimal, as the Investments holdings table prints it.
String _sharePct(double share) => '${(share * 100).toStringAsFixed(1)}%';

/// Four independent rows, each reading the collection its own screen already
/// uses and summing it with that screen's existing summary class —
/// [SavingsSummary], [PortfolioSummary], [CoverageSummary] and, for Debts,
/// [PlaidAccountTotals]. No new aggregation logic, and no second definition
/// of "total savings" or "debt" to drift from the Savings screen's. The docs
/// come from [HomeScreen]'s listeners, which the score card shares; null
/// means not loaded yet (or a listen error).
///
/// Public, and Firebase-free beyond the doc lists it is handed, so the rows can
/// be widget-tested directly — the same reason [NetWorthCardBody] is.
class WealthBreakdownCard extends StatelessWidget {
  const WealthBreakdownCard({
    super.key,
    required this.bankDocs,
    required this.investmentDocs,
    required this.policyDocs,
    required this.score,
    this.onOpenInvestments,
  });

  // Spelled out rather than _Docs: a public constructor can't expose the
  // private alias (library_private_types_in_public_api).
  final List<QueryDocumentSnapshot<Map<String, dynamic>>>? bankDocs;
  final List<QueryDocumentSnapshot<Map<String, dynamic>>>? investmentDocs;
  final List<QueryDocumentSnapshot<Map<String, dynamic>>>? policyDocs;

  /// Only for Debts' split line, which reads the cached healthy_debt_ratio.
  final ScoreState score;

  /// Opens the Investments tab from the capped holdings list's "See all"
  /// link. Null leaves the link as plain text.
  final VoidCallback? onOpenInvestments;

  /// The expanded Investments row lists at most this many holdings. A display
  /// cap only: shares are still of the whole portfolio, and the order is
  /// still [PortfolioSummary.positions]' value-descending one.
  static const holdingsShown = 10;

  ScoreProfile? get _profile => switch (score) {
        ScorePartial(:final profile) => profile,
        ScoreComplete(:final profile) => profile,
        ScoreUnavailable(:final profile) => profile,
        ScoreLoading() || ScoreNeverComputed() || ScoreFailed() => null,
      };

  static List<PlaidAccount> _bankAccounts(_Docs docs) =>
      [for (final d in docs) PlaidAccount.fromFirestore(d.id, d.data())];

  @override
  Widget build(BuildContext context) {
    // Keyed so each row's open state stays with that row if the row count
    // ever changes (see CollapsibleCard). Debts is always rendered, even at
    // zero debt, so today the count is fixed at four.
    return Column(
      children: [
        _BreakdownRow(
          key: const ValueKey('wealth-breakdown-savings'),
          label: 'Savings',
          docs: bankDocs,
          summarise: (docs) {
            final accounts = _bankAccounts(docs);
            // Transactions are only needed for spend and month-over-month
            // change, neither of which this row shows — so none are fetched.
            final summary = SavingsSummary(
              accounts: accounts,
              transactionsByAccount: const {},
            );
            final groups = savingsByInstitution(summary);
            return (
              total: formatSgd(summary.totalSavings),
              red: false,
              detail: accounts.isEmpty
                  ? const _DetailNote('No bank accounts connected yet. '
                      'Connect a bank from the Savings tab.')
                  : groups.isEmpty
                      ? const _DetailNote(
                          'No savings or checking accounts connected yet')
                      : _DetailList(children: [
                          for (final group in groups.entries) ...[
                            _DetailGroupLabel(group.key),
                            for (final account in group.value)
                              _DetailRow(
                                primary: account.accountName,
                                secondary: account.maskedLabel,
                                // balance_current, NOT the Savings screen's
                                // available-first rowBalance: the header above
                                // is SavingsSummary.totalSavings, a
                                // balance_current sum, and these rows must add
                                // up to it.
                                value: account.balanceCurrent == null
                                    ? '-'
                                    : formatSgd(account.balanceCurrent!),
                              ),
                          ],
                        ]),
            );
          },
        ),
        const SizedBox(height: 12),
        _BreakdownRow(
          key: const ValueKey('wealth-breakdown-investments'),
          label: 'Investments',
          docs: investmentDocs,
          summarise: (docs) {
            final accounts = [
              for (final d in docs)
                InvestmentAccount.fromFirestore(d.id, d.data()),
            ];
            final summary = PortfolioSummary(accounts: accounts);
            return (
              total: formatSgd(summary.totalValue),
              red: false,
              detail: accounts.isEmpty
                  ? const _DetailNote('No investment accounts connected yet. '
                      'Connect a broker from the Investments tab.')
                  : summary.positions.isEmpty
                      ? const _DetailNote('No holdings in your connected '
                          'investment accounts yet')
                      : _DetailList(children: [
                          // positions is already value-descending — the
                          // Investments holdings table's default order.
                          for (final p in summary.positions
                              .take(holdingsShown))
                            _DetailRow(
                              primary: p.holding.shortLabel,
                              secondary: p.holding.displayName ==
                                      p.holding.shortLabel
                                  ? p.account.accountName
                                  : p.holding.displayName,
                              value: p.holding.marketValue == null
                                  ? '-'
                                  : formatSgd(p.holding.marketValue!),
                              valueCaption: _sharePct(p.share),
                            ),
                          if (summary.holdingsCount > holdingsShown)
                            _SeeAllLink(
                              label: 'See all ${summary.holdingsCount} '
                                  'holdings on Investments',
                              onTap: onOpenInvestments,
                            ),
                        ]),
            );
          },
        ),
        const SizedBox(height: 12),
        _BreakdownRow(
          key: const ValueKey('wealth-breakdown-insurance'),
          label: 'Insurance Coverage',
          docs: policyDocs,
          summarise: (docs) {
            final policies = [
              for (final d in docs)
                InsurancePolicy.fromFirestore(d.id, d.data()),
            ];
            final summary = CoverageSummary.fromPolicies(policies);
            return (
              total: formatSgd(summary.totalSumAssured),
              red: false,
              detail: policies.isEmpty
                  ? const _DetailNote('No policies uploaded yet. '
                      'Upload one from the Insurance tab.')
                  : _DetailList(children: [
                      for (final policy in policies)
                        _DetailRow(
                          primary: policy.policyName,
                          // Type first: it is short, and insurer names run
                          // long ("Prudential Assurance Company Singapore
                          // (Pte) Limited") — truncation should cut those,
                          // not the type.
                          secondary:
                              '${policy.policyTypeLabel} · ${policy.insurer}',
                          // Null is the parser's fabrication-avoidance rule
                          // for personal_accident/other — shown as the
                          // Insurance screen shows it, not as $0.
                          value: policy.sumAssuredSgd == null
                              ? '-'
                              : formatSgd(policy.sumAssuredSgd!),
                        ),
                      _DetailNote(
                        '${formatSgd(summary.totalAnnualPremiums)} a year '
                        'in premiums',
                        padded: false,
                      ),
                    ]),
            );
          },
        ),
        const SizedBox(height: 12),
        _BreakdownRow(
          key: const ValueKey('wealth-breakdown-debts'),
          label: 'Debts',
          docs: bankDocs,
          summarise: (docs) {
            final accounts = _bankAccounts(docs);
            final debts = debtAccountsByOwed(accounts);
            final owed = accounts.owedTotal;
            if (accounts.isEmpty) {
              // Nothing connected: debt is unknown, not zero. 'n/a' rather
              // than '-', which would sit where a real "-$1,234" goes.
              return (
                total: 'n/a',
                red: false,
                detail: const _DetailNote('No banks connected yet, so we '
                    'can\'t see any debt. Connect a bank from the Savings '
                    'tab.'),
              );
            }
            if (debts.isEmpty) {
              return (
                total: formatSgd(0),
                red: false,
                detail: const _DetailNote('No debt on your connected '
                    'accounts, nice.'),
              );
            }
            return (
              // Figma 231:592 prints the owed total as a negative, in red.
              total: formatSgd(-owed),
              red: owed > 0,
              detail: _DetailList(children: [
                for (final account in debts)
                  _DetailRow(
                    primary: account.institutionName ?? 'Bank',
                    secondary: account.maskedLabel,
                    // balanceOwed reads balance_current only — never the
                    // remaining credit line (see PlaidAccount.balanceOwed).
                    value: formatSgd(account.balanceOwed),
                  ),
                _DetailNote(debtSplitLabel(_profile), padded: false),
              ]),
            );
          },
        ),
      ],
    );
  }
}

typedef _BreakdownSummary = ({String total, bool red, Widget detail});

class _BreakdownRow extends StatelessWidget {
  const _BreakdownRow({
    super.key,
    required this.label,
    required this.docs,
    required this.summarise,
  });

  final String label;
  final _Docs? docs;
  final _BreakdownSummary Function(_Docs) summarise;

  @override
  Widget build(BuildContext context) {
    final docs = this.docs;
    final summary = docs == null ? null : summarise(docs);
    final ink = summary != null && summary.red
        ? _debtRed(context)
        : AppTheme.primaryText(context);
    return CollapsibleCard(
      title: label,
      titleSize: 17,
      titleColor: ink,
      trailing: Text(
        summary?.total ?? '-',
        style: GoogleFonts.spaceGrotesk(
          fontWeight: FontWeight.w700,
          fontSize: 17,
          color: ink,
        ),
      ),
      enabled: summary != null,
      body: summary?.detail ?? const SizedBox(width: double.infinity),
    );
  }
}

/// The last line of a capped list, leading to the tab that shows the rest —
/// the capped score message's pattern: an InkWell, button semantics, and a
/// trailing arrow only when there is somewhere to go.
class _SeeAllLink extends StatelessWidget {
  const _SeeAllLink({required this.label, required this.onTap});

  final String label;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: onTap != null,
      hint: onTap == null ? null : 'Opens Investments',
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(8),
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 4),
          child: Text(
            onTap == null ? label : '$label →',
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w600,
              fontSize: 12,
              color: AppTheme.primaryText(context),
            ),
          ),
        ),
      ),
    );
  }
}

/// The expanded body's frame: a hairline under the header, then the rows.
class _DetailList extends StatelessWidget {
  const _DetailList({required this.children});

  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(15, 0, 15, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Divider(height: 1, thickness: 1, color: AppTheme.divider(context)),
          for (final child in children) ...[
            const SizedBox(height: 10),
            child,
          ],
        ],
      ),
    );
  }
}

/// One line of copy: an empty state on its own, or a footnote in a list.
class _DetailNote extends StatelessWidget {
  const _DetailNote(this.text, {this.padded = true});

  final String text;

  /// False inside a [_DetailList], which already pads its children.
  final bool padded;

  @override
  Widget build(BuildContext context) {
    final note = Text(
      text,
      style: GoogleFonts.inter(
        fontWeight: FontWeight.w500,
        fontSize: 11.5,
        color: AppTheme.secondaryText(context),
      ),
    );
    return padded
        ? Padding(
            padding: const EdgeInsets.fromLTRB(15, 0, 15, 12),
            child: SizedBox(width: double.infinity, child: note),
          )
        : note;
  }
}

/// An institution name heading its accounts in the Savings row.
class _DetailGroupLabel extends StatelessWidget {
  const _DetailGroupLabel(this.name);

  final String name;

  @override
  Widget build(BuildContext context) {
    return Text(
      name,
      maxLines: 1,
      overflow: TextOverflow.ellipsis,
      style: GoogleFonts.spaceGrotesk(
        fontWeight: FontWeight.w700,
        fontSize: 13,
        color: AppTheme.primaryText(context),
      ),
    );
  }
}

/// A compact item: what it is on the left, its figure on the right.
class _DetailRow extends StatelessWidget {
  const _DetailRow({
    required this.primary,
    required this.secondary,
    required this.value,
    this.valueCaption,
  });

  final String primary;
  final String secondary;
  final String value;
  final String? valueCaption;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                primary,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w600,
                  fontSize: 12.5,
                  color: AppTheme.primaryText(context),
                ),
              ),
              const SizedBox(height: 2),
              Text(
                secondary,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w500,
                  fontSize: 11,
                  color: AppTheme.tertiaryText(context),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(width: 12),
        Column(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(
              value,
              style: GoogleFonts.spaceGrotesk(
                fontWeight: FontWeight.w700,
                fontSize: 13,
                color: AppTheme.primaryText(context),
              ),
            ),
            if (valueCaption != null) ...[
              const SizedBox(height: 2),
              Text(
                valueCaption!,
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w500,
                  fontSize: 11,
                  color: AppTheme.tertiaryText(context),
                ),
              ),
            ],
          ],
        ),
      ],
    );
  }
}
