import 'package:flutter/foundation.dart';

import 'score_profile.dart';

/// Every state the Home screen's score can be in.
///
/// WHY A SEALED CLASS AND NOT AN ENUM + NULLABLE PAYLOAD
/// The states carry different payloads — [ScoreUnavailable] carries a reason
/// and no score, [ScorePartial] carries a profile with a guaranteed-null
/// overall score, [ScoreComplete] carries one with a guaranteed-non-null score.
/// A single class with six nullable fields would push "which fields are safe to
/// read here?" into the UI, which is exactly the bug this replaces. Sealing it
/// makes `switch` exhaustive: adding a state later fails to compile at every
/// place that has to handle it, rather than silently falling through.
///
/// WHY "REFRESHING" IS NOT ONE OF THESE VARIANTS
/// A pull-to-refresh must show the data that is already on screen with a
/// spinner over it, never blank it out — so refreshing is orthogonal to which
/// data state we are in, not a replacement for it. Making it a variant would
/// mean either losing the payload or duplicating every other variant with a
/// refreshing twin. It lives on [HomeScoreState] instead, alongside the score
/// state it decorates.
sealed class ScoreState {
  const ScoreState();
}

/// First fetch, before any snapshot has arrived. Distinct from
/// [ScoreNeverComputed]: this one means "we don't know yet", that one means "we
/// know, and the answer is that it has never run".
final class ScoreLoading extends ScoreState {
  const ScoreLoading();
}

/// The financialProfile document exists but the pipeline has never run for this
/// user — no `checklist_last_updated`.
///
/// Normal for a brand-new account, and the reason POST /score/refresh exists:
/// onboarding writes monthly_income_sgd client-side, so nothing server-side
/// notices that a score just became computable.
final class ScoreNeverComputed extends ScoreState {
  const ScoreNeverComputed();
}

/// The pipeline ran (or could not start) and no part of the score is
/// computable. [reason] is the only thing to branch copy on — the four cases
/// need genuinely different messages and different next actions.
///
/// [profile] is the cached run this reason came from, and is null only for
/// [ScoreUnavailableReason.noFinancialProfile], where no document exists to
/// cache anything onto.
final class ScoreUnavailable extends ScoreState {
  const ScoreUnavailable(this.reason, [this.profile]);

  final ScoreUnavailableReason reason;
  final ScoreProfile? profile;
}

/// Some domains have a tier, others do not, and the overall score is withheld
/// ([ScoreReason.incompleteDomains]).
///
/// The ordinary state for a user who has connected a bank but no investment
/// account, or vice versa. [unassessedDomains] each carry their own reason and
/// `missingFeatures`, which is what turns "no Banking tier" into "connect a
/// bank account" without the UI re-deriving anything.
final class ScorePartial extends ScoreState {
  const ScorePartial(this.profile);

  final ScoreProfile profile;

  List<DomainScore> get assessedDomains => profile.assessedDomains;
  List<DomainScore> get unassessedDomains => profile.unassessedDomains;
}

/// All three domain tiers resolved and an overall score computed.
///
/// `overall_score` is non-null if and only if all three tiers were available —
/// `compute_score_breakdown()` returns None otherwise — so [overallScore] and
/// [breakdown] are safe to read unconditionally here.
final class ScoreComplete extends ScoreState {
  const ScoreComplete(this.profile, this.overallScore, this.breakdown);

  final ScoreProfile profile;
  final double overallScore;
  final ScoreBreakdown breakdown;

  List<DomainScore> get domains => profile.orderedDomains;

  /// 0-2 cross-domain flags. Only meaningful because all three tiers exist, so
  /// `layer3_evaluated` is necessarily true here — an empty list genuinely
  /// means "nothing fired".
  List<String> get layer3Flags => profile.layer3Flags;
}

/// Something went wrong — a Firestore listen failure, a permission error, a
/// malformed cache. Deliberately NOT any of the data-not-ready states above:
/// this is "something broke", not "not ready yet", and the two must not share
/// a message.
final class ScoreFailed extends ScoreState {
  const ScoreFailed(this.message, [this.error]);

  /// Safe to show to the user.
  final String message;
  final Object? error;
}

/// Whether the user has any source data for each domain, read from the real
/// collections Home already streams — never inferred from the score cache.
///
/// WHY NOT THE REASON CODE
/// `missing_features` fires identically for "no accounts at all" and "accounts
/// connected, one feature not computable" (zero monthly expenses leaves
/// emergency_fund_months null; a debt-free user has no healthy_debt_ratio), and
/// the length of the missing list is no discriminator either — a credit-only
/// user loses both depository features while genuinely being connected. Only
/// the collections themselves can answer "is anything connected?".
///
/// Each flag is null until its first snapshot lands, which is "don't know yet",
/// not "no accounts".
@immutable
class AccountPresence {
  const AccountPresence({this.hasBankAccounts, this.hasInvestmentAccounts});

  /// At least one plaidAccounts document, of any account_type — depository,
  /// credit and loan all count.
  final bool? hasBankAccounts;

  /// At least one investmentAccounts document.
  final bool? hasInvestmentAccounts;

  /// Insurance is always true: there is nothing to connect. The pipeline
  /// scores zero policies as a real "Not Covered" tier, so an unassessed
  /// Insurance row is never a missing connection.
  bool? hasSourceData(ScoreDomain domain) => switch (domain) {
    ScoreDomain.banking => hasBankAccounts,
    ScoreDomain.investment => hasInvestmentAccounts,
    ScoreDomain.insurance => true,
  };
}

/// What one Home tier row says about its domain.
enum DomainRowStatus {
  /// A real tier from the cache.
  assessed,

  /// Not assessed, and nothing of the relevant type is connected.
  notConnected,

  /// Not assessed, but data exists — the score hasn't run yet, or one of the
  /// domain's figures can't be worked out from it.
  cannotCalculateYet,

  /// Not assessed, and account presence hasn't loaded yet.
  unknown,
}

/// The three-way (plus loading) decision for one tier row. [result] is null
/// when no score has ever been computed.
DomainRowStatus domainRowStatus(
  ScoreDomain domain,
  DomainScore? result,
  AccountPresence presence,
) {
  if (result?.isAssessed ?? false) return DomainRowStatus.assessed;
  return switch (presence.hasSourceData(domain)) {
    null => DomainRowStatus.unknown,
    false => DomainRowStatus.notConnected,
    true => DomainRowStatus.cannotCalculateYet,
  };
}

/// What the Home screen renders: the score state, plus whether a manual
/// refresh is in flight over it.
@immutable
class HomeScoreState {
  const HomeScoreState({
    this.score = const ScoreLoading(),
    this.refreshing = false,
    this.refreshError,
  });

  final ScoreState score;

  /// True while POST /score/refresh is in flight (and while waiting for the
  /// recomputed result to arrive back down the Firestore stream). The UI keeps
  /// showing [score] and overlays an indicator.
  final bool refreshing;

  /// A failed manual refresh. Distinct from [ScoreFailed]: the stream is still
  /// healthy and the data on screen is still valid — only the recompute
  /// request failed, so this belongs in a snackbar, not in place of the card.
  final String? refreshError;

  HomeScoreState copyWith({
    ScoreState? score,
    bool? refreshing,
    String? refreshError,
    bool clearRefreshError = false,
  }) {
    return HomeScoreState(
      score: score ?? this.score,
      refreshing: refreshing ?? this.refreshing,
      refreshError:
          clearRefreshError ? null : (refreshError ?? this.refreshError),
    );
  }
}
