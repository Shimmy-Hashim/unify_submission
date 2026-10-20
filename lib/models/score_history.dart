/// users/{uid}/scoreHistory — one backend-written snapshot per UTC day
/// (`data/score_profile.py::snapshot_payload`).
///
/// Deserialization plus the comparisons Home draws from it: the score card's
/// trend line, and the Score History card's capped streak and domain trends.
/// Like score_profile.dart it re-derives nothing the pipeline owns — scores,
/// `capped` and tier indexes are read back verbatim, and the only arithmetic
/// is comparing them across days.
library;

import 'package:cloud_firestore/cloud_firestore.dart';

import 'score_profile.dart';

/// The fields of one daily snapshot that Home reads — the score card (date,
/// score) and the Score History card (also `capped` and the domain tiers).
class ScoreHistoryPoint {
  const ScoreHistoryPoint({
    required this.date,
    required this.overallScore,
    this.computedAt,
    this.capped,
    this.domainTierIndex = const {},
  });

  /// The snapshot's UTC calendar day — the document ID, parsed. Null when the
  /// stored `date` string is malformed, so the point is ignored rather than
  /// crashing the parse.
  final DateTime? date;

  /// Null on a day the score could not be computed. That is a real gap, not a
  /// zero, and is skipped when looking for a value to compare against.
  final double? overallScore;

  final DateTime? computedAt;

  /// `score_breakdown.capped`, verbatim. Null on a day with no breakdown (no
  /// score) — unknown, which is not the same as "not capped".
  final bool? capped;

  /// `domains[d].tier_index` for each domain that was assessed that day
  /// (0 = healthiest, the artifacts' authoritative key). An unassessed domain
  /// is absent rather than mapped to a guess.
  final Map<ScoreDomain, int> domainTierIndex;

  factory ScoreHistoryPoint.fromMap(Map<String, dynamic> data) {
    final computedAt = data['computed_at'];
    final breakdown = data['score_breakdown'];
    final domains = data['domains'];
    return ScoreHistoryPoint(
      date: parseSnapshotDate(data['date']),
      overallScore: (data['overall_score'] as num?)?.toDouble(),
      computedAt: computedAt is Timestamp ? computedAt.toDate() : null,
      capped: breakdown is Map ? breakdown['capped'] as bool? : null,
      domainTierIndex: {
        if (domains is Map)
          for (final domain in ScoreDomain.values)
            if (domains[domain.key] is Map &&
                (domains[domain.key] as Map)['status'] == 'assessed' &&
                (domains[domain.key] as Map)['tier_index'] is num)
              domain: ((domains[domain.key] as Map)['tier_index'] as num)
                  .toInt(),
      },
    );
  }
}

/// "2026-09-13" -> DateTime.utc(2026, 9, 13). Strict YYYY-MM-DD only, because
/// that is the only shape `snapshot_date()` writes.
DateTime? parseSnapshotDate(Object? value) {
  if (value is! String) return null;
  final match = RegExp(r'^(\d{4})-(\d{2})-(\d{2})$').firstMatch(value);
  if (match == null) return null;
  return DateTime.utc(
    int.parse(match.group(1)!),
    int.parse(match.group(2)!),
    int.parse(match.group(3)!),
  );
}

/// The UTC day a computation lands on — the same rule the backend uses to key
/// the snapshot document, so "today's snapshot" means the same day here.
DateTime snapshotDayOf(DateTime computedAt) {
  final utc = computedAt.toUtc();
  return DateTime.utc(utc.year, utc.month, utc.day);
}

/// Change in the displayed score since the last real prior value.
class ScoreTrend {
  const ScoreTrend({required this.deltaPoints, required this.since});

  /// Whole points, taken between the two ROUNDED scores so the trend always
  /// agrees with the numbers a user sees on screen (71.6 -> 72 vs 68.4 -> 68
  /// reads +4, never +3).
  final int deltaPoints;

  /// The UTC day of the snapshot being compared against.
  final DateTime since;
}

/// The trend for the score currently on screen, or null when there is nothing
/// honest to compare it with.
///
/// "Current" is the live cached score ([currentScore], computed at
/// [currentComputedAt]) rather than today's snapshot: they are written from
/// the same result, and the live value is the one the card shows.
///
/// "Prior" is the most recent snapshot that:
///   - is from a UTC day strictly BEFORE the current computation's day — today's
///     own snapshot is the current score, not a prior one; and
///   - has a non-null overall_score — a day the score could not be computed is
///     skipped, so a gap never becomes the baseline and a comparison never
///     silently reaches for "yesterday" when yesterday has no score.
///
/// Returns null when no such snapshot exists (a first-ever score, or a history
/// made entirely of gap days). The card then shows no trend line at all rather
/// than a fabricated "+0".
ScoreTrend? computeScoreTrend({
  required double currentScore,
  required DateTime currentComputedAt,
  required Iterable<ScoreHistoryPoint> history,
}) {
  final currentDay = snapshotDayOf(currentComputedAt);
  ScoreHistoryPoint? prior;
  for (final point in history) {
    final date = point.date;
    if (date == null || point.overallScore == null) continue;
    if (!date.isBefore(currentDay)) continue;
    if (prior == null || date.isAfter(prior.date!)) prior = point;
  }
  if (prior == null) return null;
  return ScoreTrend(
    deltaPoints: currentScore.round() - prior.overallScore!.round(),
    since: prior.date!,
  );
}

/// Snapshots with a readable date, oldest first — the order a chart reads.
List<ScoreHistoryPoint> chronological(Iterable<ScoreHistoryPoint> history) =>
    [for (final p in history) if (p.date != null) p]
      ..sort((a, b) => a.date!.compareTo(b.date!));

/// When the current capped streak began, for "Capped … since `date`".
class CappedRun {
  const CappedRun({required this.since, required this.startsAtEarliestLoaded});

  /// The first capped day of the streak.
  final DateTime since;

  /// True when the streak runs all the way back to the oldest snapshot
  /// available, so the real start may be earlier than [since] — only
  /// possible to rule out when the window holds the user's whole history.
  final bool startsAtEarliestLoaded;
}

/// The most recent continuous run of capped days, or null when the score is
/// not capped now.
///
/// "Now" is the NEWEST snapshot. If that day has no score, or its breakdown
/// says `capped: false`, there is no current cap and so no callout — the same
/// condition under which the score card shows no capped message.
///
/// Walking back from the newest day, the run continues through every
/// `capped: true` day and ends at the first explicit `capped: false`. A gap day
/// (no score, `capped` null) is skipped without ending the run: nothing was
/// measured that day, and an unmeasured day is not evidence the cap lifted.
/// The run's start is its earliest `capped: true` day, never a gap day.
CappedRun? currentCappedRun(Iterable<ScoreHistoryPoint> history) {
  final points = chronological(history);
  if (points.isEmpty || points.last.capped != true) return null;
  var start = points.last;
  var reachedEarliest = true;
  for (var i = points.length - 2; i >= 0; i--) {
    final capped = points[i].capped;
    if (capped == false) {
      reachedEarliest = false;
      break;
    }
    if (capped == true) start = points[i];
  }
  return CappedRun(
    since: start.date!,
    // True whenever the scan ran out of data without meeting an explicit
    // `capped: false` — including when only gap days lie before [start].
    startsAtEarliestLoaded: reachedEarliest,
  );
}

/// Direction of one domain's tier across the window.
enum DomainTrend {
  improved,
  worsened,
  unchanged,

  /// No trend to show: fewer than two assessed snapshots, or the latest
  /// snapshot has no tier for the domain. Rendered as '-'.
  notEnoughHistory,
}

/// Each domain's tier_index on its earliest vs. latest ASSESSED snapshot in
/// [history]. Lower tier_index is healthier, so a drop is an improvement.
///
/// Only while the domain is assessed on the LATEST snapshot. With no tier
/// today there is no "now" to compare against, and a direction drawn between
/// two older days would sit under a score card row reading "Not connected".
///
/// Earlier unassessed days, though, are skipped rather than blocking the
/// comparison: a day where only that domain went unassessed (say a bank
/// disconnected) says nothing about its tier.
Map<ScoreDomain, DomainTrend> domainTrends(Iterable<ScoreHistoryPoint> history) {
  final points = chronological(history);
  return {
    for (final domain in ScoreDomain.values)
      domain: () {
        if (points.isEmpty || points.last.domainTierIndex[domain] == null) {
          return DomainTrend.notEnoughHistory;
        }
        final assessed = [
          for (final p in points)
            if (p.domainTierIndex[domain] != null) p.domainTierIndex[domain]!,
        ];
        if (assessed.length < 2) return DomainTrend.notEnoughHistory;
        final delta = assessed.last - assessed.first;
        if (delta < 0) return DomainTrend.improved;
        if (delta > 0) return DomainTrend.worsened;
        return DomainTrend.unchanged;
      }(),
  };
}
