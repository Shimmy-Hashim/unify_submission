import 'package:flutter/widgets.dart';
import 'package:flutter/material.dart' show Icons;

import '../utils/formatters.dart';
import 'home_score_state.dart';
import 'score_profile.dart';

/// The Insurance screen's Coverage Gaps section, as data — Figma node 84:661.
///
/// WHAT THIS SECTION IS ALLOWED TO SAY
/// Only what the backend has actually assessed. That is exactly four things,
/// and each one has a named benchmark behind it (the Insurance artifact's own
/// `out_of_model_benchmarks` block, which carries the citations):
///
///   life cover            >= 9x annual income  (5x in band 5)  LIA PGS 2022
///   critical illness      >= 3.9x annual income               LIA PGS 2022
///   hospitalisation       cover beyond MediShield Life        MOH ~71% IP prevalence
///   income protection     any disability/income cover         AIA DI Survey 2022 ~53%
///
/// The Figma shell's third card was "Accident Cover". There is no benchmark to
/// measure it against, and `compute_insurance_features()` deliberately excludes
/// personal_accident from every coverage calculation (an accidental-death
/// benefit inflating a life ratio was a confirmed real-world failure), so a
/// card for it could only ever read "no assessment available". It is replaced
/// by the two binary checks the backend really does run — same card scaffold,
/// same strip, four cards instead of three.
///
/// WHERE EACH NUMBER AND EACH SENTENCE COMES FROM
/// The two ratio lines take their ratio, their benchmark and their SGD
/// shortfall from `ml_profile.insurance_coverage`; only the sentence around
/// them is composed here, which is a presentation decision in the same class as
/// `formatSgd` itself. The two binary lines take their sentence verbatim from
/// the backend's Layer 1 checklist — that copy branches on dependants, which
/// this layer does not evaluate and must not guess at.
///
/// Nothing here invents a threshold, an urgency or a recommendation. A gap is
/// stated as the distance to a published benchmark, and cover that meets its
/// benchmark says so plainly rather than being left silent.

/// How one coverage line currently stands. Drives the pill and its colour only.
enum CoverageStatus {
  /// Meets its benchmark, or the binary check passed.
  covered,

  /// Assessed, and short of the benchmark.
  gap,

  /// Not assessed — the backend could not evaluate this line. Never rendered
  /// as a gap: an unknown must not read as a confirmed exposure.
  notAssessed,
}

/// One card in the Coverage Gaps strip.
class CoverageGapItem {
  const CoverageGapItem({
    required this.icon,
    required this.title,
    required this.message,
    required this.status,
  });

  /// A Material icon, not an emoji. These four lines used to carry shield,
  /// hospital, stethoscope and briefcase emoji, which render in the platform's
  /// own emoji font — a different weight, colour and baseline from every other
  /// mark on the screen, and not restyleable per theme. They are now the same
  /// icon system the nav bar and Settings use.
  final IconData icon;
  final String title;
  final String message;
  final CoverageStatus status;

  String get pillLabel => switch (status) {
        CoverageStatus.covered => 'Covered',
        CoverageStatus.gap => 'Gap',
        CoverageStatus.notAssessed => 'Not assessed',
      };
}

/// The whole section's content for one score state.
class CoverageGapsView {
  const CoverageGapsView._({this.items = const [], this.summary, this.notice});

  /// The four cards, or empty when nothing has been assessed.
  final List<CoverageGapItem> items;

  /// One line above the strip stating where the user stands overall. Null when
  /// there is nothing to summarise.
  final String? summary;

  /// Shown instead of the strip: why there is nothing to show yet, and what
  /// would change that. Null when [items] is populated.
  final String? notice;

  bool get hasItems => items.isNotEmpty;

  int get gapCount =>
      items.where((i) => i.status == CoverageStatus.gap).length;

  /// Builds the section from the Home score state — the same cached document
  /// every other score surface reads.
  ///
  /// The "no items" branches are deliberately specific: "we have not run yet",
  /// "we cannot run without your income" and "your age is outside the models'
  /// range" need different next actions, and collapsing them into one grey
  /// placeholder is what the shell this replaces was doing.
  factory CoverageGapsView.fromScoreState(ScoreState state) {
    switch (state) {
      case ScoreLoading():
        return const CoverageGapsView._(notice: 'Checking your cover…');

      case ScoreFailed():
        return const CoverageGapsView._(
          notice: 'Could not load your coverage assessment right now.',
        );

      case ScoreNeverComputed():
        return const CoverageGapsView._(
          notice: 'Your cover has not been assessed yet. It runs the next time '
              'you add a policy or sync an account.',
        );

      case ScoreUnavailable(:final reason):
        return CoverageGapsView._(notice: _unavailableNotice(reason));

      // A partial score still assesses insurance — a user with no bank
      // connected has no overall score but their cover is fully assessable, so
      // both states go through the same builder.
      case ScoreComplete(:final profile) || ScorePartial(:final profile):
        return _fromProfile(profile);
    }
  }

  static String _unavailableNotice(ScoreUnavailableReason reason) =>
      switch (reason) {
        ScoreUnavailableReason.incomeNotSet =>
          'Set your monthly income in Settings to see how your cover compares '
              'to the LIA benchmarks.',
        ScoreUnavailableReason.dateOfBirthNotSet ||
        ScoreUnavailableReason.noFinancialProfile =>
          'Finish setting up your profile to see your coverage gaps.',
        ScoreUnavailableReason.unsupportedAge =>
          'The coverage benchmarks cover ages 20 to 70, so your cover has not '
              'been assessed.',
        ScoreUnavailableReason.unknown =>
          'Your cover has not been assessed yet.',
      };

  static CoverageGapsView _fromProfile(ScoreProfile profile) {
    final coverage = profile.insuranceCoverage;
    if (coverage == null) {
      // The score ran but carries no coverage block — a cache written before
      // schema 1.1.0. Nothing false is shown; the next refresh fills it in.
      return const CoverageGapsView._(
        notice: 'Your cover has not been assessed yet. It runs the next time '
            'you add a policy or sync an account.',
      );
    }

    final checks = {for (final c in profile.checklist) c.check: c};
    final items = [
      _ratioItem(Icons.shield_outlined, 'Life Cover', coverage.life),
      _ratioItem(
          Icons.favorite_border, 'Critical Illness', coverage.criticalIllness),
      _checkItem(Icons.local_hospital_outlined, 'Hospitalisation',
          checks['hospitalisation_coverage']),
      _checkItem(Icons.work_outline, 'Income Protection',
          checks['income_protection_coverage']),
    ];
    return CoverageGapsView._(items: items, summary: _summaryFor(items));
  }

  /// The honest positive state the shell could not express: "no gaps" is a
  /// finding, not an absence, and it is only claimable when all four lines were
  /// actually assessed.
  static String? _summaryFor(List<CoverageGapItem> items) {
    final unknown =
        items.where((i) => i.status == CoverageStatus.notAssessed).length;
    final gaps = items.where((i) => i.status == CoverageStatus.gap).length;
    if (unknown == items.length) return null;
    if (unknown > 0) {
      return '$gaps of ${items.length - unknown} assessed benchmarks not met.';
    }
    if (gaps == 0) {
      return 'No gaps: your cover meets all ${items.length} benchmarks.';
    }
    return '$gaps of ${items.length} benchmarks not met.';
  }

  static CoverageGapItem _ratioItem(
      IconData icon, String title, CoverageGap gap) {
    final benchmark = _multiple(gap.benchmarkRatio);
    final String message;
    if (gap.meetsBenchmark) {
      message = '${_multiple(gap.ratio)} of income covered, at or above the '
          'LIA-recommended $benchmark.';
    } else if (gap.hasNoCover) {
      message = 'No cover of this type on file. The LIA benchmark is '
          '$benchmark of income: ${formatSgd(gap.recommendedSgd)} for you.';
    } else {
      message = '${_multiple(gap.ratio)} of income covered, against the '
          'LIA-recommended $benchmark: ${formatSgd(gap.shortfallSgd)} short.';
    }
    return CoverageGapItem(
      icon: icon,
      title: title,
      message: message,
      status: gap.meetsBenchmark ? CoverageStatus.covered : CoverageStatus.gap,
    );
  }

  static CoverageGapItem _checkItem(
      IconData icon, String title, ChecklistCheck? check) {
    // A null `passed` is the backend's "could not be evaluated", which it keeps
    // distinct from a failure on purpose — mirrored here rather than folded
    // into a gap.
    final status = switch (check?.passed) {
      true => CoverageStatus.covered,
      false => CoverageStatus.gap,
      null => CoverageStatus.notAssessed,
    };
    return CoverageGapItem(
      icon: icon,
      title: title,
      // Backend-owned copy: the failure wording branches on dependants, which
      // this layer never reads.
      message: check?.message ?? 'This check has not been run yet.',
      status: status,
    );
  }

  /// "9x" / "3.9x" / "2.4x" — trailing zeroes dropped, the way the artifacts
  /// and the LIA figures are written.
  static String _multiple(double value) {
    final text = value == value.roundToDouble()
        ? value.toStringAsFixed(0)
        : value.toStringAsFixed(1);
    return '$text×';
  }
}
