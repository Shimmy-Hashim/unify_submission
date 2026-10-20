import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:unify/models/coverage_gap_item.dart';
import 'package:unify/models/home_score_state.dart';
import 'package:unify/models/score_profile.dart';

/// A financialProfile document with a schema 1.1.0 score cached on it.
///
/// Shaped exactly as `data/score_profile.py::cache_payload` writes it — the
/// keys here are that module's keys, and a rename there must fail these.
Map<String, dynamic> _profileData({
  Map<String, dynamic>? insuranceCoverage = _defaultCoverage,
  bool hospitalisation = true,
  bool incomeProtection = true,
  bool? unknownChecks,
  String? unavailableReason,
  double? overallScore = 88.0,
}) {
  return {
    'overall_score': overallScore,
    'checklist_last_updated': Timestamp.fromDate(DateTime.utc(2026, 8, 1)),
    'checklist_flags': {
      'layer1': [
        {
          'check': 'income_protection_coverage',
          'passed': unknownChecks == true ? null : incomeProtection,
          'message': unknownChecks == true
              ? 'Your monthly income has not been set yet, so income '
                  'protection cover cannot be checked.'
              : incomeProtection
                  ? 'You have income protection or disability cover in place.'
                  : 'You have no income protection cover. If illness or injury '
                      'stopped your pay, there is nothing in place to replace it.',
          'scored': true,
        },
        {
          'check': 'hospitalisation_coverage',
          'passed': unknownChecks == true ? null : hospitalisation,
          'message': unknownChecks == true
              ? 'Your monthly income has not been set yet, so hospitalisation '
                  'cover cannot be checked.'
              : hospitalisation
                  ? 'You have hospitalisation cover beyond MediShield Life.'
                  : 'You have no hospitalisation cover beyond MediShield Life. '
                      'A large hospital bill would come out of your own savings.',
          'scored': true,
        },
      ],
      'layer3': const <String>[],
      'layer3_evaluated': true,
      'layer3_reason': null,
    },
    'ml_profile': {
      'schema_version': '1.1.0',
      'computed_at': Timestamp.fromDate(DateTime.utc(2026, 8, 1)),
      'age_band': 2,
      'age_band_range': '30-39',
      'domains': {
        'banking': {'status': 'assessed', 'tier_index': 0, 'tier_label': 'Stable'},
        'investment': {
          'status': 'assessed',
          'tier_index': 0,
          'tier_label': 'On Track',
        },
        'insurance': {
          'status': 'assessed',
          'tier_index': 0,
          'tier_label': 'Covered',
        },
      },
      'layer3_flags': const <String>[],
      'layer3_evaluated': true,
      'overall_score': overallScore,
      'unavailable_reason': unavailableReason,
      'score_breakdown': overallScore == null
          ? null
          : {
              'banking_score': 100.0,
              'investment_score': 100.0,
              'insurance_indicators': [1.0, 1.0, 1.0],
              'insurance_indicators_cleared': 3,
              'insurance_third': 100.0,
              'option_b_score': 100.0,
              'ceiling': 100.0,
              'capped': false,
              'overall_score': overallScore,
            },
      'insurance_coverage': insuranceCoverage,
    },
  };
}

/// Band 2, S$60,000 a year: 2x life cover against the 9x benchmark, no CI.
const _defaultCoverage = {
  'life': {
    'feature': 'life_coverage_ratio',
    'ratio': 2.0,
    'benchmark_ratio': 9.0,
    'meets_benchmark': false,
    'covered_sgd': 120000.0,
    'recommended_sgd': 540000.0,
    'shortfall_sgd': 420000.0,
  },
  'critical_illness': {
    'feature': 'ci_coverage_ratio',
    'ratio': 0.0,
    'benchmark_ratio': 3.9,
    'meets_benchmark': false,
    'covered_sgd': 0.0,
    'recommended_sgd': 234000.0,
    'shortfall_sgd': 234000.0,
  },
};

const _wellCovered = {
  'life': {
    'feature': 'life_coverage_ratio',
    'ratio': 10.0,
    'benchmark_ratio': 9.0,
    'meets_benchmark': true,
    'covered_sgd': 600000.0,
    'recommended_sgd': 540000.0,
    'shortfall_sgd': 0.0,
  },
  'critical_illness': {
    'feature': 'ci_coverage_ratio',
    'ratio': 4.0,
    'benchmark_ratio': 3.9,
    'meets_benchmark': true,
    'covered_sgd': 240000.0,
    'recommended_sgd': 234000.0,
    'shortfall_sgd': 0.0,
  },
};

CoverageGapsView _viewFor(Map<String, dynamic> data) =>
    CoverageGapsView.fromScoreState(_stateFor(data));

ScoreState _stateFor(Map<String, dynamic> data) {
  final profile = ScoreProfile.fromProfileData(data);
  if (profile.unavailableReason != null) {
    return ScoreUnavailable(profile.unavailableReason!, profile);
  }
  if (profile.overallScore == null || profile.breakdown == null) {
    return ScorePartial(profile);
  }
  return ScoreComplete(profile, profile.overallScore!, profile.breakdown!);
}

void main() {
  group('InsuranceCoverage.fromMap', () {
    test('reads the backend block verbatim, deriving nothing', () {
      final profile = ScoreProfile.fromProfileData(_profileData());
      final coverage = profile.insuranceCoverage!;
      expect(coverage.life.feature, 'life_coverage_ratio');
      expect(coverage.life.ratio, 2.0);
      expect(coverage.life.benchmarkRatio, 9.0);
      expect(coverage.life.meetsBenchmark, isFalse);
      expect(coverage.life.shortfallSgd, 420000.0);
      expect(coverage.criticalIllness.benchmarkRatio, 3.9);
      expect(coverage.criticalIllness.hasNoCover, isTrue);
      expect(coverage.life.hasNoCover, isFalse);
    });

    test('a pre-1.1.0 cache with no coverage block parses as null', () {
      final profile =
          ScoreProfile.fromProfileData(_profileData(insuranceCoverage: null));
      expect(profile.insuranceCoverage, isNull);
    });

    test('half a coverage block is treated as none at all', () {
      final profile = ScoreProfile.fromProfileData(
        _profileData(insuranceCoverage: {'life': _defaultCoverage['life']}),
      );
      expect(profile.insuranceCoverage, isNull);
    });
  });

  group('CoverageGapsView — assessed', () {
    test('states the shortfall against the LIA benchmark, in both units', () {
      final view = _viewFor(_profileData());
      expect(view.hasItems, isTrue);
      expect(view.items.length, 4);

      final life = view.items.first;
      expect(life.title, 'Life Cover');
      expect(life.status, CoverageStatus.gap);
      expect(life.pillLabel, 'Gap');
      expect(life.message, contains('2×'));
      expect(life.message, contains('9×'));
      expect(life.message, contains('\$420,000'));
    });

    test('zero cover of a type reads differently from partial cover', () {
      final ci = _viewFor(_profileData()).items[1];
      expect(ci.title, 'Critical Illness');
      expect(ci.message, contains('No cover of this type on file'));
      // The recommended amount, not a "0x of 3.9x" shortfall sentence.
      expect(ci.message, contains('3.9×'));
      expect(ci.message, contains('\$234,000'));
      expect(ci.message, isNot(contains('short')));
    });

    test('the two binary lines use the backend copy verbatim', () {
      final view = _viewFor(_profileData(
        hospitalisation: false,
        incomeProtection: false,
      ));
      final hospitalisation = view.items[2];
      final incomeProtection = view.items[3];
      expect(hospitalisation.title, 'Hospitalisation');
      expect(hospitalisation.status, CoverageStatus.gap);
      expect(hospitalisation.message,
          'You have no hospitalisation cover beyond MediShield Life. '
          'A large hospital bill would come out of your own savings.');
      expect(incomeProtection.status, CoverageStatus.gap);
      expect(incomeProtection.message, startsWith('You have no income '));
    });

    test('a well-covered user gets an honest positive state, not silence', () {
      final view = _viewFor(_profileData(insuranceCoverage: _wellCovered));
      expect(view.gapCount, 0);
      expect(view.items.every((i) => i.status == CoverageStatus.covered), isTrue);
      expect(view.items.every((i) => i.pillLabel == 'Covered'), isTrue);
      expect(view.summary, 'No gaps: your cover meets all 4 benchmarks.');
      expect(view.items.first.message, contains('at or above'));
    });

    test('a mixed user is counted, not graded', () {
      final view = _viewFor(_profileData(hospitalisation: false));
      // life + CI + hospitalisation.
      expect(view.gapCount, 3);
      expect(view.summary, '3 of 4 benchmarks not met.');
    });

    test('an unevaluated check never counts as a gap', () {
      final view = _viewFor(_profileData(unknownChecks: true));
      expect(view.items[2].status, CoverageStatus.notAssessed);
      expect(view.items[2].pillLabel, 'Not assessed');
      expect(view.gapCount, 2); // the two ratio lines only
      expect(view.summary, '2 of 2 assessed benchmarks not met.');
    });

    test('a partial score still assesses cover', () {
      // No overall score (no bank connected), but insurance is fully assessed.
      final view = _viewFor(_profileData(overallScore: null));
      expect(view.hasItems, isTrue);
      expect(view.items.length, 4);
    });
  });

  group('CoverageGapsView — nothing to show', () {
    test('income not set names income, not a generic pending state', () {
      final view = _viewFor(_profileData(
        insuranceCoverage: null,
        unknownChecks: true,
        unavailableReason: 'income_not_set',
        overallScore: null,
      ));
      expect(view.hasItems, isFalse);
      expect(view.notice, contains('monthly income'));
      expect(view.notice, contains('LIA'));
    });

    test('an unsupported age says so rather than blaming the user', () {
      final view = _viewFor(_profileData(
        insuranceCoverage: null,
        unknownChecks: true,
        unavailableReason: 'unsupported_age',
        overallScore: null,
      ));
      expect(view.notice, contains('20 to 70'));
    });

    test('a score that has never run is not a coverage claim', () {
      final view = CoverageGapsView.fromScoreState(const ScoreNeverComputed());
      expect(view.hasItems, isFalse);
      expect(view.notice, contains('not been assessed yet'));
    });

    test('loading and failure are distinct from both of those', () {
      expect(CoverageGapsView.fromScoreState(const ScoreLoading()).notice,
          'Checking your cover…');
      expect(CoverageGapsView.fromScoreState(const ScoreFailed('boom')).notice,
          contains('Could not load'));
    });

    test('a score with no coverage block shows no cards and no zeroes', () {
      final view = _viewFor(_profileData(insuranceCoverage: null));
      expect(view.hasItems, isFalse);
      expect(view.summary, isNull);
      expect(view.notice, isNotNull);
    });
  });

  group('band-specific benchmarks', () {
    test('band 5 renders the 5x life benchmark it was cached with', () {
      final view = _viewFor(_profileData(insuranceCoverage: const {
        'life': {
          'feature': 'life_coverage_ratio',
          'ratio': 6.0,
          'benchmark_ratio': 5.0,
          'meets_benchmark': true,
          'covered_sgd': 300000.0,
          'recommended_sgd': 250000.0,
          'shortfall_sgd': 0.0,
        },
        'critical_illness': {
          'feature': 'ci_coverage_ratio',
          'ratio': 1.0,
          'benchmark_ratio': 3.9,
          'meets_benchmark': false,
          'covered_sgd': 50000.0,
          'recommended_sgd': 195000.0,
          'shortfall_sgd': 145000.0,
        },
      }));
      expect(view.items.first.status, CoverageStatus.covered);
      expect(view.items.first.message, contains('5×'));
      expect(view.items.first.message, isNot(contains('9×')));
    });
  });
}
