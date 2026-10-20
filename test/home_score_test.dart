import 'dart:async';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:unify/models/home_score_state.dart';
import 'package:unify/models/score_profile.dart';
import 'package:unify/services/health_score_service.dart';
import 'package:unify/services/home_score_controller.dart';

/// A HealthScoreService whose stream and refresh call are driven by the test.
class _FakeService extends HealthScoreService {
  _FakeService() : super.forTesting();

  final controller = StreamController<ScoreState>.broadcast();
  int refreshCalls = 0;
  Object? refreshError;

  @override
  Stream<ScoreState> watch(String uid) => controller.stream;

  @override
  Future<void> requestRefresh({Duration timeout = const Duration(seconds: 45)}) async {
    refreshCalls++;
    final error = refreshError;
    if (error != null) throw error;
  }
}

/// A cached-score document as `cache_payload()` writes it. Field spelling and
/// nesting match a real users/{uid}/financialProfile/profile document.
Map<String, dynamic> _profileDoc({
  double? overallScore,
  String? unavailableReason,
  Map<String, dynamic>? domains,
  Map<String, dynamic>? breakdown,
  DateTime? computedAt,
}) {
  final at = Timestamp.fromDate(computedAt ?? DateTime.utc(2026, 8, 1, 3, 44));
  return {
    'monthly_income_sgd': 4000.0,
    'overall_score': overallScore,
    'checklist_last_updated': at,
    'checklist_flags': {
      'layer1': [
        {
          'check': 'income_protection_coverage',
          'passed': false,
          'message': 'You have no income protection cover.',
          'scored': true,
        },
        {
          'check': 'hospitalisation_coverage',
          'passed': null,
          'message': 'Your monthly income has not been set yet.',
          'scored': true,
        },
      ],
      'layer3': const <String>[],
      'layer3_evaluated': false,
      'layer3_reason': 'incomplete_domains',
    },
    'ml_profile': {
      'schema_version': '1.0.0',
      'computed_at': at,
      'age_band': 2,
      'age_band_range': '30-39',
      'unavailable_reason': unavailableReason,
      'overall_score': overallScore,
      'overall_score_reason':
          overallScore == null ? 'incomplete_domains' : null,
      'layer3_flags': const <String>[],
      'layer3_evaluated': false,
      'score_breakdown': breakdown,
      'artifacts': {
        'schema_version': '1.0.0',
        'generated_utc': '2026-07-26T13:56:49.672664+00:00',
      },
      'domains': domains ??
          {
            'banking': {
              'status': 'not_assessed',
              'tier_index': null,
              'tier_label': null,
              'reason': 'missing_features',
              'missing_features': ['savings_rate', 'dti'],
            },
            'investment': {
              'status': 'not_assessed',
              'tier_index': null,
              'tier_label': null,
              'reason': 'missing_features',
              'missing_features': ['investment_rate'],
            },
            'insurance': {
              'status': 'assessed',
              'tier_index': 1,
              'tier_label': 'Not Covered',
              'reason': null,
              'missing_features': const <String>[],
            },
          },
    },
  };
}

const _fullDomains = {
  'banking': {
    'status': 'assessed',
    'tier_index': 1,
    'tier_label': 'Building',
    'reason': null,
    'missing_features': <String>[],
  },
  'investment': {
    'status': 'assessed',
    'tier_index': 0,
    'tier_label': 'On Track',
    'reason': null,
    'missing_features': <String>[],
  },
  'insurance': {
    'status': 'assessed',
    'tier_index': 1,
    'tier_label': 'Not Covered',
    'reason': null,
    'missing_features': <String>[],
  },
};

const _fullBreakdown = {
  'banking_score': 0.0,
  'investment_score': 100.0,
  'insurance_indicators': [0.0, 0.0, 0.0],
  'insurance_indicators_cleared': 0,
  'insurance_third': 0.0,
  'option_b_score': 33.333333333333336,
  'ceiling': 39.0,
  'capped': false,
  'overall_score': 33.333333333333336,
};

void main() {
  group('stateFromProfileData', () {
    test('no cached run -> ScoreNeverComputed', () {
      final state = HealthScoreService.stateFromProfileData({
        'monthly_income_sgd': 4000.0,
        'checklist_last_updated': null,
        'ml_profile': null,
      });
      expect(state, isA<ScoreNeverComputed>());
    });

    test('a timestamp without an ml_profile is still never computed', () {
      final state = HealthScoreService.stateFromProfileData({
        'checklist_last_updated': Timestamp.now(),
        'ml_profile': null,
      });
      expect(state, isA<ScoreNeverComputed>());
    });

    test('unavailable_reason -> ScoreUnavailable with the parsed reason', () {
      for (final entry in {
        'income_not_set': ScoreUnavailableReason.incomeNotSet,
        'unsupported_age': ScoreUnavailableReason.unsupportedAge,
        'date_of_birth_not_set': ScoreUnavailableReason.dateOfBirthNotSet,
      }.entries) {
        final state = HealthScoreService.stateFromProfileData(
            _profileDoc(unavailableReason: entry.key));
        expect(state, isA<ScoreUnavailable>());
        expect((state as ScoreUnavailable).reason, entry.value);
        // The checklist is still parsed — an unavailable score still has copy
        // to show, with passed=null where the check could not be evaluated.
        expect(state.profile!.checklist, hasLength(2));
      }
    });

    test('an unrecognised reason code is unavailable, not a crash', () {
      final state = HealthScoreService.stateFromProfileData(
          _profileDoc(unavailableReason: 'some_future_reason'));
      expect((state as ScoreUnavailable).reason,
          ScoreUnavailableReason.unknown);
    });

    test('null overall score -> ScorePartial, with missing features', () {
      final state = HealthScoreService.stateFromProfileData(_profileDoc());
      expect(state, isA<ScorePartial>());
      final partial = state as ScorePartial;
      expect(partial.assessedDomains.map((d) => d.domain),
          [ScoreDomain.insurance]);
      expect(partial.unassessedDomains.map((d) => d.domain),
          [ScoreDomain.banking, ScoreDomain.investment]);
      final banking = partial.profile.domains[ScoreDomain.banking]!;
      expect(banking.reason, ScoreReason.missingFeatures);
      // Artifact spelling in, readable labels out.
      expect(banking.missingFeatureLabels,
          ['Savings rate', 'Debt-to-income ratio']);
      expect(partial.profile.overallScoreReason,
          ScoreReason.incompleteDomains);
      expect(partial.profile.layer3Evaluated, isFalse);
    });

    test('score + breakdown -> ScoreComplete', () {
      final state = HealthScoreService.stateFromProfileData(_profileDoc(
        overallScore: 33.333333333333336,
        domains: _fullDomains,
        breakdown: _fullBreakdown,
      ));
      expect(state, isA<ScoreComplete>());
      final complete = state as ScoreComplete;
      expect(complete.overallScore, closeTo(33.33, 0.01));
      expect(complete.breakdown.ceiling, 39.0);
      expect(complete.breakdown.capped, isFalse);
      expect(complete.breakdown.insuranceIndicatorsCleared, 0);
      expect(complete.domains.map((d) => d.tierLabel),
          ['Building', 'On Track', 'Not Covered']);
      expect(complete.profile.ageBand, 2);
      expect(complete.profile.artifacts!.schemaVersion, '1.0.0');
    });

    test('domains always cover all three, even if the cache omits one', () {
      final state = HealthScoreService.stateFromProfileData(
          _profileDoc(unavailableReason: 'income_not_set', domains: const {}));
      final profile = (state as ScoreUnavailable).profile!;
      expect(profile.orderedDomains, hasLength(3));
      // A whole-pipeline failure stamps the same reason onto every domain.
      expect(profile.orderedDomains.map((d) => d.reason),
          everyElement(ScoreReason.incomeNotSet));
    });
  });

  group('healthyDebtRatio', () {
    Map<String, dynamic> withRatio(Map<String, dynamic> doc, Object? ratio) {
      (doc['ml_profile'] as Map)['healthy_debt_ratio'] = ratio;
      return doc;
    }

    ScoreProfile? profileOf(ScoreState state) => switch (state) {
          ScorePartial(:final profile) => profile,
          ScoreComplete(:final profile) => profile,
          ScoreUnavailable(:final profile) => profile,
          ScoreLoading() || ScoreNeverComputed() || ScoreFailed() => null,
        };

    test('parsed from ml_profile on every state that carries a profile', () {
      final complete = HealthScoreService.stateFromProfileData(withRatio(
        _profileDoc(
          overallScore: 33.333333333333336,
          domains: _fullDomains,
          breakdown: _fullBreakdown,
        ),
        0.7134,
      ));
      expect((complete as ScoreComplete).profile.healthyDebtRatio, 0.7134);

      final partial = HealthScoreService.stateFromProfileData(
          withRatio(_profileDoc(), 0.25));
      expect((partial as ScorePartial).profile.healthyDebtRatio, 0.25);

      final unavailable = HealthScoreService.stateFromProfileData(withRatio(
          _profileDoc(unavailableReason: 'income_not_set'), 0.5));
      expect((unavailable as ScoreUnavailable).profile!.healthyDebtRatio, 0.5);
    });

    test('0.0 is a real value (all unhealthy debt), not absent', () {
      final state = HealthScoreService.stateFromProfileData(
          withRatio(_profileDoc(), 0.0));
      expect((state as ScorePartial).profile.healthyDebtRatio, 0.0);
      final whole = HealthScoreService.stateFromProfileData(
          withRatio(_profileDoc(), 1));
      expect((whole as ScorePartial).profile.healthyDebtRatio, 1.0);
    });

    test('null when not computable, and on a pre-1.2.0 cache without the key',
        () {
      final none = HealthScoreService.stateFromProfileData(
          withRatio(_profileDoc(), null));
      expect((none as ScorePartial).profile.healthyDebtRatio, isNull);

      final legacy = HealthScoreService.stateFromProfileData(_profileDoc());
      expect((legacy as ScorePartial).profile.healthyDebtRatio, isNull);
    });

    test('states with no profile expose no ratio', () {
      const states = <ScoreState>[
        ScoreLoading(),
        ScoreNeverComputed(),
        ScoreFailed('Could not load your score.'),
      ];
      for (final state in states) {
        expect(profileOf(state)?.healthyDebtRatio, isNull);
      }
    });
  });

  group('HomeScoreController', () {
    test('starts in ScoreLoading, before any snapshot arrives', () {
      final service = _FakeService();
      final controller = HomeScoreController(uid: 'u1', service: service);
      addTearDown(controller.dispose);

      expect(controller.state.score, isA<ScoreLoading>());
      expect(controller.state.refreshing, isFalse);
    });

    test('refresh stays true until the stream echoes a NEW computation',
        () async {
      final service = _FakeService();
      final controller = HomeScoreController(uid: 'u1', service: service);
      addTearDown(controller.dispose);

      final first = HealthScoreService.stateFromProfileData(_profileDoc());
      service.controller.add(first);
      await Future<void>.delayed(Duration.zero);
      expect(controller.state.score, isA<ScorePartial>());

      final refreshFuture = controller.refresh();
      await Future<void>.delayed(Duration.zero);
      expect(service.refreshCalls, 1);
      expect(controller.state.refreshing, isTrue);
      // Existing data is still on screen — refresh never blanks it out.
      expect(controller.state.score, isA<ScorePartial>());

      // A repeat of the SAME computation must not end the refresh.
      service.controller.add(first);
      await Future<void>.delayed(Duration.zero);
      expect(controller.state.refreshing, isTrue);

      service.controller.add(HealthScoreService.stateFromProfileData(
          _profileDoc(
              overallScore: 33.3,
              domains: _fullDomains,
              breakdown: _fullBreakdown,
              computedAt: DateTime.utc(2026, 8, 1, 4))));
      await refreshFuture;

      expect(controller.state.refreshing, isFalse);
      expect(controller.state.refreshError, isNull);
      expect(controller.state.score, isA<ScoreComplete>());
    });

    test('a failed refresh sets refreshError and leaves the score alone',
        () async {
      final service = _FakeService()
        ..refreshError = HealthScoreException('Server unreachable');
      final controller = HomeScoreController(uid: 'u1', service: service);
      addTearDown(controller.dispose);

      service.controller.add(HealthScoreService.stateFromProfileData(
          _profileDoc(
              overallScore: 33.3,
              domains: _fullDomains,
              breakdown: _fullBreakdown)));
      await Future<void>.delayed(Duration.zero);

      await controller.refresh();

      expect(controller.state.refreshError, 'Server unreachable');
      expect(controller.state.refreshing, isFalse);
      // A failed recompute is NOT a failed screen.
      expect(controller.state.score, isA<ScoreComplete>());

      controller.acknowledgeRefreshError();
      expect(controller.state.refreshError, isNull);
    });

    test('a stream error becomes ScoreFailed', () async {
      final service = _FakeService();
      final controller = HomeScoreController(uid: 'u1', service: service);
      addTearDown(controller.dispose);

      service.controller.addError(StateError('permission denied'));
      await Future<void>.delayed(Duration.zero);

      expect(controller.state.score, isA<ScoreFailed>());
      expect((controller.state.score as ScoreFailed).message, isNotEmpty);
    });
  });
}
