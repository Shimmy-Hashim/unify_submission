import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:unify/widgets/insight_card.dart';
import 'package:unify/widgets/tier_segment_bar.dart';

void main() {
  group('Layer 3 flag -> insight copy', () {
    // The keys are ml/layer3.py's four locked flag names, verbatim. They
    // arrive as raw strings in ml_profile.layer3_flags, so a drifted key
    // silently drops a card instead of failing — this test is the guard.
    test('every locked flag name maps to a card', () {
      const lockedFlags = [
        'Wealth Without Protection',
        'Protection First',
        'Strong Foundation, Room to Grow',
        'Aggressive Investing, Weak Foundation',
      ];
      for (final flag in lockedFlags) {
        expect(insightForFlag(flag), isNotNull, reason: 'missing copy: $flag');
      }
      expect(kInsightsByFlag, hasLength(lockedFlags.length));
    });

    test('carries the locked title and subtitle copy', () {
      final wealth = insightForFlag('Wealth Without Protection')!;
      expect(wealth.title, 'Uninsured Wealth');
      expect(wealth.subtitle, 'Strong finances, no coverage');

      final protection = insightForFlag('Protection First')!;
      expect(protection.title, 'Protection First');
      expect(protection.subtitle, 'Well covered, banking still building');

      final grow = insightForFlag('Strong Foundation, Room to Grow')!;
      expect(grow.title, 'Room to Grow');
      expect(grow.subtitle, 'Saving well, not investing yet');

      final shaky = insightForFlag('Aggressive Investing, Weak Foundation')!;
      expect(shaky.title, 'Shaky Foundation');
      expect(shaky.subtitle, 'Investing fast, banking needs work');
    });

    test('an unknown flag is skipped, not rendered blank', () {
      expect(insightForFlag('Some Future Rule'), isNull);
    });

    test('the two coverage-gap patterns read as action, not as neutral', () {
      expect(insightForFlag('Wealth Without Protection')!.tone,
          InsightTone.risk);
      expect(insightForFlag('Aggressive Investing, Weak Foundation')!.tone,
          InsightTone.risk);
    });
  });

  group('empty insights', () {
    // The state that cannot be produced from any test account on hand: all
    // three domains assessed AND no Layer 3 rule firing. Covered here instead.
    test('"nothing fired" and "could not tell" are different messages', () {
      final evaluated = emptyInsightsMessage(allAssessed: true);
      final notEvaluated = emptyInsightsMessage(allAssessed: false);
      expect(evaluated, isNot(equals(notEvaluated)));
    });

    test('the not-evaluated message says why, without claiming a result', () {
      final message = emptyInsightsMessage(allAssessed: false);
      expect(message, contains('once all three areas are assessed'));
    });

    test('the evaluated message states a finding, not a gap', () {
      final message = emptyInsightsMessage(allAssessed: true);
      expect(message, contains('Nothing to flag'));
      // Must not imply data is still missing — that is the other state.
      expect(message.toLowerCase(), isNot(contains('yet')));
    });
  });

  group('TierSegmentBar', () {
    test('value colour distinguishes weak from healthy and unassessed', () {
      final healthy =
          TierSegmentBar.valueColor(TierSegment.healthy, Brightness.light);
      final weak =
          TierSegmentBar.valueColor(TierSegment.weak, Brightness.light);
      final none =
          TierSegmentBar.valueColor(TierSegment.notAssessed, Brightness.light);
      expect({healthy, weak, none}, hasLength(3));
      // Figma 161:443 — a weak tier reads in the dark red, not in body black.
      expect(weak, const Color(0xFF731414));
    });

    // The three states have to stay tellable apart in dark mode too, which is
    // not free: the light palette gets there with a near-black and a very dark
    // red, and both of those collapse into the card on a dark background.
    test('the three states stay distinct in dark mode', () {
      final healthy =
          TierSegmentBar.valueColor(TierSegment.healthy, Brightness.dark);
      final weak = TierSegmentBar.valueColor(TierSegment.weak, Brightness.dark);
      final none =
          TierSegmentBar.valueColor(TierSegment.notAssessed, Brightness.dark);
      expect({healthy, weak, none}, hasLength(3));
      // A weak tier must still read as a problem, not as body text.
      expect(weak, isNot(equals(healthy)));
      // Both assessed states have to clear the #1E1E1E card they sit on.
      for (final color in [healthy, weak]) {
        expect(color.computeLuminance(), greaterThan(0.25));
      }
    });
  });
}
