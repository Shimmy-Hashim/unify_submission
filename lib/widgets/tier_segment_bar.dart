import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../utils/app_styles.dart';

/// The two-segment tier bar from the redesigned Home screen
/// (Figma 134:283 "Full Assessment" / 143:379 "Partial Assessment").
///
/// WHY TWO FIXED HALVES AND NOT A PROGRESS FILL
/// Every domain in the score ships as exactly two tiers — Banking is "Stable"
/// or "Building", Investment "On Track" or "Getting Started", Insurance
/// "Covered" or "Not Covered". There is no continuous value inside a tier
/// (three validation rounds established there is no defensible one), so a
/// partially-filled bar would invent precision the model does not have. The
/// bar therefore lights the half the user is in and leaves the other neutral.
///
/// One widget covers all three states because the colour rules are a single
/// table, and splitting them per-state is how the "assessed" and "not
/// assessed" treatments drift apart.
enum TierSegment {
  /// Healthiest tier (artifact tier_index 0) — right half green.
  healthy,

  /// Weaker tier (tier_index 1) — left half red.
  weak,

  /// Domain has no tier yet. Both halves neutral; the caller dims the row.
  notAssessed,
}

class TierSegmentBar extends StatelessWidget {
  const TierSegmentBar({
    super.key,
    required this.segment,
    required this.lowLabel,
    required this.highLabel,
  });

  final TierSegment segment;

  /// The weaker tier's label, under the left half (e.g. "Building").
  final String lowLabel;

  /// The healthier tier's label, under the right half (e.g. "Stable").
  final String highLabel;

  // Figma 161:432-447. Exact values, not approximations. The two tier fills
  // are saturated enough to hold up on either card colour and stay as they
  // are; only the neutral half and the sub-label inks are theme-resolved.
  static const _green = Color(0xFF16A34A);
  static const _red = Color(0xFFEF4444);

  /// Sub-label colours: the label under the *active* half takes that half's
  /// dark tone, the inactive one stays grey. The two dark tones are only
  /// legible against white — on a #1E1E1E card #731414 is near-invisible — so
  /// dark mode takes the light tint of the same hue instead.
  static const _greenText = Color(0xFF055926);
  static const _redText = Color(0xFF731414);
  static const _greenTextDark = Color(0xFF6EE7A0);
  static const _redTextDark = Color(0xFFFF8A80);
  static const _mutedText = Color(0xFF8C8C8C);

  static const _barHeight = 15.0;
  static const _barRadius = 6.0;
  static const _gap = 2.0;

  /// The tier-value colour for a row in this state — exported so the row
  /// header and the bar can never disagree about whether a tier reads as a
  /// problem. Figma uses #731414 for "Not Covered" and #1A1A1A otherwise.
  ///
  /// Takes a [Brightness] rather than a BuildContext so it stays callable
  /// from a plain unit test, which is where the three-states-are-distinct
  /// guarantee is actually checked.
  static Color valueColor(TierSegment segment, Brightness brightness) {
    final dark = brightness == Brightness.dark;
    return switch (segment) {
      TierSegment.healthy =>
        dark ? const Color(0xFFF0F0F0) : const Color(0xFF1A1A1A),
      TierSegment.weak => dark ? _redTextDark : _redText,
      TierSegment.notAssessed => _mutedText,
    };
  }

  /// The middle band's hue. Only the overall score has a middle band — every
  /// domain ships two tiers — so the bar itself never paints it.
  static const _amber = Color(0xFFF59E0B);

  /// The overall score's colour, by its band: 0-39 red, 40-69 amber, 70-100
  /// green. The boundaries are ml/overall_score.py's ceiling ladder
  /// ("39.0 = the top of At Risk", "69.0 = the top of Building") — the only
  /// score banding this project defines — and the red and green are this bar's
  /// own fills, so the score and the domain rows beside it can never disagree
  /// about what a healthy or weak colour looks like. (Moved here from the
  /// removed ScoreGauge.bandColor, which used the same three values.)
  ///
  /// Meant for the large score numeral, which clears contrast on both card
  /// colours at its size; the saturated fills are used in both themes.
  static Color scoreColor(double score) {
    if (score < 40) return _red;
    if (score < 70) return _amber;
    return _green;
  }

  /// Ink for small up/down text beside a score (the trend line). Light mode
  /// uses the fills, as the Figma trend does; dark mode takes the same light
  /// tints the sub-labels use, since 11px saturated green is not legible on
  /// the #1E1E1E card.
  static Color changeColor({required bool up, required Brightness brightness}) {
    final dark = brightness == Brightness.dark;
    if (up) return dark ? _greenTextDark : _green;
    return dark ? _redTextDark : _red;
  }

  Color _leftFill(BuildContext context) => segment == TierSegment.weak
      ? _red
      : AppTheme.neutralSegment(context);

  Color _rightFill(BuildContext context) => segment == TierSegment.healthy
      ? _green
      : AppTheme.neutralSegment(context);

  Color _leftLabelColor(BuildContext context) => segment == TierSegment.weak
      ? (AppTheme.isDark(context) ? _redTextDark : _redText)
      : _mutedText;

  Color _rightLabelColor(BuildContext context) =>
      segment == TierSegment.healthy
          ? (AppTheme.isDark(context) ? _greenTextDark : _greenText)
          : _mutedText;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            Expanded(child: _segmentBox(_leftFill(context))),
            const SizedBox(width: _gap),
            Expanded(child: _segmentBox(_rightFill(context))),
          ],
        ),
        const SizedBox(height: 3),
        Row(
          children: [
            Expanded(
              child: Text(
                lowLabel,
                style: _labelStyle(_leftLabelColor(context)),
              ),
            ),
            const SizedBox(width: _gap),
            Expanded(
              child: Text(
                highLabel,
                textAlign: TextAlign.right,
                style: _labelStyle(_rightLabelColor(context)),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _segmentBox(Color color) => Container(
        height: _barHeight,
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(_barRadius),
        ),
      );

  /// Inter Medium 10 (Figma 161:434).
  static TextStyle _labelStyle(Color color) => GoogleFonts.inter(
        fontWeight: FontWeight.w500,
        fontSize: 10,
        height: 1.2,
        color: color,
      );
}
