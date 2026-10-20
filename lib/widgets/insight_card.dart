import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../utils/app_styles.dart';

/// Insights — the Layer 3 cross-domain flag cards on Home (Figma 134:397).
///
/// Layer 3 fires only when all three domain tiers exist, so a card here always
/// means "the model compared your three areas and found this pattern". That is
/// why the two empty states below are worded differently: one is a result, the
/// other is a gap.

enum InsightTone { risk, caution, positive }

class InsightCopy {
  const InsightCopy({
    required this.emoji,
    required this.title,
    required this.subtitle,
    required this.pill,
    required this.tone,
  });

  final String emoji;
  final String title;
  final String subtitle;

  /// Severity chip. The Figma card design carries one on every card and spells
  /// the "Uninsured Wealth" case out exactly (red "Action Needed"); the amber
  /// and green variants come from its other two cards.
  final String pill;
  final InsightTone tone;
}

/// The four locked Layer 3 rules, keyed by `ml/layer3.py`'s flag names.
///
/// Keys must match those constants exactly — they arrive as raw strings in
/// `ml_profile.layer3_flags`, so a typo here silently drops a card rather than
/// failing loudly. [insightForFlag] is the only reader.
const kInsightsByFlag = <String, InsightCopy>{
  'Wealth Without Protection': InsightCopy(
    emoji: '🛡️',
    title: 'Uninsured Wealth',
    subtitle: 'Strong finances, no coverage',
    pill: 'Action Needed',
    tone: InsightTone.risk,
  ),
  'Protection First': InsightCopy(
    emoji: '🛡️',
    title: 'Protection First',
    subtitle: 'Well covered, banking still building',
    pill: 'On Track',
    tone: InsightTone.positive,
  ),
  'Strong Foundation, Room to Grow': InsightCopy(
    emoji: '📈',
    title: 'Room to Grow',
    subtitle: 'Saving well, not investing yet',
    pill: 'Medium Risk',
    tone: InsightTone.caution,
  ),
  'Aggressive Investing, Weak Foundation': InsightCopy(
    emoji: '💧',
    title: 'Shaky Foundation',
    subtitle: 'Investing fast, banking needs work',
    pill: 'Action Needed',
    tone: InsightTone.risk,
  ),
};

/// Null for a flag this build does not know — the card is skipped rather than
/// rendered blank.
InsightCopy? insightForFlag(String flag) => kInsightsByFlag[flag];

/// The two empty states, which must not read the same.
///
/// [allAssessed] false — Layer 3 never ran, because it needs all three tiers.
/// We genuinely cannot say whether there is a cross-domain problem. Wording is
/// the Partial Assessment frame's own (Figma 143:379).
///
/// [allAssessed] true — Layer 3 ran against all three tiers and no rule fired.
/// That is a finding, not a gap, and must not be dressed up as "not enough
/// data yet".
String emptyInsightsMessage({required bool allAssessed}) => allAssessed
    ? 'Nothing to flag: no mismatches between your banking, investing '
        'and cover.'
    : 'No insights yet. Check back once all three areas are assessed.';

class InsightCard extends StatelessWidget {
  const InsightCard({super.key, required this.insight});

  final InsightCopy insight;

  static const _pillColors = {
    InsightTone.risk: (bg: Color(0xFFFFEBEE), fg: Color(0xFFC62828)),
    InsightTone.caution: (bg: Color(0xFFFFF3E0), fg: Color(0xFFE65100)),
    InsightTone.positive: (bg: Color(0xFFE8F5E9), fg: Color(0xFF2E7D32)),
  };

  @override
  Widget build(BuildContext context) {
    final base = _pillColors[insight.tone]!;
    final pill = AppTheme.chip(
      context,
      background: base.bg,
      foreground: base.fg,
    );
    return Container(
      width: 200,
      padding: const EdgeInsets.fromLTRB(15, 8, 12, 10),
      decoration: BoxDecoration(
        color: AppTheme.surface(context),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [AppTheme.cardShadow(context)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(insight.emoji, style: const TextStyle(fontSize: 13)),
          Text(
            insight.title,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w600,
              fontSize: 11,
              color: AppTheme.primaryText(context),
            ),
          ),
          const SizedBox(height: 4),
          Expanded(
            child: Text(
              insight.subtitle,
              style: GoogleFonts.inter(
                fontWeight: FontWeight.w600,
                fontSize: 10,
                height: 1.35,
                color: AppTheme.subtleBody(context),
              ),
            ),
          ),
          // Align, because a Container with a non-null `alignment` expands to
          // fill its parent — the pill must hug its label (Figma w-71).
          Align(
            alignment: Alignment.centerLeft,
            child: Container(
              height: 18,
              constraints: const BoxConstraints(minWidth: 71),
              padding: const EdgeInsets.symmetric(horizontal: 10),
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: pill.background,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                insight.pill,
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w600,
                  fontSize: 8,
                  color: pill.foreground,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
