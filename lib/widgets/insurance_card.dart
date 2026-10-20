import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../models/insurance_policy.dart';
import '../utils/app_styles.dart';
import '../utils/formatters.dart';

// ---- Audit palette, scoped to this widget ------------------------------
// These shadow shared AppTheme tokens rather than editing them, the same
// containment Settings, Savings and Investments used: the shared values are
// still read by Home, and Home has its own audit task.

/// Secondary text — the insurer/type subtitle, "Sum Assured", the stat labels,
/// the filename row, and the exclusions detail line.
///
/// Light is the WCAG AA fix: [AppTheme.muted] (#AAAAAA) measures about 2.3:1
/// on white, against the 4.5:1 minimum. #6B6B6B clears it at 5.3:1, and is the
/// value the audit frame uses for every one of these labels.
///
/// Dark deliberately does not follow it down — #AAAAAA already clears 6.9:1 on
/// the #1E1E1E card, and #6B6B6B there would fall to roughly 2.9:1, breaking
/// dark mode to fix light. The old flat constant could not express that, which
/// is why it had to become a pair.
Color _muted(BuildContext context) =>
    AppTheme.isDark(context) ? AppTheme.muted : const Color(0xFF6B6B6B);

/// Primary ink. [AppTheme.primaryText]'s light value is pure `Colors.black`;
/// the brand value is #202020, sampled from the audit frame.
Color _primaryText(BuildContext context) =>
    AppTheme.isDark(context) ? Colors.white : AppColors.brandBlack;

/// The one neutral chip for the coverage-type tag.
///
/// The tag used to take one of six arbitrary colour pairs keyed off
/// policy_type (blue for life, purple for critical illness, teal for
/// hospitalisation...). Those hues encoded nothing a reader could act on —
/// "Life" is a descriptive label, not a risk level — while sitting in the same
/// pill shape as the app's genuine red/amber/green status chips, so the card
/// spent its only colour vocabulary on the one thing that carries no signal.
/// One neutral pair instead, the same #F2F3F5/#555555 the Investments strip
/// already uses for its neutral badge. The exclusions box and the renewal
/// strip keep their reds and ambers: those do mean something.
const _kTagBg = Color(0xFFF2F3F5);
const _kTagFg = Color(0xFF555555);

/// The card's one alert colour, used by both the imminent-renewal date and the
/// exclusions box. #E53935 only clears about 4:1 on the dark card, so dark mode
/// takes a light tint of the same red. The lift moved to [AppTheme.alertRed]
/// once Delete Account needed the same one; this keeps the card's light red in
/// one place rather than repeating the literal at all three call sites.
Color _alertRed(BuildContext context) =>
    AppTheme.alertRed(context, const Color(0xFFE53935));

/// Coverage card for one parsed policy — Figma node 84:548 ("PRU MAN" card).
///
/// The mockup shows granular per-rider benefit badges ("Total Disability",
/// "Retrenchment Benefit" etc.); the parser schema doesn't extract riders as
/// a list, so the single policy_type badge is shown instead — which is why
/// this card has one tag and never a wrapping row of them. Exclusions stay in
/// their own box rather than doubling as badges — rendering exclusions in the
/// benefit-badge row would misread as coverage.
///
/// REMOVED BY THE UI/UX AUDIT (frames dated August 2026, see [AppColors])
/// The coloured icon badge at the top left (a 34x28 tinted tile carrying one
/// of six policy-type emoji), the "Plan Type" stat column, and the green
/// "Parsed" pill. Each is documented at its own site below. The header text
/// now starts at the card's own padding, the same treatment the bank icons got
/// on Savings and the holding swatches got on Investments.
class InsuranceCard extends StatelessWidget {
  const InsuranceCard({
    super.key,
    required this.policy,
    this.onDelete,
    this.deleting = false,
    this.deleteDisabled = false,
  });

  final InsurancePolicy policy;

  /// Shows the delete control at the right of the footer row when non-null.
  /// The confirmation dialog and the backend call belong to the caller.
  final VoidCallback? onDelete;

  /// This card's own delete is in flight: the control becomes a spinner.
  final bool deleting;

  /// Another card's delete is in flight: the control stays visible but can't
  /// be tapped. Settings' disconnect rows do the same, for the same reason — a
  /// control that vanishes mid-interaction is worse than one that is visibly
  /// unavailable.
  final bool deleteDisabled;

  /// "AIA · Whole Life Participating Plan with CI Rider", or just "AIA".
  ///
  /// The type half is dropped whenever it would only restate the tag sitting
  /// directly beneath it. `policy_type_raw` is the parser's verbatim read of
  /// what the policy calls itself and usually says more than the enum bucket
  /// does, so it earns its place; the bare `policyTypeLabel` fallback did not —
  /// on a personal-accident policy it printed "Personal Accident" here and
  /// "Personal Accident" again in the tag one line down. A raw type that
  /// happens to match the label exactly is caught the same way.
  String get _subtitle {
    final raw = policy.policyTypeRaw?.trim();
    if (raw == null ||
        raw.isEmpty ||
        raw.toLowerCase() == policy.policyTypeLabel.toLowerCase()) {
      return policy.insurer;
    }
    return '${policy.insurer} · $raw';
  }

  @override
  Widget build(BuildContext context) {
    final renewal = policy.renewalDate;
    final now = DateTime.now();
    final renewingSoon = renewal != null &&
        renewal.isBefore(DateTime(now.year + 1, now.month, now.day));

    return Container(
      decoration: BoxDecoration(
        color: AppTheme.surface(context),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [AppTheme.cardShadow(context)],
      ),
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildHeader(context),
          if (renewal != null && renewingSoon) ...[
            const SizedBox(height: 12),
            _buildRenewalStrip(context, renewal, now),
          ],
          const SizedBox(height: 12),
          _TypeBadge(policy: policy),
          const SizedBox(height: 14),
          Divider(height: 1, thickness: 1, color: AppTheme.hairline(context)),
          const SizedBox(height: 10),
          _buildStatsRow(context, renewingSoon),
          if (policy.exclusions.isNotEmpty) ...[
            const SizedBox(height: 12),
            _buildExclusionsBox(context),
          ],
          const SizedBox(height: 12),
          Divider(height: 1, thickness: 1, color: AppTheme.hairline(context)),
          const SizedBox(height: 10),
          _buildFooter(context),
        ],
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                policy.policyName,
                style: GoogleFonts.spaceGrotesk(
                  fontWeight: FontWeight.w700,
                  fontSize: 15,
                  color: _primaryText(context),
                ),
              ),
              Text(
                _subtitle,
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w500,
                  fontSize: 11,
                  color: _muted(context),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(width: 8),
        Column(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(
              // Null is the parser's fabrication-avoidance rule for
              // personal_accident/other — expected, not an error.
              policy.sumAssuredSgd == null
                  ? '-'
                  : formatSgd(policy.sumAssuredSgd!),
              style: GoogleFonts.spaceGrotesk(
                fontWeight: FontWeight.w700,
                fontSize: 17,
                color: _primaryText(context),
              ),
            ),
            Text(
              'Sum Assured',
              style: GoogleFonts.inter(
                fontWeight: FontWeight.w500,
                fontSize: 10,
                color: _muted(context),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildRenewalStrip(BuildContext context, DateTime renewal,
      DateTime now) {
    final expired = renewal.isBefore(now);
    final months =
        (renewal.year - now.year) * 12 + renewal.month - now.month;
    // The hourglass emoji that prefixed all three of these is gone; the
    // wording and the amber chip carry the warning on their own.
    final label = expired
        ? 'Renewal was ${formatShortDate(renewal)}. Check your policy'
        : months < 1
            ? 'Renews ${formatShortDate(renewal)} · less than a month left'
            : 'Renews ${formatShortDate(renewal)} · '
                '$months month${months == 1 ? '' : 's'} left';
    final chip = AppTheme.chip(
      context,
      background: const Color(0xFFFFF3E0),
      foreground: const Color(0xFFE65100),
    );
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: chip.background,
        borderRadius: BorderRadius.circular(12),
      ),
      alignment: Alignment.center,
      child: Text(
        label,
        textAlign: TextAlign.center,
        style: GoogleFonts.inter(
          fontWeight: FontWeight.w600,
          fontSize: 10,
          color: chip.foreground,
        ),
      ),
    );
  }

  /// Two columns, not three. The third used to be "Plan Type", whose value was
  /// `policyTypeLabel` — the exact string already rendered by the tag above it,
  /// and (whenever the parser returned no distinct `policy_type_raw`) by the
  /// subtitle as well, so one policy type was printed three times on one card.
  /// The renewal column is relabelled from "Renews" to "Renewal Date" so every
  /// card names the same thing in the same words.
  ///
  /// A policy whose PDF carried no renewal date still shows "—" here rather
  /// than a fabricated one, per the parser's own no-fabrication rule.
  Widget _buildStatsRow(BuildContext context, bool renewingSoon) {
    final renewal = policy.renewalDate;
    return IntrinsicHeight(
      child: Row(
        children: [
          _Stat(
            label: 'Annual Premium',
            value: policy.annualPremiumSgd == null
                ? '-'
                : '${formatSgd(policy.annualPremiumSgd!)}/yr',
          ),
          VerticalDivider(
              width: 1, thickness: 1, color: AppTheme.hairline(context)),
          _Stat(
            label: 'Renewal Date',
            value: renewal == null ? '-' : formatShortDate(renewal),
            valueColor:
                renewingSoon ? _alertRed(context) : _primaryText(context),
          ),
        ],
      ),
    );
  }

  Widget _buildExclusionsBox(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        // A blush-white wash on white; the dark equivalent is a red-tinted
        // step *up* from the #1E1E1E card, not a lighter white.
        color: AppTheme.isDark(context)
            ? _alertRed(context).withValues(alpha: 0.12)
            : const Color(0xFFFFF8F8),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.warning_amber_rounded,
                  size: 12, color: _alertRed(context)),
              const SizedBox(width: 4),
              Text(
                'Key Exclusions (from policy PDF)',
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w700,
                  fontSize: 10,
                  color: _alertRed(context),
                ),
              ),
            ],
          ),
          const SizedBox(height: 2),
          // The detail line under the red header. This was
          // AppTheme.tertiaryText (#888888, about 3.5:1 on the blush-white
          // wash) — a second, quieter grey inside what reads as one warning
          // block, which is exactly the mixed-colour run the audit flagged.
          Text(
            policy.exclusions.join(' · '),
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w700,
              fontSize: 10,
              color: _muted(context),
            ),
          ),
        ],
      ),
    );
  }

  /// The source filename, and nothing else.
  ///
  /// The green "Parsed" badge that sat at the right was removed: this card only
  /// exists because a PDF parsed successfully, so the badge could never read
  /// anything but "Parsed" and told the user nothing a card's presence did not
  /// already tell them. The paperclip emoji before the filename is now
  /// Material's description glyph, matching the icon system everywhere else.
  ///
  /// The delete control, when offered, sits at the right end of this row.
  Widget _buildFooter(BuildContext context) {
    return Row(
      children: [
        Icon(Icons.description_outlined, size: 12, color: _muted(context)),
        const SizedBox(width: 6),
        Expanded(
          child: Text(
            policy.originalFilename ?? 'policy.pdf',
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w700,
              fontSize: 10,
              color: _muted(context),
            ),
          ),
        ),
        if (onDelete != null) ...[
          const SizedBox(width: 8),
          _buildDeleteAction(context),
        ],
      ],
    );
  }

  Widget _buildDeleteAction(BuildContext context) {
    if (deleting) return const _DeleteSpinner();
    return IconButton(
      icon: Opacity(
        opacity: deleteDisabled ? 0.4 : 1,
        child: const _DeleteIcon(),
      ),
      onPressed: deleteDisabled ? null : onDelete,
      tooltip: 'Delete ${policy.policyName}',
      constraints: const BoxConstraints(minWidth: 40, minHeight: 40),
      padding: EdgeInsets.zero,
      visualDensity: VisualDensity.compact,
    );
  }
}

/// Delete Account's row badge from Settings (`_RowIcon(Icons.delete_outline,
/// danger: true)`), repeated here rather than imported: that widget is private
/// to settings_screen.dart. Same icon, 32x32 tile, radius 10, and the same
/// [AppTheme.chip] danger pair, so the dark-mode inversion matches too. If one
/// changes, change both.
class _DeleteIcon extends StatelessWidget {
  const _DeleteIcon();

  @override
  Widget build(BuildContext context) {
    final tint = AppTheme.chip(
      context,
      background: AppColors.dangerTint,
      foreground: AppColors.danger,
    );
    return Container(
      width: 32,
      height: 32,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        color: tint.background,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Icon(Icons.delete_outline, size: 18, color: tint.foreground),
    );
  }
}

/// Settings' `_RowSpinner`, sized to the delete IconButton's 40x40 hit box
/// rather than its 32x32 so the footer row does not shift when a delete starts.
class _DeleteSpinner extends StatelessWidget {
  const _DeleteSpinner();

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 40,
      height: 40,
      child: Center(
        child: SizedBox(
          width: 16,
          height: 16,
          child: CircularProgressIndicator(
            strokeWidth: 2,
            color: _primaryText(context),
          ),
        ),
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  const _Stat({
    required this.label,
    required this.value,
    this.valueColor,
  });

  final String label;
  final String value;

  /// Null means "whatever body text is in this theme"; only the
  /// renewing-soon stat overrides it.
  final Color? valueColor;

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Column(
        children: [
          FittedBox(
            fit: BoxFit.scaleDown,
            child: Text(
              label,
              style: GoogleFonts.inter(fontSize: 13, color: _muted(context)),
            ),
          ),
          const SizedBox(height: 4),
          FittedBox(
            fit: BoxFit.scaleDown,
            child: Text(
              value,
              style: GoogleFonts.spaceGrotesk(
                fontWeight: FontWeight.w700,
                fontSize: 17,
                color: valueColor ?? _primaryText(context),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _TypeBadge extends StatelessWidget {
  const _TypeBadge({required this.policy});

  final InsurancePolicy policy;

  @override
  Widget build(BuildContext context) {
    final chip =
        AppTheme.chip(context, background: _kTagBg, foreground: _kTagFg);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 2),
      decoration: BoxDecoration(
        color: chip.background,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(
        policy.policyTypeLabel,
        style: GoogleFonts.inter(
          fontWeight: FontWeight.w600,
          fontSize: 10,
          color: chip.foreground,
        ),
      ),
    );
  }
}
