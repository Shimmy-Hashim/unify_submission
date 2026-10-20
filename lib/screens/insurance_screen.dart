import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../models/coverage_gap_item.dart';
import '../models/home_score_state.dart';
import '../models/insurance_policy.dart';
import '../services/firebase_service.dart';
import '../services/health_score_service.dart';
import '../services/insurance_service.dart';
import '../utils/app_styles.dart';
import '../utils/formatters.dart';
import '../widgets/insurance_card.dart';

// ---- Audit palette, scoped to this screen -------------------------------
// These three shadow shared AppTheme tokens rather than editing them, the same
// containment Settings, Savings and Investments used: the shared values are
// still read by Home, and Home has its own audit task.

/// Secondary text — section labels, the empty-policies line, the upload card's
/// captions.
///
/// Light is the WCAG AA fix: [AppTheme.muted] (#AAAAAA) measures about 2.3:1
/// on white, against the 4.5:1 minimum. #6B6B6B clears it at 5.3:1, and is the
/// value the audit frame uses.
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

/// The Total Coverage card at the top. [AppTheme.heroSurface] is pure black in
/// light mode; the frame's card is #202020. Dark mode keeps heroSurface's own
/// #2A2A2A for the documented reason — #202020 on the #121212 background sits
/// under the #1E1E1E of the ordinary cards around it, so the one card meant to
/// stand out would recede instead.
Color _heroSurface(BuildContext context) =>
    AppTheme.isDark(context) ? const Color(0xFF2A2A2A) : AppColors.brandBlack;

/// Sits on the hero card, which is dark in both modes — no variant needed.
const _kNotCoveredOrange = Color(0xFFFFA726);

/// Insurance tab — Figma node 84:519.
///
/// Every section is real data: Total Coverage summary (computed from
/// insurancePolicies), Coverage Gaps (the cached score's own coverage ratios
/// against the LIA benchmarks — see _CoverageGapsSection), Active Policies list
/// (Firestore stream), and the upload flow (backend POST /insurance/parse).
class InsuranceScreen extends StatefulWidget {
  const InsuranceScreen({super.key});

  @override
  State<InsuranceScreen> createState() => _InsuranceScreenState();
}

class _InsuranceScreenState extends State<InsuranceScreen> {
  bool _uploading = false;

  /// The policy currently being deleted, if any. Held per-policy rather than
  /// as a single bool, as Settings' disconnect does, so the spinner sits on
  /// the card the user actually tapped and a second delete cannot start while
  /// one is in flight.
  String? _deletingPolicyId;

  Future<void> _confirmDelete(InsurancePolicy policy) async {
    if (_deletingPolicyId != null) return;
    final confirmed =
        await showDeletePolicyConfirmation(context, policy.policyName);
    if (confirmed != true || !mounted) return;

    setState(() => _deletingPolicyId = policy.policyId);
    String message;
    try {
      await InsuranceService.instance.deletePolicy(policy.policyId);
      // No local list to prune: the insurancePolicies stream fires as soon as
      // the backend's delete lands, so this card (and Home's figures) update
      // on their own.
      message = 'Deleted "${policy.policyName}".';
    } on InsuranceDeleteException catch (e) {
      // Every backend failure path leaves the policy in place, so the card
      // stays and the user can simply try again.
      message = e.message;
    } catch (_) {
      message = 'Could not delete "${policy.policyName}". Please try again.';
    }
    if (!mounted) return;
    setState(() => _deletingPolicyId = null);
    _showMessage(message);
  }

  Future<void> _pickAndUpload() async {
    if (_uploading) return;
    // One try around the picker AND the upload: the picker throws too (a
    // platform error, an unreadable file), and nothing it throws may escape.
    // _uploading is still only set once a file is in hand, so the parsing
    // state never shows while the picker is open.
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['pdf'],
        withData: true,
      );
      final file = result?.files.firstOrNull;
      if (file == null || !mounted) return; // picker cancelled
      final bytes = file.bytes;
      if (bytes == null || bytes.isEmpty) {
        _showMessage('Could not read the selected file. Try again.');
        return;
      }

      setState(() => _uploading = true);
      final doc = await InsuranceService.instance.uploadAndParse(
        bytes: bytes,
        filename: file.name,
      );
      // The Firestore stream picks up the new document automatically.
      _showMessage('Added "${doc['policy_name'] ?? 'policy'}" to your list.');
    } on InsuranceParseException catch (e) {
      _showMessage(e.message);
    } catch (_) {
      _showMessage('Could not read the selected file. Try again.');
    } finally {
      // _showMessage checks mounted itself; setState needs it here.
      if (mounted && _uploading) setState(() => _uploading = false);
    }
  }

  void _showMessage(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    final uid = FirebaseService.instance.currentUser?.uid;
    return Scaffold(
      backgroundColor: AppTheme.background(context),
      body: uid == null
          ? const Center(child: Text('Not signed in'))
          : SafeArea(
              child: StreamBuilder<QuerySnapshot<Map<String, dynamic>>>(
                stream: FirebaseService.instance
                    .insurancePolicies(uid)
                    .orderBy('uploaded_at', descending: true)
                    .snapshots(),
                builder: (context, snap) {
                  if (snap.hasError) {
                    return Center(
                      child: Text('Could not load policies: ${snap.error}'),
                    );
                  }
                  if (!snap.hasData) {
                    return const Center(child: CircularProgressIndicator());
                  }
                  final policies = snap.data!.docs
                      .map((d) => InsurancePolicy.fromFirestore(d.id, d.data()))
                      .toList();
                  return _buildContent(uid, policies);
                },
              ),
            ),
    );
  }

  Widget _buildContent(String uid, List<InsurancePolicy> policies) {
    final summary = CoverageSummary.fromPolicies(policies);
    return ListView(
      padding: const EdgeInsets.fromLTRB(10, 12, 10, 24),
      children: [
        TotalCoverageCard(summary: summary),
        const SizedBox(height: 20),
        const _SectionLabel('COVERAGE GAPS'),
        const SizedBox(height: 8),
        _CoverageGapsSection(uid: uid),
        const SizedBox(height: 20),
        const _SectionLabel('ACTIVE POLICIES'),
        const SizedBox(height: 8),
        if (policies.isEmpty)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 24),
            child: Text(
              'No policies yet. Upload one below to get started.',
              textAlign: TextAlign.center,
              style: GoogleFonts.inter(fontSize: 13, color: _muted(context)),
            ),
          )
        else
          for (final policy in policies) ...[
            InsuranceCard(
              policy: policy,
              onDelete: () => _confirmDelete(policy),
              deleting: _deletingPolicyId == policy.policyId,
              deleteDisabled: _deletingPolicyId != null,
            ),
            const SizedBox(height: 14),
          ],
        const SizedBox(height: 6),
        const _SectionLabel('ADD A POLICY'),
        const SizedBox(height: 8),
        _UploadCard(uploading: _uploading, onTap: _pickAndUpload),
      ],
    );
  }
}

/// The gate in front of a policy delete. Returns true only if the user
/// explicitly confirms — false on Cancel, and null if the dialog is dismissed
/// by a barrier tap or the back button, both of which callers must treat as
/// "do nothing".
///
/// Modelled on Settings' showDisconnectConfirmation, and deliberately as light:
/// Delete Account's typed "DELETE" gate is for destroying the whole account,
/// while this removes one policy that re-uploading the PDF brings back.
///
/// Top-level and public so the gate itself is testable without the
/// Firestore-backed screen around it.
Future<bool?> showDeletePolicyConfirmation(
  BuildContext context,
  String policyName,
) {
  return showDialog<bool>(
    context: context,
    builder: (dialogContext) => AlertDialog(
      backgroundColor: AppTheme.surface(dialogContext),
      title: Text(
        'Are you sure you want to delete ‘$policyName’?',
        style: AppTextStyles.fieldLabel
            .copyWith(color: _primaryText(dialogContext), fontSize: 17),
      ),
      content: Text(
        'It will be removed from your coverage totals, and your Financial '
        'Health Score will be recalculated without it.\n\n'
        'Unify does not keep a copy of the PDF. To add it back, upload the '
        'policy again.',
        style: GoogleFonts.inter(
          fontSize: 13,
          height: 1.45,
          color: _muted(dialogContext),
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(dialogContext).pop(false),
          child: Text(
            'Cancel',
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w600,
              color: _primaryText(dialogContext),
            ),
          ),
        ),
        TextButton(
          onPressed: () => Navigator.of(dialogContext).pop(true),
          child: Text(
            'Delete',
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w600,
              color: AppTheme.alertRed(dialogContext, AppColors.danger),
            ),
          ),
        ),
      ],
    ),
  );
}

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
          color: _muted(context),
        ),
      ),
    );
  }
}

/// Black summary card — Figma node 84:647. All figures are straight sums
/// over the user's policies; "Accident" groups personal_accident +
/// hospitalisation per the screen's three-column breakdown.
///
/// Public only so its pills can be widget-tested; the screen around it is
/// backed by a live Firestore stream and is not.
@visibleForTesting
class TotalCoverageCard extends StatelessWidget {
  const TotalCoverageCard({super.key, required this.summary});

  final CoverageSummary summary;

  @override
  Widget build(BuildContext context) {
    final n = summary.policyCount;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
      decoration: BoxDecoration(
        color: _heroSurface(context),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Total Coverage',
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w500,
              fontSize: 13,
              color: Colors.white.withValues(alpha: 0.55),
            ),
          ),
          const SizedBox(height: 2),
          Text(
            formatSgd(summary.totalSumAssured),
            style: GoogleFonts.spaceGrotesk(
              fontWeight: FontWeight.w700,
              fontSize: 40,
              height: 1.1,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            'Across $n active ${n == 1 ? 'policy' : 'policies'} · '
            '${formatSgd(summary.totalAnnualPremiums)}/yr in premiums',
            style: GoogleFonts.spaceGrotesk(
              fontWeight: FontWeight.w700,
              fontSize: 12,
              color: Colors.white.withValues(alpha: 0.55),
            ),
          ),
          const SizedBox(height: 12),
          Divider(
            height: 1,
            thickness: 1,
            color: Colors.white.withValues(alpha: 0.12),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              _CoverageColumn(
                label: 'Life Cover',
                sum: summary.lifeSumAssured,
                policyCount: summary.lifePolicyCount,
                noAmountLabel: 'On file',
              ),
              _CoverageColumn(
                label: 'Accident',
                sum: summary.accidentSumAssured,
                policyCount: summary.accidentPolicyCount,
              ),
              _CoverageColumn(
                label: 'Critical Illness',
                sum: summary.ciSumAssured,
                policyCount: summary.ciPolicyCount,
                noAmountLabel: 'On file',
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _CoverageColumn extends StatelessWidget {
  const _CoverageColumn({
    required this.label,
    required this.sum,
    required this.policyCount,
    this.noAmountLabel = 'Covered',
  });

  final String label;
  final double sum;
  final int policyCount;

  /// Shown when policies exist but none carries a sum assured. Life and CI
  /// pass "On file": the Coverage Gaps cards judge those two by amount, so a
  /// null-sum policy there is a gap, and "Covered" above it would contradict
  /// the card.
  final String noAmountLabel;

  @override
  Widget build(BuildContext context) {
    // A category can hold policies whose sum assured is legitimately null
    // (personal_accident under the fabrication-avoidance rule, or a PDF that
    // states none) — show [noAmountLabel] rather than a made-up $0.
    final String value;
    final Color color;
    if (policyCount == 0) {
      value = 'Not covered';
      color = _kNotCoveredOrange;
    } else if (sum <= 0) {
      value = noAmountLabel;
      color = Colors.white;
    } else {
      value = formatSgd(sum);
      color = Colors.white;
    }
    return Expanded(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Text(
            label,
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w500,
              fontSize: 10,
              color: Colors.white.withValues(alpha: 0.45),
            ),
          ),
          const SizedBox(height: 4),
          FittedBox(
            fit: BoxFit.scaleDown,
            child: Text(
              value,
              style: GoogleFonts.spaceGrotesk(
                fontWeight: FontWeight.w700,
                fontSize: 13,
                color: color,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Coverage Gaps — Figma node 84:661, now real.
///
/// Streams the SAME cached score document Home reads (financialProfile), so
/// the numbers on these cards and the Insurance tier behind the score can never
/// disagree: both come from one `compute_insurance_features()` run. Nothing is
/// computed here — see models/coverage_gap_item.dart for what each line is
/// allowed to say and where its benchmark comes from.
///
/// A second StreamBuilder rather than lifting the score into the screen's
/// existing policy stream: the score is this section's dependency alone, and
/// every other section of the screen keeps working (and keeps rendering) if it
/// is slow, absent or failing.
class _CoverageGapsSection extends StatelessWidget {
  const _CoverageGapsSection({required this.uid});

  final String uid;

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<ScoreState>(
      stream: HealthScoreService.instance.watch(uid),
      builder: (context, snap) {
        final state = snap.hasError
            ? ScoreFailed('${snap.error}')
            : (snap.data ?? const ScoreLoading());
        final view = CoverageGapsView.fromScoreState(state);

        if (!view.hasItems) {
          return _GapNoticeCard(message: view.notice ?? '');
        }
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (view.summary != null) ...[
              Padding(
                padding: const EdgeInsets.only(left: 4, bottom: 8),
                child: Text(
                  view.summary!,
                  style: GoogleFonts.inter(
                    fontWeight: FontWeight.w500,
                    fontSize: 11,
                    color: AppTheme.secondaryText(context),
                  ),
                ),
              ),
            ],
            // The peek affordance is a real clip, restored.
            //
            // This carried `clipBehavior: Clip.none`, which switches the
            // viewport's clip off entirely: the off-screen cards were painted
            // in full, past the scroll view's own right edge, and only the
            // outer ListView's bounds at the physical screen edge stopped
            // them. The cut the user saw was therefore the screen ending, not
            // the strip continuing — and the overflowing cards' shadows
            // painted across the list's 10pt right padding on the way out.
            // Default (hard-edge) clipping puts the cut back on the strip,
            // so the last card is genuinely trimmed by its own container.
            //
            // The 12pt of bottom padding is why Clip.none was reached for in
            // the first place, and is the same fix the Investments Risk Alerts
            // strip needed: AppTheme.cardShadow is offset (0,4) with a 4pt
            // blur, so a viewport sized exactly to the card clips the shadow
            // out of existence and the cards read as flat panels. Padding
            // gives the shadow somewhere to paint without turning clipping
            // off. 2pt each side does the same for the blur's horizontal
            // spread on the first and last cards.
            //
            // Overflow is genuine here, not forced: four 200pt cards plus
            // three 8pt gaps is 824pt against a viewport of roughly 340 on a
            // Pixel 7, so three and a bit cards' worth is always off-strip.
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.fromLTRB(2, 0, 2, 12),
              child: Row(
                children: [
                  for (final (i, item) in view.items.indexed) ...[
                    if (i > 0) const SizedBox(width: 8),
                    _CoverageGapCard(item: item),
                  ],
                ],
              ),
            ),
          ],
        );
      },
    );
  }
}

/// One coverage line. Same scaffold as the shell it replaces (rounded-16,
/// shadow, icon + title + description + badge pill) and the same shape as
/// Home's InsightCard, widened to 200 because these carry a real sentence.
class _CoverageGapCard extends StatelessWidget {
  const _CoverageGapCard({required this.item});

  final CoverageGapItem item;

  /// Reuses the Insights palette: green for a met benchmark, amber for a
  /// stated gap, neutral grey for "not assessed". Amber, not red — the section
  /// describes a distance to a benchmark, it does not raise an alarm.
  static const _pillColors = {
    CoverageStatus.covered: (bg: Color(0xFFE8F5E9), fg: Color(0xFF2E7D32)),
    CoverageStatus.gap: (bg: Color(0xFFFFF3E0), fg: Color(0xFFE65100)),
    CoverageStatus.notAssessed: (bg: Color(0xFFEEEEEE), fg: Color(0xFF757575)),
  };

  @override
  Widget build(BuildContext context) {
    final base = _pillColors[item.status]!;
    final chip =
        AppTheme.chip(context, background: base.bg, foreground: base.fg);
    return Container(
      width: 200,
      height: 158,
      padding: const EdgeInsets.fromLTRB(14, 8, 14, 10),
      decoration: BoxDecoration(
        color: AppTheme.surface(context),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [AppTheme.cardShadow(context)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(item.icon, size: 16, color: _muted(context)),
          const SizedBox(height: 2),
          Text(
            item.title,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w600,
              fontSize: 11,
              color: _primaryText(context),
            ),
          ),
          const SizedBox(height: 4),
          Expanded(
            child: Text(
              item.message,
              style: GoogleFonts.inter(
                fontWeight: FontWeight.w600,
                fontSize: 10,
                height: 1.4,
                color: AppTheme.subtleBody(context),
              ),
            ),
          ),
          // Align, because a Container with a non-null `alignment` expands to
          // fill its parent — the pill must hug its label.
          Align(
            alignment: Alignment.centerLeft,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
              decoration: BoxDecoration(
                color: chip.background,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                item.pillLabel,
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w600,
                  fontSize: 8,
                  color: chip.foreground,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Shown instead of the strip when there is nothing assessed to show. Full
/// width rather than a card in a strip: there is one thing to say, and saying
/// it three times across three identical cards (what the shell did) implied
/// three separate pending findings where there were none.
class _GapNoticeCard extends StatelessWidget {
  const _GapNoticeCard({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
      decoration: BoxDecoration(
        color: AppTheme.surface(context),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [AppTheme.cardShadow(context)],
      ),
      child: Text(
        message,
        style: GoogleFonts.inter(
          fontWeight: FontWeight.w600,
          fontSize: 11,
          height: 1.45,
          color: AppTheme.subtleBody(context),
        ),
      ),
    );
  }
}

/// Upload card — Figma node 84:522. The dashed drop-zone doubles as the
/// progress indicator while the backend parses the PDF.
class _UploadCard extends StatelessWidget {
  const _UploadCard({required this.uploading, required this.onTap});

  final bool uploading;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
      decoration: BoxDecoration(
        color: AppTheme.surface(context),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [AppTheme.cardShadow(context)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Upload Policy PDF',
            style: GoogleFonts.spaceGrotesk(
              fontWeight: FontWeight.w700,
              fontSize: 15,
              color: _primaryText(context),
            ),
          ),
          Text(
            'Supports PDF policies from any provider',
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w700,
              fontSize: 11,
              // Was a translucent #AAAAAA at 67%, landing near #C9C9C9 on
              // white — about 1.7:1, the lowest-contrast text on the screen.
              // Joins the rest of the secondary copy at #6B6B6B.
              color: _muted(context),
            ),
          ),
          const SizedBox(height: 10),
          InkWell(
            onTap: uploading ? null : onTap,
            borderRadius: BorderRadius.circular(14),
            child: CustomPaint(
              painter: _DashedBorderPainter(
                color: AppTheme.dashedBorder(context),
                radius: 14,
              ),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(
                  horizontal: 24,
                  vertical: 14,
                ),
                decoration: BoxDecoration(
                  color: AppTheme.inset(context),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: uploading
                    ? _buildParsing(context)
                    : _buildIdle(context),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildIdle(BuildContext context) {
    return Column(
      children: [
        Icon(Icons.description_outlined, size: 28, color: _muted(context)),
        const SizedBox(height: 4),
        Text(
          'Tap to upload',
          style: GoogleFonts.spaceGrotesk(
            fontWeight: FontWeight.w700,
            fontSize: 13,
            color: _primaryText(context),
          ),
        ),
        const SizedBox(height: 4),
        Text(
          'Upload your insurance policy PDF to extract coverage, '
          'exclusions, premiums and key dates automatically.',
          textAlign: TextAlign.center,
          style: GoogleFonts.inter(
            fontWeight: FontWeight.w500,
            fontSize: 10,
            height: 1.5,
            color: _muted(context),
          ),
        ),
      ],
    );
  }

  Widget _buildParsing(BuildContext context) {
    return Column(
      children: [
        const SizedBox(height: 6),
        SizedBox(
          width: 26,
          height: 26,
          child: CircularProgressIndicator(
            strokeWidth: 2.5,
            color: _primaryText(context),
          ),
        ),
        const SizedBox(height: 10),
        Text(
          'Parsing your policy…',
          style: GoogleFonts.spaceGrotesk(
            fontWeight: FontWeight.w700,
            fontSize: 13,
            color: _primaryText(context),
          ),
        ),
        const SizedBox(height: 4),
        Text(
          'Extracting coverage, exclusions and key dates. '
          'This can take a few seconds.',
          textAlign: TextAlign.center,
          style: GoogleFonts.inter(
            fontWeight: FontWeight.w500,
            fontSize: 10,
            height: 1.5,
            color: _muted(context),
          ),
        ),
        const SizedBox(height: 6),
      ],
    );
  }
}

/// Flutter has no built-in dashed border; draws one around a rounded rect.
class _DashedBorderPainter extends CustomPainter {
  _DashedBorderPainter({required this.color, required this.radius});

  final Color color;
  final double radius;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    final path = Path()
      ..addRRect(RRect.fromRectAndRadius(
        Offset.zero & size,
        Radius.circular(radius),
      ));
    const dash = 5.0, gap = 4.0;
    for (final metric in path.computeMetrics()) {
      var distance = 0.0;
      while (distance < metric.length) {
        canvas.drawPath(
          metric.extractPath(distance, distance + dash),
          paint,
        );
        distance += dash + gap;
      }
    }
  }

  @override
  bool shouldRepaint(_DashedBorderPainter oldDelegate) =>
      color != oldDelegate.color || radius != oldDelegate.radius;
}
