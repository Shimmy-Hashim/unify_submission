import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../utils/app_styles.dart';

/// A surface card whose header row expands and collapses a body beneath it —
/// the shared shell behind Home's Wealth Breakdown rows and its Score History
/// card, which had grown two line-for-line copies of the same widget.
///
/// Owns only the open/closed state and the chrome: the rounded card, the tap
/// target, the title, the 180ms chevron turn and the 180ms cross-fade. What
/// the header says on the right ([trailing]) and what opens beneath it
/// ([body]) belong to the caller.
///
/// While [enabled] is false — the caller's data has not landed yet, or its
/// listener failed — the header does nothing on tap and the body stays shut.
/// An already-open card keeps its open state underneath, so it reopens by
/// itself when the data comes back rather than silently snapping closed.
///
/// Give each instance a key when it sits in a list whose length can change:
/// the open state lives in this widget's State, and without a key Flutter
/// hands that State to whichever card ends up in the same slot.
class CollapsibleCard extends StatefulWidget {
  const CollapsibleCard({
    super.key,
    required this.title,
    required this.titleSize,
    this.titleColor,
    required this.trailing,
    required this.enabled,
    required this.body,
  });

  final String title;
  final double titleSize;

  /// Defaults to the theme's primary ink. Debts sets its red here, per Figma.
  final Color? titleColor;

  /// Right of the title, before the chevron — a total, a trend, a dash.
  final Widget trailing;

  final bool enabled;
  final Widget body;

  @override
  State<CollapsibleCard> createState() => _CollapsibleCardState();
}

class _CollapsibleCardState extends State<CollapsibleCard> {
  static const _duration = Duration(milliseconds: 180);

  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: AppTheme.surface(context),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [AppTheme.cardShadow(context)],
      ),
      child: Column(
        children: [
          InkWell(
            borderRadius: BorderRadius.circular(20),
            onTap: widget.enabled
                ? () => setState(() => _expanded = !_expanded)
                : null,
            child: Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 15, vertical: 12),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      widget.title,
                      style: GoogleFonts.spaceGrotesk(
                        fontWeight: FontWeight.w700,
                        fontSize: widget.titleSize,
                        color:
                            widget.titleColor ?? AppTheme.primaryText(context),
                      ),
                    ),
                  ),
                  widget.trailing,
                  const SizedBox(width: 6),
                  AnimatedRotation(
                    turns: _expanded ? 0.5 : 0,
                    duration: _duration,
                    child: Icon(Icons.keyboard_arrow_down,
                        color: AppTheme.primaryText(context), size: 22),
                  ),
                ],
              ),
            ),
          ),
          AnimatedCrossFade(
            duration: _duration,
            crossFadeState: _expanded && widget.enabled
                ? CrossFadeState.showSecond
                : CrossFadeState.showFirst,
            firstChild: const SizedBox(width: double.infinity),
            secondChild: widget.body,
          ),
        ],
      ),
    );
  }
}
