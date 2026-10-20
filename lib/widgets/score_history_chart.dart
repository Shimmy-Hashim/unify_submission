import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../models/score_history.dart';
import '../utils/app_styles.dart';
import 'tier_segment_bar.dart';

/// The Score History card's line chart: overall score per day.
///
/// PAINTED, NOT A PACKAGE. The project has no chart dependency — fl_chart was
/// removed once its only consumer went (pubspec.yaml records why) — and one
/// small line chart does not justify bringing a package back. Drawn the same
/// way DashedBorderPainter is: a CustomPainter handed its resolved colours,
/// since a painter has no BuildContext to read a theme from.
///
/// WHAT IS DRAWN, AND WHAT IS DELIBERATELY NOT
///   - x is real time between the first and last snapshot, not an index, so
///     a week with no refreshes reads as a week.
///   - y is the full 0-100 range, fixed. Auto-scaling a 66 -> 67 history
///     would draw a cliff out of one point.
///   - The two faint guides sit at 40 and 70, the score band boundaries the
///     marker colours come from (TierSegmentBar.scoreColor), so a marker's
///     colour and its height always tell the same story.
///   - A day with no score is a BREAK in the line, never a point at zero —
///     backend main.py: "The chart should show it as a gap." It is marked
///     with a small hollow ring on the baseline so the break is legible.
///   - The line itself is neutral grey; colour lives only on the markers.
///     Lime is never used — it is reserved for actions.
class ScoreHistoryChart extends StatelessWidget {
  const ScoreHistoryChart({super.key, required this.points});

  /// Oldest first ([chronological]).
  final List<ScoreHistoryPoint> points;

  static const height = 132.0;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: height,
      width: double.infinity,
      child: CustomPaint(
        painter: _ScoreHistoryPainter(
          points: points,
          line: AppTheme.tertiaryText(context),
          guide: AppTheme.divider(context),
          label: AppTheme.tertiaryText(context),
          surface: AppTheme.surface(context),
        ),
      ),
    );
  }
}

class _ScoreHistoryPainter extends CustomPainter {
  _ScoreHistoryPainter({
    required this.points,
    required this.line,
    required this.guide,
    required this.label,
    required this.surface,
  });

  final List<ScoreHistoryPoint> points;
  final Color line;
  final Color guide;
  final Color label;
  final Color surface;

  static const _leftGutter = 26.0;
  static const _bottomGutter = 18.0;
  static const _topPad = 8.0;
  static const _sidePad = 8.0;
  static const _markerRadius = 4.0;

  static const _months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
  ];

  TextPainter _text(String s) => TextPainter(
        text: TextSpan(
          text: s,
          style: GoogleFonts.inter(
              fontSize: 9, fontWeight: FontWeight.w500, color: label),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

  @override
  void paint(Canvas canvas, Size size) {
    final plot = Rect.fromLTRB(
      _leftGutter + _sidePad,
      _topPad,
      size.width - _sidePad,
      size.height - _bottomGutter,
    );
    double yFor(double score) =>
        plot.bottom - (score.clamp(0, 100) / 100) * plot.height;

    // Guides: baseline, the two band boundaries, and the top.
    final guidePaint = Paint()
      ..color = guide
      ..strokeWidth = 1;
    for (final value in const [0.0, 40.0, 70.0, 100.0]) {
      final y = yFor(value);
      canvas.drawLine(Offset(plot.left, y), Offset(plot.right, y), guidePaint);
      final t = _text(value.toInt().toString());
      t.paint(canvas, Offset(_leftGutter - t.width, y - t.height / 2));
    }

    if (points.isEmpty) return;

    final first = points.first.date!;
    final last = points.last.date!;
    final spanDays = last.difference(first).inHours / 24;
    double xFor(DateTime d) => spanDays == 0
        ? plot.center.dx
        : plot.left + (d.difference(first).inHours / 24 / spanDays) * plot.width;

    // Line segments between consecutive SCORED days; a gap day breaks it.
    final linePaint = Paint()
      ..color = line
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke
      ..strokeJoin = StrokeJoin.round
      ..strokeCap = StrokeCap.round;
    Path? path;
    for (final p in points) {
      final score = p.overallScore;
      if (score == null) {
        if (path != null) canvas.drawPath(path, linePaint);
        path = null;
        continue;
      }
      final o = Offset(xFor(p.date!), yFor(score));
      path == null ? path = (Path()..moveTo(o.dx, o.dy)) : path.lineTo(o.dx, o.dy);
    }
    if (path != null) canvas.drawPath(path, linePaint);

    // Markers over the line.
    for (final p in points) {
      final x = xFor(p.date!);
      final score = p.overallScore;
      if (score == null) {
        canvas.drawCircle(
          Offset(x, plot.bottom),
          2.5,
          Paint()
            ..color = label
            ..style = PaintingStyle.stroke
            ..strokeWidth = 1.2,
        );
        continue;
      }
      final o = Offset(x, yFor(score));
      canvas.drawCircle(o, _markerRadius + 1.5, Paint()..color = surface);
      canvas.drawCircle(
          o, _markerRadius, Paint()..color = TierSegmentBar.scoreColor(score));
    }

    // Date labels: first and last day (one label when they are the same).
    String fmt(DateTime d) => '${d.day} ${_months[d.month - 1]}';
    final labelY = plot.bottom + 5;
    final firstLabel = _text(fmt(first));
    if (spanDays == 0) {
      firstLabel.paint(
          canvas, Offset(plot.center.dx - firstLabel.width / 2, labelY));
    } else {
      firstLabel.paint(canvas, Offset(plot.left - 4, labelY));
      final lastLabel = _text(fmt(last));
      lastLabel.paint(
          canvas, Offset(plot.right - lastLabel.width + 4, labelY));
    }
  }

  @override
  bool shouldRepaint(_ScoreHistoryPainter old) =>
      old.points != points ||
      old.line != line ||
      old.guide != guide ||
      old.label != label ||
      old.surface != surface;
}
