import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:unify/models/insurance_policy.dart';
import 'package:unify/screens/insurance_screen.dart';
import 'package:unify/widgets/insurance_card.dart';

/// Policy delete: the confirmation gate and the card's control.
///
/// The Insurance screen itself is backed by a live Firestore stream and is not
/// widget-tested, the same reason Settings' disconnect is covered through its
/// gate (test/settings_disconnect_test.dart). What matters for a destructive
/// action is pinned here: the delete is unreachable without an explicit
/// confirmation, every other way out of the dialog does nothing, and the
/// card's control is shown, spinning, or disabled-but-visible as asked.
void main() {
  const policy = InsurancePolicy(
    policyId: 'p1',
    policyName: 'PRUActive Term',
    insurer: 'Prudential',
    policyType: 'life',
    sumAssuredSgd: 500000,
    annualPremiumSgd: 1200,
    originalFilename: 'PRUActive Term.pdf',
  );

  /// Pumps a button that opens the dialog and records what it resolved to,
  /// plus how many times a stand-in "delete" actually ran.
  Future<Map<String, dynamic>> openDialog(
    WidgetTester tester, {
    required Future<void> Function(WidgetTester) act,
  }) async {
    bool? resolved;
    var deleteCalls = 0;

    await tester.pumpWidget(MaterialApp(
      home: Scaffold(
        body: Builder(
          builder: (context) => ElevatedButton(
            onPressed: () async {
              resolved =
                  await showDeletePolicyConfirmation(context, policy.policyName);
              // Mirrors the real call site's guard exactly.
              if (resolved != true) return;
              deleteCalls++;
            },
            child: const Text('open'),
          ),
        ),
      ),
    ));

    await tester.tap(find.text('open'));
    await tester.pumpAndSettle();
    await act(tester);
    await tester.pumpAndSettle();

    return {'resolved': resolved, 'calls': deleteCalls};
  }

  group('showDeletePolicyConfirmation', () {
    testWidgets('asks by policy name and says what happens', (tester) async {
      await openDialog(tester, act: (t) async {
        expect(
          find.text('Are you sure you want to delete ‘PRUActive Term’?'),
          findsOneWidget,
        );
        final body = t.widget<Text>(find.byWidgetPredicate((w) =>
            w is Text && (w.data ?? '').contains('coverage totals')));
        expect(body.data, contains('recalculated'));
        // Recoverable by re-uploading — the reason this is a plain confirm
        // rather than Delete Account's typed gate.
        expect(body.data, contains('upload the policy again'));
        expect(find.text('Cancel'), findsOneWidget);
        expect(find.text('Delete'), findsOneWidget);
        await t.tap(find.text('Cancel'));
      });
    });

    testWidgets('Cancel resolves false and does NOT delete', (tester) async {
      final result = await openDialog(tester,
          act: (t) => t.tap(find.text('Cancel')));
      expect(result['resolved'], isFalse);
      expect(result['calls'], 0);
      expect(find.byType(AlertDialog), findsNothing);
    });

    testWidgets('a barrier tap resolves null and does NOT delete',
        (tester) async {
      final result = await openDialog(tester,
          act: (t) => t.tapAt(const Offset(5, 5)));
      expect(result['resolved'], isNull);
      expect(result['calls'], 0);
    });

    testWidgets('Delete resolves true and deletes exactly once',
        (tester) async {
      final result = await openDialog(tester,
          act: (t) => t.tap(find.text('Delete')));
      expect(result['resolved'], isTrue);
      expect(result['calls'], 1);
    });
  });

  group('InsuranceCard delete control', () {
    Future<void> pumpCard(WidgetTester tester, InsuranceCard card) =>
        tester.pumpWidget(MaterialApp(
          home: Scaffold(body: SingleChildScrollView(child: card)),
        ));

    testWidgets('absent when no onDelete is given', (tester) async {
      await pumpCard(tester, const InsuranceCard(policy: policy));
      expect(find.byIcon(Icons.delete_outline), findsNothing);
    });

    testWidgets('shown on the filename row, and tapping it calls onDelete',
        (tester) async {
      var taps = 0;
      await pumpCard(
          tester, InsuranceCard(policy: policy, onDelete: () => taps++));

      final icon = find.byIcon(Icons.delete_outline);
      expect(icon, findsOneWidget);
      // Same row as the filename, to its right.
      final filename = find.text('PRUActive Term.pdf');
      expect(tester.getCenter(icon).dy,
          closeTo(tester.getCenter(filename).dy, 1));
      expect(tester.getCenter(icon).dx,
          greaterThan(tester.getCenter(filename).dx));

      await tester.tap(icon);
      expect(taps, 1);
    });

    testWidgets('a spinner replaces it while this card is deleting',
        (tester) async {
      await pumpCard(
        tester,
        InsuranceCard(policy: policy, onDelete: () {}, deleting: true),
      );
      expect(find.byType(CircularProgressIndicator), findsOneWidget);
      expect(find.byIcon(Icons.delete_outline), findsNothing);
    });

    testWidgets('visible but disabled while another card is deleting',
        (tester) async {
      var taps = 0;
      await pumpCard(
        tester,
        InsuranceCard(
            policy: policy, onDelete: () => taps++, deleteDisabled: true),
      );
      expect(find.byIcon(Icons.delete_outline), findsOneWidget);
      final button = tester.widget<IconButton>(find.byType(IconButton));
      expect(button.onPressed, isNull);
      await tester.tap(find.byIcon(Icons.delete_outline),
          warnIfMissed: false);
      expect(taps, 0);
    });
  });
}
