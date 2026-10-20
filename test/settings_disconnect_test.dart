import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:unify/screens/settings_screen.dart';

/// Disconnect's confirmation gate.
///
/// The Connections section itself is backed by live Firestore streams and is
/// not widget-tested (the same reason savings/investments screens are covered
/// through their models instead). What IS testable, and what actually matters
/// for a destructive action, is the gate in front of it: the disconnect must
/// not be reachable without an explicit confirmation, and every other way out
/// of the dialog must leave the caller doing nothing.
///
/// `showDisconnectConfirmation` returning anything other than `true` is what
/// the caller checks (`if (confirmed != true) return;`), so these tests pin
/// exactly that contract.
void main() {
  /// Pumps a button that opens the dialog and records what it resolved to,
  /// plus how many times a stand-in "disconnect" actually ran.
  Future<Map<String, dynamic>> openDialog(
    WidgetTester tester, {
    required String institution,
    required Future<void> Function(WidgetTester) act,
  }) async {
    bool? resolved;
    var disconnectCalls = 0;

    await tester.pumpWidget(MaterialApp(
      home: Scaffold(
        body: Builder(
          builder: (context) => ElevatedButton(
            onPressed: () async {
              resolved = await showDisconnectConfirmation(context, institution);
              // Mirrors the real call site's guard exactly.
              if (resolved != true) return;
              disconnectCalls++;
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

    return {'resolved': resolved, 'calls': disconnectCalls};
  }

  testWidgets('the dialog names the institution and what will be lost',
      (tester) async {
    await openDialog(
      tester,
      institution: 'Tartan Bank',
      act: (t) async {
        // Assert while the dialog is still up, then close it.
        expect(find.text('Disconnect Tartan Bank?'), findsOneWidget);

        final body = t.widget<Text>(find.byWidgetPredicate((w) =>
            w is Text && (w.data ?? '').contains('remove all accounts')));
        final text = body.data!;
        expect(text, contains('Tartan Bank'));
        expect(text, contains('transaction history'));
        expect(text, contains('cannot be undone'));
        // The user is told this is recoverable by reconnecting — the reason
        // this action does not carry Delete Account's typed confirmation.
        expect(text, contains('reconnect'));

        expect(find.text('Cancel'), findsOneWidget);
        expect(find.text('Disconnect'), findsOneWidget);

        await t.tap(find.text('Cancel'));
      },
    );
  });

  testWidgets('Cancel resolves false and does NOT disconnect', (tester) async {
    final result = await openDialog(
      tester,
      institution: 'Tartan Bank',
      act: (t) async => t.tap(find.text('Cancel')),
    );
    expect(result['resolved'], isFalse);
    expect(result['calls'], 0);
  });

  testWidgets('dismissing by tapping outside resolves null and does NOT '
      'disconnect', (tester) async {
    final result = await openDialog(
      tester,
      institution: 'Tartan Bank',
      // A barrier tap is the easiest accidental dismissal, and it returns
      // null rather than false — the case a `== false` check would miss.
      act: (t) async => t.tapAt(const Offset(10, 10)),
    );
    expect(result['resolved'], isNull);
    expect(result['calls'], 0);
  });

  testWidgets('only an explicit Disconnect tap proceeds', (tester) async {
    final result = await openDialog(
      tester,
      institution: 'Tartan Bank',
      act: (t) async => t.tap(find.text('Disconnect')),
    );
    expect(result['resolved'], isTrue);
    expect(result['calls'], 1);
  });
}
