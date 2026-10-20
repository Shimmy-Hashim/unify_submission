import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:unify/utils/app_styles.dart';
import 'package:unify/widgets/collapsible_card.dart';

Widget _card(String title, {Key? key, bool enabled = true}) => CollapsibleCard(
      key: key,
      title: title,
      titleSize: 17,
      trailing: const Text('\$1'),
      enabled: enabled,
      body: Text('$title body'),
    );

Future<void> _pump(WidgetTester tester, List<Widget> cards) =>
    tester.pumpWidget(MaterialApp(
      theme: AppTheme.lightTheme,
      home: Scaffold(body: Column(children: cards)),
    ));

/// Open/closed state of the card titled [title].
CrossFadeState _fade(WidgetTester tester, String title) => tester
    .widget<AnimatedCrossFade>(find.descendant(
      of: find.ancestor(
          of: find.text(title), matching: find.byType(CollapsibleCard)),
      matching: find.byType(AnimatedCrossFade),
    ))
    .crossFadeState;

void main() {
  testWidgets('keyed: open state follows its card when one is inserted above',
      (tester) async {
    await _pump(tester, [
      _card('Savings', key: const ValueKey('savings')),
      _card('Investments', key: const ValueKey('investments')),
    ]);
    await tester.tap(find.text('Savings'));
    await tester.pumpAndSettle();

    await _pump(tester, [
      _card('Debts', key: const ValueKey('debts')),
      _card('Savings', key: const ValueKey('savings')),
      _card('Investments', key: const ValueKey('investments')),
    ]);
    await tester.pumpAndSettle();
    expect(_fade(tester, 'Debts'), CrossFadeState.showFirst);
    expect(_fade(tester, 'Savings'), CrossFadeState.showSecond);
    expect(_fade(tester, 'Investments'), CrossFadeState.showFirst);
  });

  testWidgets('why the keys exist: unkeyed, the open state stays in the slot',
      (tester) async {
    await _pump(tester, [_card('Savings'), _card('Investments')]);
    await tester.tap(find.text('Savings'));
    await tester.pumpAndSettle();

    await _pump(
        tester, [_card('Debts'), _card('Savings'), _card('Investments')]);
    await tester.pumpAndSettle();
    expect(_fade(tester, 'Debts'), CrossFadeState.showSecond); // wrong card
    expect(_fade(tester, 'Savings'), CrossFadeState.showFirst);
  });

  testWidgets('disabled: no tap handler, stays shut; reopens when re-enabled',
      (tester) async {
    await _pump(tester, [_card('Savings')]);
    await tester.tap(find.text('Savings'));
    await tester.pumpAndSettle();
    expect(_fade(tester, 'Savings'), CrossFadeState.showSecond);

    await _pump(tester, [_card('Savings', enabled: false)]);
    await tester.pumpAndSettle();
    expect(_fade(tester, 'Savings'), CrossFadeState.showFirst);
    expect(tester.widget<InkWell>(find.byType(InkWell)).onTap, isNull);

    await _pump(tester, [_card('Savings')]);
    await tester.pumpAndSettle();
    expect(_fade(tester, 'Savings'), CrossFadeState.showSecond);
  });
}
