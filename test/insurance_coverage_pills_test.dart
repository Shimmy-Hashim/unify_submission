import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:unify/models/insurance_policy.dart';
import 'package:unify/screens/insurance_screen.dart';

/// The Total Coverage card's three header pills.
///
/// Life and CI say "On file", not "Covered", when a policy exists but no sum
/// assured was read: the Coverage Gaps cards judge those two by amount, so
/// "Covered" would sit above a card saying "Gap". Accident has no gap card and
/// keeps "Covered".
void main() {
  InsurancePolicy policy(String type, {double? sum}) => InsurancePolicy(
        policyId: '$type-$sum',
        policyName: 'Test Plan',
        insurer: 'Prudential',
        policyType: type,
        sumAssuredSgd: sum,
      );

  Future<void> pumpCard(WidgetTester tester, List<InsurancePolicy> policies,
      {double width = 400}) async {
    tester.view.physicalSize = Size(width, 800);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(MaterialApp(
      home: Scaffold(
        body: TotalCoverageCard(
          summary: CoverageSummary.fromPolicies(policies),
        ),
      ),
    ));
  }

  /// The value text under a pill's label, read from the same column.
  String pillValue(WidgetTester tester, String label) {
    final column = find.ancestor(
      of: find.text(label),
      matching: find.byType(Column),
    ).first;
    final texts = tester
        .widgetList<Text>(
            find.descendant(of: column, matching: find.byType(Text)))
        .map((t) => t.data)
        .toList();
    expect(texts.first, label);
    return texts[1]!;
  }

  testWidgets('no policies: every pill says "Not covered"', (tester) async {
    await pumpCard(tester, const []);
    expect(pillValue(tester, 'Life Cover'), 'Not covered');
    expect(pillValue(tester, 'Accident'), 'Not covered');
    expect(pillValue(tester, 'Critical Illness'), 'Not covered');
  });

  testWidgets('life and CI with a null sum say "On file"', (tester) async {
    await pumpCard(tester, [
      policy('life'),
      policy('critical_illness'),
    ]);
    expect(pillValue(tester, 'Life Cover'), 'On file');
    expect(pillValue(tester, 'Critical Illness'), 'On file');
    expect(find.text('Covered'), findsNothing);
  });

  testWidgets('life and CI with a zero sum say "On file"', (tester) async {
    await pumpCard(tester, [
      policy('life', sum: 0),
      policy('critical_illness', sum: 0),
    ]);
    expect(pillValue(tester, 'Life Cover'), 'On file');
    expect(pillValue(tester, 'Critical Illness'), 'On file');
  });

  testWidgets('a positive sum shows the amount', (tester) async {
    await pumpCard(tester, [
      policy('life', sum: 500000),
      policy('critical_illness', sum: 150000),
      policy('personal_accident', sum: 20000),
    ]);
    expect(pillValue(tester, 'Life Cover'), '\$500,000');
    expect(pillValue(tester, 'Critical Illness'), '\$150,000');
    expect(pillValue(tester, 'Accident'), '\$20,000');
  });

  testWidgets('a null sum beside a positive one still shows the amount',
      (tester) async {
    await pumpCard(tester, [
      policy('life'),
      policy('life', sum: 300000),
    ]);
    expect(pillValue(tester, 'Life Cover'), '\$300,000');
  });

  testWidgets('Accident with a policy and no sum still says "Covered"',
      (tester) async {
    await pumpCard(tester, [policy('personal_accident')]);
    expect(pillValue(tester, 'Accident'), 'Covered');
    await pumpCard(tester, [policy('hospitalisation')]);
    expect(pillValue(tester, 'Accident'), 'Covered');
  });

  testWidgets('"On file" fits a narrow phone without overflow',
      (tester) async {
    await pumpCard(
      tester,
      [policy('life'), policy('personal_accident'), policy('critical_illness')],
      width: 320,
    );
    expect(tester.takeException(), isNull);
    expect(pillValue(tester, 'Life Cover'), 'On file');
    expect(pillValue(tester, 'Critical Illness'), 'On file');
  });
}
