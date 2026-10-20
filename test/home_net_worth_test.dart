import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:unify/models/user_financial_profile.dart';
import 'package:unify/screens/home_screen.dart';
import 'package:unify/services/app_preferences.dart';
import 'package:unify/utils/app_styles.dart';
import 'package:unify/utils/formatters.dart';

/// user@example.com's real financialProfile values (read from Firestore;
/// assets − debts reconcile to net worth to the cent).
final _withDebt = UserFinancialProfile.fromMap({
  'monthly_income_sgd': 5000,
  'net_worth_sgd': -116876.0,
  'total_assets_sgd': 264106.18,
  'total_debts_sgd': 380982.18,
});

/// Holds the selected tab the way _NetWorthCard's State does, and rebuilds
/// on Hide Balances the way MainShell does.
class _Harness extends StatefulWidget {
  const _Harness(this.profile);

  final UserFinancialProfile? profile;

  @override
  State<_Harness> createState() => _HarnessState();
}

class _HarnessState extends State<_Harness> {
  WealthFigure selected = WealthFigure.netWorth;

  @override
  Widget build(BuildContext context) => ListenableBuilder(
        listenable: AppPreferences.instance,
        builder: (context, _) => NetWorthCardBody(
          profile: widget.profile,
          selected: selected,
          onSelect: (f) => setState(() => selected = f),
        ),
      );
}

Future<void> _pump(WidgetTester tester, UserFinancialProfile? profile) async {
  tester.view.physicalSize = const Size(1200, 800);
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.reset);
  addTearDown(() => AppPreferences.instance.setHideBalances(false));
  await tester.pumpWidget(MaterialApp(
    theme: AppTheme.lightTheme,
    home: Scaffold(body: SizedBox(width: 1100, child: _Harness(profile))),
  ));
}

Color _tabFill(WidgetTester tester, String label) {
  final box = tester.widget<Container>(find
      .ancestor(of: find.text(label), matching: find.byType(Container))
      .first);
  return (box.decoration! as BoxDecoration).color!;
}

void main() {
  testWidgets('defaults to Net Worth: lime on Net Worth, real value',
      (tester) async {
    await _pump(tester, _withDebt);
    expect(find.text('-\$116,876'), findsOneWidget);
    expect(find.text('Savings + investments − debts'), findsOneWidget);
    expect(find.text('Total Net Worth'), findsNothing); // tabs replace title
    expect(_tabFill(tester, 'Net Worth'), AppColors.limeAccent);
    expect(_tabFill(tester, 'Total Assets'), isNot(AppColors.limeAccent));
  });

  testWidgets('tapping swaps the figure, the caption and the lime',
      (tester) async {
    await _pump(tester, _withDebt);

    await tester.tap(find.text('Total Assets'));
    await tester.pump();
    expect(find.text('\$264,106'), findsOneWidget);
    expect(find.text('-\$116,876'), findsNothing);
    expect(find.text('Savings + investments, before \$380,982 in debts'),
        findsOneWidget);
    expect(_tabFill(tester, 'Total Assets'), AppColors.limeAccent);
    expect(_tabFill(tester, 'Net Worth'), isNot(AppColors.limeAccent));

    await tester.tap(find.text('Net Worth'));
    await tester.pump();
    expect(find.text('-\$116,876'), findsOneWidget);
    expect(_tabFill(tester, 'Net Worth'), AppColors.limeAccent);
  });

  testWidgets('hide balances masks the figure and debts on both tabs',
      (tester) async {
    await _pump(tester, _withDebt);
    await AppPreferences.instance.setHideBalances(true);
    await tester.pump();
    expect(find.text(kMaskedAmount), findsOneWidget);
    expect(find.textContaining('116'), findsNothing);

    await tester.tap(find.text('Total Assets'));
    await tester.pump();
    expect(find.text(kMaskedAmount), findsOneWidget);
    expect(find.text('Savings + investments, before $kMaskedAmount in debts'),
        findsOneWidget);
    expect(find.textContaining('264'), findsNothing);
    expect(find.textContaining('380'), findsNothing);

    await AppPreferences.instance.setHideBalances(false);
    await tester.pump();
    expect(find.text('\$264,106'), findsOneWidget);
  });

  testWidgets('profile predating total_assets_sgd: dash, not a false \$0',
      (tester) async {
    await _pump(
        tester, UserFinancialProfile.fromMap({'net_worth_sgd': 1500.0}));
    expect(find.text('\$1,500'), findsOneWidget);
    await tester.tap(find.text('Total Assets'));
    await tester.pump();
    expect(find.text('-'), findsOneWidget);
    expect(find.text('Savings + investments, before debts'), findsOneWidget);
  });

  testWidgets('no profile yet: dash on either tab', (tester) async {
    await _pump(tester, null);
    expect(find.text('-'), findsOneWidget);
    await tester.tap(find.text('Total Assets'));
    await tester.pump();
    expect(find.text('-'), findsOneWidget);
  });
}
