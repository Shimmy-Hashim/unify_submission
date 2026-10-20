import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:unify/screens/auth/auth_ui.dart';
import 'package:unify/utils/app_styles.dart';

/// [UnifyTextField] and [DependantsStepper] are shared by two flows with
/// opposite theming rules: Settings (Edit Profile, Password & Security, Delete
/// Account) follows the user's preference, while the auth flow is light-only
/// by design and hardcodes a light background.
///
/// They used to hardcode black, which made them correct for auth and
/// black-on-black in Settings' dark mode. They now resolve against the ambient
/// theme, which fixes Settings but means the auth screens MUST pin their own
/// subtree to [AppTheme.lightTheme] — otherwise a user who selected Dark and
/// signed out would get white labels on a light screen.
///
/// These tests pin down both halves of that arrangement, because a screenshot
/// of the auth flow costs a sign-out to obtain.
void main() {
  Color labelColorOf(WidgetTester tester, String label) =>
      tester.widget<Text>(find.text(label)).style!.color!;

  Widget host({required ThemeData theme, Widget Function(Widget)? wrapper}) {
    const field = UnifyTextField(label: 'Name', hint: 'Enter Full Name');
    return MaterialApp(
      theme: theme,
      home: Scaffold(
        body: wrapper == null ? field : wrapper(field),
      ),
    );
  }

  group('UnifyTextField follows the ambient theme', () {
    testWidgets('light theme renders the Figma black label', (tester) async {
      await tester.pumpWidget(host(theme: AppTheme.lightTheme));
      expect(labelColorOf(tester, 'Name'), Colors.black);
    });

    testWidgets('dark theme renders a white label', (tester) async {
      await tester.pumpWidget(host(theme: AppTheme.darkTheme));
      expect(labelColorOf(tester, 'Name'), Colors.white);
    });
  });

  group('the auth flow stays light regardless of preference', () {
    // The exact regression the light pin exists to prevent: dark app, auth
    // subtree, light background — the label has to stay readable.
    testWidgets('a light-pinned subtree inside a dark app keeps black ink',
        (tester) async {
      await tester.pumpWidget(host(
        theme: AppTheme.darkTheme,
        wrapper: (child) => Theme(data: AppTheme.lightTheme, child: child),
      ));
      expect(labelColorOf(tester, 'Name'), Colors.black);
    });

    testWidgets('the two ThemeDatas actually differ in brightness',
        (tester) async {
      expect(AppTheme.lightTheme.brightness, Brightness.light);
      expect(AppTheme.darkTheme.brightness, Brightness.dark);
    });
  });

  group('DependantsStepper follows the ambient theme', () {
    Widget stepper(ThemeData theme) => MaterialApp(
          theme: theme,
          home: Scaffold(
            body: DependantsStepper(value: 2, onChanged: (_) {}),
          ),
        );

    testWidgets('light keeps black ink on label, value and controls',
        (tester) async {
      await tester.pumpWidget(stepper(AppTheme.lightTheme));
      expect(labelColorOf(tester, 'Dependants'), Colors.black);
      expect(labelColorOf(tester, '2'), Colors.black);
      expect(
        tester.widget<Icon>(find.byIcon(Icons.remove)).color,
        Colors.black,
      );
    });

    testWidgets('dark lifts label, value and controls to white',
        (tester) async {
      await tester.pumpWidget(stepper(AppTheme.darkTheme));
      expect(labelColorOf(tester, 'Dependants'), Colors.white);
      expect(labelColorOf(tester, '2'), Colors.white);
      expect(
        tester.widget<Icon>(find.byIcon(Icons.add)).color,
        Colors.white,
      );
    });
  });
}
