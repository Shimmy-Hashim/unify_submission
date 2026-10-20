import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:unify/models/app_user.dart';
import 'package:unify/screens/settings/edit_profile_screen.dart';

/// Edit Profile must never save part of the form.
///
/// No Firebase app exists under flutter_test, so every FirebaseService call
/// throws. The screen's one-shot income read therefore fails on its own,
/// which is exactly the state under test, with no fake Firestore needed. It
/// also means any attempted write would throw and surface as the "Could not
/// save profile" SnackBar, so that SnackBar's absence proves no write was
/// attempted.
void main() {
  final user = AppUser(
    uid: 'u1',
    name: 'Test User',
    email: 'test@example.com',
    dateOfBirth: DateTime(1995, 6, 15),
    dependants: 1,
  );

  testWidgets('income failed to load: Save writes nothing and says why',
      (tester) async {
    await tester.pumpWidget(MaterialApp(home: EditProfileScreen(user: user)));
    await tester.pumpAndSettle();

    // The failed read is already showing its hint in the income field.
    expect(find.text('Could not load. Try again later'), findsOneWidget);

    await tester.tap(find.text('Save Changes'));
    await tester.pumpAndSettle();

    expect(find.textContaining('Could not save profile'), findsNothing);
    // The hint, plus the same text as the field's validation error.
    expect(find.text('Could not load. Try again later'), findsNWidgets(2));
    // Still on the form: a successful save pops the screen.
    expect(find.text('Save Changes'), findsOneWidget);
  });
}
