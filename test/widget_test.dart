// Widget tests for auth/onboarding require Firebase to be faked
// (e.g. fake_cloud_firestore + firebase_auth_mocks) — deferred until the
// screen implementations are built out. This placeholder keeps `flutter test`
// green in the meantime.
import 'package:flutter_test/flutter_test.dart';

import 'package:unify/models/app_user.dart';

void main() {
  test('AppUser.age computes from date_of_birth', () {
    final now = DateTime.now();
    final user = AppUser(
      uid: 'u1',
      name: 'Test',
      email: 't@example.com',
      dateOfBirth: DateTime(now.year - 25, now.month, now.day),
      dependants: 0,
    );
    expect(user.age, 25);
  });
}
