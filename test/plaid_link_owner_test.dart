import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

/// The duplicate-snackbar bug was structural, not logical: Savings,
/// Investments and Settings each subscribed to the process-global
/// `PlaidLink.onSuccess`, so one connection produced one snackbar per mounted
/// screen. The fix is an invariant — PlaidLinkService is the only subscriber —
/// and the cheapest way to keep it is to assert that nothing else reaches for
/// the SDK. There is no widget-level test for this: `PlaidLink` is a static
/// plugin surface over a platform channel, so the real behaviour can only be
/// exercised on a device.
void main() {
  test('PlaidLinkService is the only file that touches the Plaid Link SDK',
      () {
    const owner = 'lib\\services\\plaid_link_service.dart';
    final offenders = <String>[];

    for (final entity in Directory('lib').listSync(recursive: true)) {
      if (entity is! File || !entity.path.endsWith('.dart')) continue;
      if (entity.path.replaceAll('/', '\\').endsWith(owner)) continue;
      if (entity.readAsStringSync().contains('package:plaid_flutter')) {
        offenders.add(entity.path);
      }
    }

    expect(
      offenders,
      isEmpty,
      reason: 'These files import plaid_flutter directly. Listening to '
          'PlaidLink.onSuccess outside PlaidLinkService reintroduces the '
          'duplicate success-snackbar bug — go through '
          'PlaidLinkService.connect() instead.',
    );
  });
}
