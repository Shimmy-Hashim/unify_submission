import 'dart:async';

import 'package:flutter/material.dart';

import '../models/plaid_refresh_result.dart';
import '../services/plaid_service.dart';
import '../utils/formatters.dart';

/// The one place pull-to-refresh copy is decided, for Home, Savings and
/// Investments alike. Pure, so every status can be unit tested.
///
/// Pass the [result] of a completed call, or the [error] it threw. A timeout
/// is worded as a delay, not a failure: the server may well have finished.
String refreshMessage({PlaidRefreshResult? result, Object? error}) {
  if (error != null || result == null) {
    if (error is PlaidServiceException) {
      if (error.isNetwork) {
        return 'Taking longer than expected. Pull to refresh again shortly.';
      }
      if (error.statusCode == 401) {
        return 'Your session has expired. Sign in again.';
      }
    }
    return "Couldn't refresh your accounts. Try again later.";
  }
  return switch (result.status) {
    PlaidRefreshStatus.refreshed => 'Updated',
    PlaidRefreshStatus.skippedRecent => 'Already up to date',
    PlaidRefreshStatus.noConnections => 'No connected accounts to refresh',
    PlaidRefreshStatus.partial => _partialMessage(result.failedItems),
    PlaidRefreshStatus.failed ||
    PlaidRefreshStatus.unknown =>
      "Couldn't refresh your accounts. Try again later.",
  };
}

/// "Couldn't refresh Tartan Bank. In Settings, disconnect it, then connect it
/// again." Disconnect first, because the app has no Plaid update mode and a
/// plain reconnect creates a duplicate Item. Two Items from one bank name the
/// bank once; several banks read "A, B and C" and take "them".
String _partialMessage(List<PlaidItemRefresh> failed) {
  final names = <String>[];
  for (final item in failed) {
    final name = item.institutionName ?? 'one of your banks';
    if (!names.contains(name)) names.add(name);
  }
  if (names.isEmpty) names.add('one of your banks');
  final joined = names.length == 1
      ? names.single
      : '${names.sublist(0, names.length - 1).join(', ')} and ${names.last}';
  final pronoun = names.length == 1 ? 'it' : 'them';
  return "Couldn't refresh $joined. In Settings, disconnect $pronoun, "
      'then connect $pronoun again.';
}

/// Pull-to-refresh for the account screens. Wraps [child] (which must be
/// scrollable with AlwaysScrollableScrollPhysics so short lists still pull),
/// runs [onRefresh], and reports the outcome in a SnackBar.
///
/// [onRefresh] returning null means "nothing was started" (the caller's own
/// guard declined), and shows nothing. A pull while one is already running
/// does nothing either.
class AccountsRefreshIndicator extends StatefulWidget {
  const AccountsRefreshIndicator({
    super.key,
    required this.onRefresh,
    required this.child,
  });

  final Future<PlaidRefreshResult?> Function() onRefresh;
  final Widget child;

  @override
  State<AccountsRefreshIndicator> createState() =>
      _AccountsRefreshIndicatorState();
}

class _AccountsRefreshIndicatorState extends State<AccountsRefreshIndicator> {
  bool _inFlight = false;

  Future<void> _refresh() async {
    if (_inFlight) return;
    _inFlight = true;
    String? message;
    try {
      final result = await widget.onRefresh();
      if (result != null) message = refreshMessage(result: result);
    } catch (e) {
      message = refreshMessage(error: e);
    } finally {
      _inFlight = false;
    }
    if (message == null || !mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) =>
      RefreshIndicator(onRefresh: _refresh, child: widget.child);
}

/// "Updated 5 min ago" for a hero card, from the newest last_synced on the
/// screen's Firestore stream. Renders nothing when [synced] is null.
///
/// Rebuilds itself every 30 s so a relative time does not sit at "just now"
/// for an hour on a screen nobody touches.
class UpdatedLabel extends StatefulWidget {
  const UpdatedLabel({super.key, required this.synced, this.style, this.now});

  final DateTime? synced;
  final TextStyle? style;

  /// Test hook for the current time; defaults to the device clock.
  final DateTime Function()? now;

  @override
  State<UpdatedLabel> createState() => _UpdatedLabelState();
}

class _UpdatedLabelState extends State<UpdatedLabel> {
  Timer? _ticker;

  @override
  void initState() {
    super.initState();
    _ticker = Timer.periodic(const Duration(seconds: 30), (_) {
      if (mounted && widget.synced != null) setState(() {});
    });
  }

  @override
  void dispose() {
    _ticker?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final synced = widget.synced;
    if (synced == null) return const SizedBox.shrink();
    return Text(
      updatedLabel(synced, now: (widget.now ?? DateTime.now)()),
      style: widget.style,
      maxLines: 1,
      overflow: TextOverflow.ellipsis,
    );
  }
}
