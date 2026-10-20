import 'dart:async';

import 'package:plaid_flutter/plaid_flutter.dart';

import 'plaid_service.dart';

/// How one [PlaidLinkService.connect] call ended.
enum PlaidConnectStatus {
  /// Link succeeded and the backend exchange completed.
  connected,

  /// The user backed out of Link without an error. Show nothing.
  cancelled,

  /// Link or the exchange failed. See [PlaidConnectResult.message].
  failed,

  /// Another screen already had a Link session open. Show nothing — the
  /// screen that started that session owns its feedback.
  busy,
}

/// The outcome of one Link session, returned to whichever screen started it.
class PlaidConnectResult {
  const PlaidConnectResult.connected({
    required this.institutionName,
    required this.accountCount,
  })  : status = PlaidConnectStatus.connected,
        message = null;

  const PlaidConnectResult.cancelled()
      : status = PlaidConnectStatus.cancelled,
        institutionName = null,
        accountCount = 0,
        message = null;

  /// [message] is safe to show directly. It is null when Plaid ended the
  /// session with an error that carried no text — the calling screen supplies
  /// its own wording in that case ("bank"/"broker"/"account").
  const PlaidConnectResult.failed([this.message])
      : status = PlaidConnectStatus.failed,
        institutionName = null,
        accountCount = 0;

  const PlaidConnectResult.busy()
      : status = PlaidConnectStatus.busy,
        institutionName = null,
        accountCount = 0,
        message = null;

  final PlaidConnectStatus status;

  /// Null unless [status] is [PlaidConnectStatus.connected], and possibly null
  /// even then if the backend did not name the institution.
  final String? institutionName;

  /// Banking accounts stored by the exchange. 0 unless connected.
  final int accountCount;

  final String? message;
}

/// Sole owner of the Plaid Link lifecycle.
///
/// `PlaidLink.onSuccess`/`onExit` are process-global broadcast streams: an
/// event from one Link session is delivered to every listener. Savings,
/// Investments and Settings all offer a Connect action, and when each screen
/// subscribed for itself, one connection produced two or three identical
/// success snackbars from the screens sitting alive in the IndexedStack.
///
/// This service subscribes once, for the process lifetime, and hands the
/// outcome back through a Future to the single screen that called [connect].
/// Screens no longer touch `PlaidLink` at all, so only the initiating screen
/// can show feedback.
///
/// The exchange itself still goes through [PlaidService.exchangePublicToken],
/// which keeps its own per-public_token deduplication.
class PlaidLinkService {
  PlaidLinkService._();
  static final PlaidLinkService instance = PlaidLinkService._();

  StreamSubscription<LinkSuccess>? _successSub;
  StreamSubscription<LinkExit>? _exitSub;

  /// The in-flight session, or null when no Link flow is running. Also the
  /// "is a session already open" flag — only one Link session can exist at a
  /// time, since the native SDK presents a single full-screen flow.
  Completer<PlaidConnectResult>? _pending;

  /// Creates a link token, presents Plaid Link, and completes once the user
  /// finishes or backs out. Never throws — every failure comes back as
  /// [PlaidConnectStatus.failed] so callers have one place to handle results.
  Future<PlaidConnectResult> connect() async {
    if (_pending != null) return const PlaidConnectResult.busy();
    _ensureSubscribed();
    final session = Completer<PlaidConnectResult>();
    _pending = session;
    try {
      final linkToken = await PlaidService.instance.createLinkToken();
      await PlaidLink.open(
        configuration: LinkTokenConfiguration(token: linkToken),
      );
    } on PlaidServiceException catch (e) {
      _resolve(session, PlaidConnectResult.failed(e.message));
    } catch (_) {
      _resolve(
        session,
        const PlaidConnectResult.failed(
            'Could not open Plaid Link. Please try again.'),
      );
    }
    // Link is up; _onSuccess/_onExit resolves the session when it closes
    // (unless it already failed above and never got that far).
    return session.future;
  }

  /// Subscribes on first use and never cancels: the streams are global and
  /// this service outlives every screen, so there is nothing to tear down.
  void _ensureSubscribed() {
    _successSub ??= PlaidLink.onSuccess.listen(_onSuccess);
    _exitSub ??= PlaidLink.onExit.listen(_onExit);
  }

  Future<void> _onSuccess(LinkSuccess event) async {
    // Claim the session before awaiting: the exchange takes seconds, and a
    // trailing onExit must not resolve this as a cancellation meanwhile.
    final session = _pending;
    _pending = null;

    PlaidConnectResult result;
    try {
      // Still exchanged even with no session pending (nothing to report to,
      // but the token is single-use — dropping it would lose the connection).
      final body =
          await PlaidService.instance.exchangePublicToken(event.publicToken);
      result = PlaidConnectResult.connected(
        institutionName: body['institution_name'] as String?,
        accountCount: (body['accounts'] as List?)?.length ?? 0,
      );
    } on PlaidServiceException catch (e) {
      result = PlaidConnectResult.failed(e.message);
    } catch (_) {
      result = const PlaidConnectResult.failed(
          'The connection could not be completed. Please try again.');
    }
    if (session != null) _resolve(session, result);
  }

  void _onExit(LinkExit event) {
    final session = _pending;
    if (session == null) return;
    // A plain cancellation carries no error — 'null' is what the SDK reports
    // for the no-error case.
    final error = event.error;
    if (error == null || error.code == 'null') {
      _resolve(session, const PlaidConnectResult.cancelled());
      return;
    }
    final message = error.displayMessage ?? error.message;
    _resolve(
      session,
      PlaidConnectResult.failed(message.isEmpty ? null : message),
    );
  }

  /// Completes [session] and releases ownership, so a later event for the same
  /// flow is ignored rather than resolving it twice.
  void _resolve(
      Completer<PlaidConnectResult> session, PlaidConnectResult result) {
    if (identical(_pending, session)) _pending = null;
    if (!session.isCompleted) session.complete(result);
  }
}
