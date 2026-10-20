import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/plaid_refresh_result.dart';
import '../utils/constants.dart';
import 'firebase_service.dart';

/// Raised when a Plaid backend call fails; [message] is safe to show
/// directly to the user (same pattern as InsuranceParseException).
class PlaidServiceException implements Exception {
  PlaidServiceException(this.message, {this.statusCode, this.isNetwork = false});
  final String message;

  /// The HTTP status when the server answered with an error, else null.
  final int? statusCode;

  /// True when no answer arrived at all: no connection, or the request timed
  /// out. A timeout says nothing about whether the server's work failed.
  final bool isNetwork;

  @override
  String toString() => message;
}

/// Client for the backend Plaid endpoints (Phase 1):
///   POST /plaid/link-token      → link_token for launching Plaid Link
///   POST /plaid/exchange-token  → stores the Item + accounts, runs the
///                                 initial transaction sync server-side
///
/// The Plaid access_token never reaches this client — the backend keeps it
/// in the (client-unreadable) plaidItems collection.
class PlaidService {
  PlaidService._();
  static final PlaidService instance = PlaidService._();

  String get _baseUrl => kBackendUrl;

  Future<Map<String, dynamic>> _post(
    String path, {
    Map<String, dynamic>? body,
    Duration timeout = const Duration(seconds: 30),
  }) async {
    final token = await FirebaseService.instance.getIdToken();
    if (token == null) {
      throw PlaidServiceException('You are signed out. Sign in again.');
    }
    final http.Response response;
    try {
      response = await http
          .post(
            Uri.parse('$_baseUrl$path'),
            headers: {
              'Authorization': 'Bearer $token',
              'Content-Type': 'application/json',
            },
            body: jsonEncode(body ?? {}),
          )
          .timeout(timeout);
    } catch (_) {
      throw PlaidServiceException(
        'Could not reach the Unify server. Check that the backend is '
        'running, then try again.',
        isNetwork: true,
      );
    }
    if (response.statusCode == 200) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    }
    throw PlaidServiceException(_errorMessage(response),
        statusCode: response.statusCode);
  }

  /// Brings every connected Item up to date without reconnecting: new
  /// transactions, current balances and holdings, then the derived profile
  /// fields and the score, all server-side in one call (POST /plaid/refresh).
  /// The results then arrive through the screens' existing Firestore streams.
  ///
  /// 90 s because the server works through each Item in turn (about 4 s per
  /// Item measured locally). Throws [PlaidServiceException]; a timeout comes
  /// back with [PlaidServiceException.isNetwork] set.
  Future<PlaidRefreshResult> refresh() async {
    final body =
        await _post('/plaid/refresh', timeout: const Duration(seconds: 90));
    return PlaidRefreshResult.fromJson(body);
  }

  /// Returns a short-lived link_token for launching Plaid Link.
  Future<String> createLinkToken() async {
    final body = await _post('/plaid/link-token');
    final linkToken = body['link_token'];
    if (linkToken is! String || linkToken.isEmpty) {
      throw PlaidServiceException(
          'The server returned an invalid link token. Try again.');
    }
    return linkToken;
  }

  /// Deduplicates [exchangePublicToken] by public_token. PlaidLink.onSuccess
  /// is a single global stream, and when Savings, Investments and Settings
  /// each listened to it directly, one successful Link session fired three
  /// exchanges and three server-side initial syncs (verified in the backend
  /// log). PlaidLinkService is now the stream's only subscriber, so it calls
  /// this once per session — but the dedup stays as the guarantee at the
  /// network layer, independent of how many callers exist above it. Plaid
  /// public_tokens are single-use, so caching for the process lifetime is
  /// safe.
  final Map<String, Future<Map<String, dynamic>>> _exchanges = {};

  /// Exchanges the public_token from Plaid Link. The backend stores the
  /// Item + banking accounts and runs the initial transaction sync before
  /// returning, so this can take several seconds. Concurrent or repeat calls
  /// for the same token share one request.
  Future<Map<String, dynamic>> exchangePublicToken(String publicToken) {
    final existing = _exchanges[publicToken];
    if (existing != null) return existing;
    final request = _post(
      '/plaid/exchange-token',
      body: {'public_token': publicToken},
      timeout: const Duration(minutes: 2),
    );
    _exchanges[publicToken] = request;
    // A failed exchange is worth retrying; a successful one never is.
    request.catchError((Object error) {
      _exchanges.remove(publicToken);
      throw error;
    });
    return request;
  }

  /// Revokes one Plaid Item at Plaid and deletes everything stored under it
  /// (its accounts, their transactions, and its investment accounts).
  ///
  /// The backend authorises this against the caller's own ID token — the
  /// item_id only selects a document beneath the caller's own subtree, so a
  /// foreign item_id comes back 404 rather than touching anything. Nothing
  /// here needs to (or can) check ownership client-side.
  ///
  /// Returns the backend's summary. Throws [PlaidServiceException] with a
  /// user-facing message on failure; in every failure case the connection is
  /// left intact and the call can simply be retried.
  Future<Map<String, dynamic>> disconnectItem(String itemId) {
    return _post(
      '/plaid/disconnect-item',
      body: {'item_id': itemId},
      // Revocation plus a scoped cascade plus a score refresh; slower than a
      // plain read, nowhere near the exchange path's initial full sync.
      timeout: const Duration(seconds: 60),
    );
  }

  static String _errorMessage(http.Response response) {
    try {
      final detail =
          (jsonDecode(response.body) as Map<String, dynamic>)['detail'];
      if (detail is String && detail.isNotEmpty) return detail;
    } catch (_) {
      // fall through to the generic message
    }
    if (response.statusCode == 401) {
      return 'Your session has expired. Sign in again.';
    }
    return 'Bank connection failed (HTTP ${response.statusCode}). '
        'Please try again.';
  }
}
