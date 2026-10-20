import 'dart:convert';

import 'package:http/http.dart' as http;

import '../utils/constants.dart';
import 'firebase_service.dart';

/// Raised when DELETE /account fails; [message] is safe to show directly to
/// the user (same contract as PlaidServiceException).
class AccountServiceException implements Exception {
  AccountServiceException(this.message);
  final String message;

  @override
  String toString() => message;
}

/// Client for the account-lifecycle endpoints. Currently one:
///   DELETE /account  → erases every Firestore document for the user, then
///                      the Firebase Auth record
///
/// There is no uid parameter anywhere here on purpose — the backend derives it
/// from the verified ID token, so this client cannot ask for anyone else's
/// account to be deleted even if it tried.
class AccountService {
  AccountService._();
  static final AccountService instance = AccountService._();

  String get _baseUrl => kBackendUrl;

  /// Deletes the signed-in user's account. Returns the backend's summary of
  /// what was removed.
  ///
  /// The caller MUST have reauthenticated immediately before this — see
  /// DeleteAccountScreen. The two-minute timeout matches the exchange-token
  /// call: the cascade walks every transactions subcollection the user owns,
  /// which is by far the largest thing this app deletes.
  Future<Map<String, dynamic>> deleteAccount() async {
    final token = await FirebaseService.instance.getIdToken();
    if (token == null) {
      throw AccountServiceException('You are signed out. Sign in again.');
    }
    final http.Response response;
    try {
      response = await http.delete(
        Uri.parse('$_baseUrl/account'),
        headers: {
          'Authorization': 'Bearer $token',
          'Content-Type': 'application/json',
        },
      ).timeout(const Duration(minutes: 2));
    } catch (_) {
      throw AccountServiceException(
        'Could not reach the Unify server. Your account was not deleted. '
        'Check that the backend is running, then try again.',
      );
    }
    if (response.statusCode == 200) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    }
    throw AccountServiceException(_errorMessage(response));
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
    return 'Your account was not deleted (HTTP ${response.statusCode}). '
        'Please try again.';
  }
}
