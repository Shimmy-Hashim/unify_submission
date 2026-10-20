import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../utils/constants.dart';
import 'firebase_service.dart';

/// Raised when the backend rejects or fails an upload; [message] is safe to
/// show directly to the user.
class InsuranceParseException implements Exception {
  InsuranceParseException(this.message);
  final String message;

  @override
  String toString() => message;
}

/// Raised when the backend refuses or fails a policy delete; [message] is safe
/// to show directly to the user. In every failure case the policy is still
/// there and the delete can simply be retried.
class InsuranceDeleteException implements Exception {
  InsuranceDeleteException(this.message);
  final String message;

  @override
  String toString() => message;
}

/// Client for the backend insurance parser endpoint (POST /insurance/parse).
///
/// The endpoint verifies the Firebase ID token, parses the PDF with Gemini,
/// writes the extracted fields to users/{uid}/insurancePolicies/{policyId}
/// and returns the written document. The PDF itself is never persisted
/// anywhere — the client holds it in memory only long enough to send it.
class InsuranceService {
  InsuranceService._();
  static final InsuranceService instance = InsuranceService._();

  String get _baseUrl => kBackendUrl;

  /// Uploads [bytes] as [filename] and returns the parsed policy document.
  /// Throws [InsuranceParseException] with a user-facing message on failure.
  Future<Map<String, dynamic>> uploadAndParse({
    required Uint8List bytes,
    required String filename,
  }) async {
    final token = await FirebaseService.instance.getIdToken();
    if (token == null) {
      throw InsuranceParseException('You are signed out. Sign in again.');
    }

    final request = http.MultipartRequest(
      'POST',
      Uri.parse('$_baseUrl/insurance/parse'),
    )
      ..headers['Authorization'] = 'Bearer $token'
      ..files.add(
        http.MultipartFile.fromBytes('file', bytes, filename: filename),
      );

    final http.Response response;
    try {
      // Gemini parsing with reflection retries can take a while on large
      // policies — allow well beyond a normal request timeout.
      final streamed =
          await request.send().timeout(const Duration(minutes: 3));
      response = await http.Response.fromStream(streamed);
    } on InsuranceParseException {
      rethrow;
    } catch (_) {
      throw InsuranceParseException(
        'Could not reach the Unify server. Check that the backend is '
        'running, then try again.',
      );
    }

    if (response.statusCode == 200) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    }
    throw InsuranceParseException(_errorMessage(response));
  }

  /// Deletes one of the signed-in user's policies (DELETE /insurance/{id}).
  ///
  /// The backend authorises this against the caller's own ID token — the
  /// policy_id only selects a document beneath the caller's own subtree, so a
  /// foreign id comes back 404 rather than touching anything. The backend also
  /// recomputes the score before answering, which is why this allows longer
  /// than a plain read.
  ///
  /// Nothing to prune locally on success: Insurance and Home both stream
  /// insurancePolicies, so the card disappears once the delete lands.
  Future<Map<String, dynamic>> deletePolicy(String policyId) async {
    final token = await FirebaseService.instance.getIdToken();
    if (token == null) {
      throw InsuranceDeleteException('You are signed out. Sign in again.');
    }
    final http.Response response;
    try {
      response = await http.delete(
        Uri.parse('$_baseUrl/insurance/${Uri.encodeComponent(policyId)}'),
        headers: {'Authorization': 'Bearer $token'},
      ).timeout(const Duration(seconds: 60));
    } catch (_) {
      throw InsuranceDeleteException(
        'Could not reach the Unify server. The policy was not deleted. '
        'Check your connection, then try again.',
      );
    }
    if (response.statusCode == 200) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    }
    throw InsuranceDeleteException(_errorMessage(
      response,
      fallback: 'The policy was not deleted (HTTP ${response.statusCode}). '
          'Please try again.',
    ));
  }

  static String _errorMessage(http.Response response, {String? fallback}) {
    // FastAPI error bodies are {"detail": "..."} — already user-appropriate
    // wording for 400/413/502/503.
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
    return fallback ??
        'Upload failed (HTTP ${response.statusCode}). Please try again.';
  }
}
