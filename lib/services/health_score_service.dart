import 'dart:convert';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import '../models/home_score_state.dart';
import '../models/score_history.dart';
import '../models/score_profile.dart';
import '../utils/constants.dart';
import 'firebase_service.dart';

/// Raised when POST /score/refresh fails; [message] is safe to show directly to
/// the user (same pattern as PlaidServiceException / InsuranceParseException).
class HealthScoreException implements Exception {
  HealthScoreException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// The Phase 3 Financial Health Score, for the client.
///
/// READS COME FROM FIRESTORE, NOT FROM GET /score.
/// `GET /score` is `read_cached_score()`, which reads exactly four fields off
/// users/{uid}/financialProfile/profile and returns them — it computes nothing.
/// The client can read the same document directly, so the endpoint would only
/// add a network hop, an auth round trip and a polling problem. Streaming the
/// document instead means:
///   - every one of the four existing background triggers (the three Plaid sync
///     endpoints and /insurance/parse, all funnelling through
///     `refresh_score_nonfatal`) pushes its result to the screen the moment it
///     writes the cache, with no polling and no manual invalidation;
///   - the score stays consistent with the rest of the app, which is
///     StreamBuilder-on-Firestore everywhere (Savings, Investments, Insurance);
///   - the screen keeps working offline from Firestore's local cache.
/// The nested `ml_profile` map deserializes cleanly from a raw snapshot — every
/// value in it is a Firestore-native scalar, list, map or Timestamp (the tuple
/// in `score_breakdown` is already flattened to a list backend-side), so there
/// is nothing GET /score's JSON would decode that this cannot.
///
/// WRITES ARE THE ONE THING THE BACKEND MUST DO.
/// [requestRefresh] posts to /score/refresh purely to make the pipeline run.
/// Its HTTP response is deliberately discarded: the same call writes the cache,
/// so the recomputed result arrives through the stream above like any other
/// trigger's would. Reading it from the response as well would give the screen
/// two sources for one number.
class HealthScoreService {
  HealthScoreService._();

  /// Subclass hook so a test can stub [watch] / [requestRefresh] without
  /// Firebase or a running backend. Production code uses [instance].
  @visibleForTesting
  HealthScoreService.forTesting();

  static final HealthScoreService instance = HealthScoreService._();

  String get _baseUrl => kBackendUrl;

  /// The live score state for [uid].
  ///
  /// Errors are NOT converted to [ScoreFailed] here — the stream propagates
  /// them so a StreamBuilder can use `snapshot.hasError` normally.
  /// [HomeScoreController] is what turns them into a state.
  Stream<ScoreState> watch(String uid) => FirebaseService.instance
      .financialProfileDoc(uid)
      .snapshots()
      .map(stateFromSnapshot);

  /// How many daily snapshots [watchHistory] reads — GET /score/history's own
  /// default window (main.py's HISTORY_DEFAULT_LIMIT), so the stream and the
  /// endpoint agree on what "recent history" means.
  static const historyLimit = 90;

  /// The user's daily score snapshots, newest first — streamed for the same
  /// reasons [watch] streams instead of calling GET /score: GET /score/history
  /// is a plain read of a collection the client may read directly
  /// (firestore.rules: owner read-only), and a stream picks up the snapshot a
  /// refresh writes without any polling.
  Stream<List<ScoreHistoryPoint>> watchHistory(String uid) =>
      FirebaseService.instance
          .scoreHistory(uid)
          .orderBy('date', descending: true)
          .limit(historyLimit)
          .snapshots()
          .map((snap) => [
                for (final doc in snap.docs)
                  ScoreHistoryPoint.fromMap(doc.data()),
              ]);

  /// The document -> state mapping, exposed for testing without Firestore.
  ///
  /// The decision tree is the backend's, read back rather than re-derived:
  ///   no document              -> no_financial_profile (the same rule
  ///                               `refresh_score_profile` uses to decide
  ///                               whether to write at all)
  ///   no checklist_last_updated-> never computed (`read_cached_score`'s
  ///                               `computed: false`)
  ///   unavailable_reason set   -> whole-pipeline unavailable
  ///   overall_score null       -> partial (incomplete_domains)
  ///   otherwise                -> complete
  @visibleForTesting
  static ScoreState stateFromSnapshot(
      DocumentSnapshot<Map<String, dynamic>> snapshot) {
    if (!snapshot.exists) {
      return const ScoreUnavailable(ScoreUnavailableReason.noFinancialProfile);
    }
    return stateFromProfileData(snapshot.data() ?? const {});
  }

  @visibleForTesting
  static ScoreState stateFromProfileData(Map<String, dynamic> data) {
    // Both, not either: they are written together by `cache_payload`, so one
    // without the other is a half-written or pre-Phase-3 document, and reading
    // a score out of it would be reading a score that was never computed.
    if (data['checklist_last_updated'] == null || data['ml_profile'] == null) {
      return const ScoreNeverComputed();
    }

    final profile = ScoreProfile.fromProfileData(data);

    final unavailable = profile.unavailableReason;
    if (unavailable != null) return ScoreUnavailable(unavailable, profile);

    final score = profile.overallScore;
    final breakdown = profile.breakdown;
    // `compute_score_breakdown()` returns None unless all five of its inputs
    // are present, so a null score here is precisely the partial case. The two
    // are checked together because they are written together — one without the
    // other would mean the cache disagrees with itself.
    if (score == null || breakdown == null) return ScorePartial(profile);

    return ScoreComplete(profile, score, breakdown);
  }

  /// Triggers a server-side recompute (POST /score/refresh) and caches it.
  ///
  /// Returns nothing on purpose — see the class comment. Throws
  /// [HealthScoreException] with a user-facing message on failure.
  Future<void> requestRefresh(
      {Duration timeout = const Duration(seconds: 45)}) async {
    final token = await FirebaseService.instance.getIdToken();
    if (token == null) {
      throw HealthScoreException('You are signed out. Sign in again.');
    }

    final http.Response response;
    try {
      response = await http.post(
        Uri.parse('$_baseUrl/score/refresh'),
        headers: {
          'Authorization': 'Bearer $token',
          'Content-Type': 'application/json',
        },
        body: '{}',
      ).timeout(timeout);
    } catch (_) {
      throw HealthScoreException(
        'Could not reach the Unify server. Check that the backend is '
        'running, then try again.',
      );
    }

    // A user-data problem (no income, unsupported age, one connected domain)
    // is a 200 with a reason code, not an error — it lands in the cache and
    // comes back down the stream as the matching state. Only a real failure
    // reaches here.
    if (response.statusCode != 200) {
      throw HealthScoreException(_errorMessage(response));
    }
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
    return 'Could not refresh your score (HTTP ${response.statusCode}). '
        'Please try again.';
  }
}
