import 'dart:async';

import 'package:flutter/foundation.dart';

import '../models/home_score_state.dart';
import '../models/plaid_refresh_result.dart';
import 'health_score_service.dart';
import 'plaid_service.dart';

/// Holds the Home screen's score state: the live Firestore stream, plus the
/// pull-to-refresh action layered over it.
///
/// A ChangeNotifier rather than a bare StreamBuilder because the two halves
/// have to be combined. The stream alone cannot say "a manual refresh is in
/// flight", and a `_refreshing` bool in the widget alone cannot survive the
/// rebuild the next snapshot causes. Owning both here means Slice 2's UI reads
/// one immutable [HomeScoreState] and never sequences anything itself.
///
/// Refresh is a fallback, not the normal path. Reads are read-only by design —
/// the three Plaid sync endpoints and /insurance/parse each recompute the score
/// server-side and the result arrives through the stream on its own.
class HomeScoreController extends ChangeNotifier {
  HomeScoreController({
    required this.uid,
    HealthScoreService? service,
    Future<PlaidRefreshResult> Function()? plaidRefresh,
  })  : _service = service ?? HealthScoreService.instance,
        _plaidRefresh = plaidRefresh ?? PlaidService.instance.refresh {
    _subscribe();
  }

  final String uid;
  final HealthScoreService _service;
  final Future<PlaidRefreshResult> Function() _plaidRefresh;

  /// In-flight guard for [refreshAll], separate from [HomeScoreState.refreshing]
  /// on purpose: that flag drives the Re-run pill's spinner, and a pull must
  /// not change how the pill looks or behaves.
  bool _refreshingAll = false;

  /// Pull-to-refresh: brings the connected accounts up to date through POST
  /// /plaid/refresh, which also recomputes the derived profile fields and the
  /// score server-side. Nothing is awaited from the stream afterwards; the
  /// rescored result arrives through it like any other server write.
  ///
  /// Returns null without calling anything when a refresh is already running.
  /// Errors propagate to the caller, which owns the message.
  Future<PlaidRefreshResult?> refreshAll() async {
    if (_refreshingAll) return null;
    _refreshingAll = true;
    try {
      return await _plaidRefresh();
    } finally {
      _refreshingAll = false;
    }
  }

  StreamSubscription<ScoreState>? _subscription;
  HomeScoreState _state = const HomeScoreState();

  /// How long to keep the refresh indicator up waiting for the recomputed
  /// score to come back down the stream. Generous — the write has already
  /// landed server-side by the time we start waiting, so this only expires if
  /// the listener itself is wedged, in which case dropping the indicator is
  /// better than holding it forever.
  static const _streamEchoTimeout = Duration(seconds: 10);

  HomeScoreState get state => _state;

  /// When the score on screen was last computed, or null if there isn't one.
  /// Used to tell a refresh's result apart from the snapshot that preceded it.
  DateTime? _computedAt;
  Completer<void>? _awaitingEcho;

  void _subscribe() {
    _subscription = _service.watch(uid).listen(
      _onState,
      onError: (Object error, StackTrace stack) {
        debugPrint('score stream failed for $uid: $error');
        _emit(_state.copyWith(
          score: ScoreFailed(
            'Could not load your score right now. Pull down to try again.',
            error,
          ),
          refreshing: false,
        ));
        _completeEcho();
      },
    );
  }

  void _onState(ScoreState score) {
    final computedAt = _computedAtOf(score);
    final changed = computedAt != null && computedAt != _computedAt;
    _computedAt = computedAt;
    _emit(_state.copyWith(score: score));
    // A snapshot carrying a NEW computation is the refresh landing. Anything
    // else — the pre-refresh snapshot, a metadata-only event — is not, so the
    // indicator stays up.
    if (changed) _completeEcho();
  }

  /// The Re-run Score pill (score only). Asks the backend to recompute, then waits for the result
  /// to arrive through the stream so the indicator covers the whole round trip
  /// rather than disappearing while the screen still shows the old number.
  ///
  /// Never throws: a failed refresh becomes [HomeScoreState.refreshError],
  /// leaving whatever is already on screen intact. Safe to call while a refresh
  /// is already running (it is a no-op).
  Future<void> refresh() async {
    if (_state.refreshing) return;
    _emit(_state.copyWith(refreshing: true, clearRefreshError: true));

    final echo = Completer<void>();
    _awaitingEcho = echo;
    try {
      await _service.requestRefresh();
      // The POST has written the cache; wait for the listener to deliver it.
      await echo.future.timeout(_streamEchoTimeout, onTimeout: () {});
    } on HealthScoreException catch (e) {
      _emit(_state.copyWith(refreshError: e.message));
    } catch (e) {
      debugPrint('score refresh failed for $uid: $e');
      _emit(_state.copyWith(
          refreshError: 'Could not refresh your score. Please try again.'));
    } finally {
      _awaitingEcho = null;
      _emit(_state.copyWith(refreshing: false));
    }
  }

  /// Clears a refresh error once the UI has shown it (e.g. after a snackbar),
  /// so it is not re-shown on the next rebuild.
  void acknowledgeRefreshError() {
    if (_state.refreshError == null) return;
    _emit(_state.copyWith(clearRefreshError: true));
  }

  void _completeEcho() {
    final echo = _awaitingEcho;
    if (echo != null && !echo.isCompleted) echo.complete();
  }

  void _emit(HomeScoreState next) {
    if (_disposed) return;
    _state = next;
    notifyListeners();
  }

  static DateTime? _computedAtOf(ScoreState score) => switch (score) {
        ScoreComplete(:final profile) => profile.computedAt,
        ScorePartial(:final profile) => profile.computedAt,
        ScoreUnavailable(:final profile) => profile?.computedAt,
        ScoreLoading() || ScoreNeverComputed() || ScoreFailed() => null,
      };

  bool _disposed = false;

  @override
  void dispose() {
    _disposed = true;
    _subscription?.cancel();
    _completeEcho();
    super.dispose();
  }
}
