import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import 'firebase_service.dart';

/// The four toggle keys. Must match notifications/rules.py's KIND_* constants
/// exactly — they are the field names in the shared Firestore document, so a
/// typo here is a toggle the backend never sees.
class NotificationKinds {
  const NotificationKinds._();

  static const push = 'push_notifications';
  static const riskAlerts = 'risk_alerts';
  static const premiumDue = 'premium_due_reminders';
  static const monthlyScore = 'monthly_score_update';

  /// Same defaults as rules.DEFAULT_SETTINGS. A user with no document yet gets
  /// these; nothing is delivered regardless until the OS grants permission and
  /// a token is registered, so "on by default" cannot produce an unwanted push.
  static const defaults = <String, bool>{
    push: true,
    riskAlerts: true,
    premiumDue: true,
    monthlyScore: true,
  };
}

/// Settings > Notifications & Alerts, and the FCM plumbing behind it.
///
/// PERMISSION IS THE SOURCE OF TRUTH FOR THE MASTER TOGGLE.
/// The stored `push_notifications` value says what the user asked for; the OS
/// says what can actually be delivered. [pushEnabled] is the AND of the two, so
/// the master switch can never sit in the "on" position while Android or iOS is
/// dropping every push — which would be the app telling the user something
/// untrue about itself. When permission is missing, [permissionBlocked] is what
/// the UI uses to explain why the switch will not stay on.
class NotificationService extends ChangeNotifier {
  NotificationService._();
  static final NotificationService instance = NotificationService._();

  final FirebaseMessaging _messaging = FirebaseMessaging.instance;

  /// Set by main() so a foreground push can surface without a BuildContext.
  static final GlobalKey<ScaffoldMessengerState> messengerKey =
      GlobalKey<ScaffoldMessengerState>();

  AuthorizationStatus _osStatus = AuthorizationStatus.notDetermined;
  Map<String, bool> _settings = Map.of(NotificationKinds.defaults);
  String? _token;
  bool _wired = false;

  String? get token => _token;

  /// True only when the OS will actually deliver. `provisional` counts — that
  /// is iOS's quiet-delivery grant, which is a real (if silent) delivery path.
  bool get osAuthorized =>
      _osStatus == AuthorizationStatus.authorized ||
      _osStatus == AuthorizationStatus.provisional;

  /// The user turned pushes on but the OS is refusing — the state the UI has
  /// to explain rather than silently show as "off".
  bool get permissionBlocked =>
      !osAuthorized && (_settings[NotificationKinds.push] ?? false);

  /// The effective master switch. See the class comment.
  bool get pushEnabled =>
      osAuthorized && (_settings[NotificationKinds.push] ?? false);

  /// A category's effective state — a category is only live if the master is.
  bool enabled(String kind) =>
      pushEnabled && (_settings[kind] ?? NotificationKinds.defaults[kind]!);

  /// The stored value regardless of the master, for rendering a sub-toggle's
  /// own position while the master is off.
  bool storedValue(String kind) =>
      _settings[kind] ?? NotificationKinds.defaults[kind]!;

  // ---- lifecycle ----

  /// Called once after sign-in (from AuthGate) and again on resume.
  ///
  /// Deliberately does NOT request permission: a permission prompt on first
  /// launch, before the user has seen what notifications are for, is the
  /// reliable way to get it denied permanently. The prompt happens when they
  /// turn the master toggle on in Settings.
  Future<void> initialize(String uid) async {
    await refreshOsStatus();
    await _loadSettings(uid);
    if (osAuthorized) {
      // Permission already granted on a previous run — make sure this device's
      // token is registered and current before anything tries to send to it.
      await _registerToken(uid);
    }
    _wireListeners(uid);
    notifyListeners();
  }

  /// Re-reads the OS permission. Called on app resume, because the user can
  /// revoke notification permission in system settings while the app sits in
  /// the background and the toggle must not keep claiming to be on.
  Future<void> refreshOsStatus() async {
    try {
      final settings = await _messaging.getNotificationSettings();
      _osStatus = settings.authorizationStatus;
    } catch (e) {
      debugPrint('Could not read notification permission: $e');
      _osStatus = AuthorizationStatus.notDetermined;
    }
    notifyListeners();
  }

  Future<void> _loadSettings(String uid) async {
    try {
      final snap =
          await FirebaseService.instance.notificationSettingsDoc(uid).get();
      _settings = _merge(snap.data());
    } catch (e) {
      debugPrint('Could not read notification settings: $e');
    }
  }

  /// Per-key merge over the defaults, mirroring read_settings() in
  /// service.py — a document written before a toggle existed still yields a
  /// sane value for the new one.
  static Map<String, bool> _merge(Map<String, dynamic>? stored) => {
        for (final entry in NotificationKinds.defaults.entries)
          entry.key: (stored?[entry.key] as bool?) ?? entry.value,
      };

  // ---- permission + token ----

  /// Requests OS permission and, if granted, registers this device's token.
  ///
  /// Returns the resulting status so the caller can tell the three outcomes
  /// apart: granted, denied now, or denied permanently (the user must go to
  /// system settings — Android stops showing the dialog after two refusals and
  /// iOS after one, so re-prompting would do nothing at all).
  Future<AuthorizationStatus> requestPermission(String uid) async {
    try {
      final settings = await _messaging.requestPermission(
        alert: true,
        badge: true,
        sound: true,
      );
      _osStatus = settings.authorizationStatus;
      if (osAuthorized) await _registerToken(uid);
    } catch (e) {
      debugPrint('Permission request failed: $e');
    }
    notifyListeners();
    return _osStatus;
  }

  Future<void> _registerToken(String uid) async {
    try {
      // APNs has to have handed the device a token before FCM can mint one on
      // iOS; getToken() throws rather than waits if it has not. Android has no
      // such step. Either way a missing token is not fatal — the next resume
      // retries, and the backend simply has no device to send to until then.
      final token = await _messaging.getToken();
      if (token == null) return;
      _token = token;
      await _writeToken(uid, token);
    } catch (e) {
      debugPrint('Could not obtain FCM token: $e');
    }
  }

  Future<void> _writeToken(String uid, String token) async {
    // Document ID is the token itself: idempotent re-registration, and the
    // backend can prune one dead device without touching the others.
    await FirebaseService.instance.notificationTokens(uid).doc(token).set({
      'token': token,
      'platform': _platformName,
      'last_seen_at': FieldValue.serverTimestamp(),
    }, SetOptions(merge: true));
  }

  static String get _platformName {
    if (kIsWeb) return 'web';
    if (Platform.isAndroid) return 'android';
    if (Platform.isIOS) return 'ios';
    return Platform.operatingSystem;
  }

  // ---- toggles ----

  /// Writes one toggle to Firestore, requesting OS permission first when the
  /// master is being switched on.
  ///
  /// Returns false when the master could not be turned on because the OS
  /// refused — the caller shows the explanation, and nothing is written, so the
  /// switch springs back rather than lying about its state.
  Future<bool> setToggle(String uid, String kind, bool value) async {
    if (kind == NotificationKinds.push && value) {
      final status = await requestPermission(uid);
      if (status != AuthorizationStatus.authorized &&
          status != AuthorizationStatus.provisional) {
        return false;
      }
    }
    final previous = _settings[kind];
    _settings = {..._settings, kind: value};
    notifyListeners(); // optimistic — the switch should not lag the tap
    try {
      await FirebaseService.instance
          .notificationSettingsDoc(uid)
          .set({kind: value}, SetOptions(merge: true));
      return true;
    } catch (e) {
      debugPrint('Could not save notification setting: $e');
      _settings = {..._settings, kind: previous ?? NotificationKinds.defaults[kind]!};
      notifyListeners();
      return false;
    }
  }

  // ---- message handling ----

  void _wireListeners(String uid) {
    if (_wired) return;
    _wired = true;

    // FCM rotates tokens (reinstall, restore, app-data clear). Without this the
    // stored token goes stale and the device stops receiving with no visible
    // cause.
    _messaging.onTokenRefresh.listen((token) async {
      final currentUid = FirebaseService.instance.currentUser?.uid;
      if (currentUid == null) return;
      _token = token;
      await _writeToken(currentUid, token);
    });

    // Foreground: Android does NOT draw a system notification for a message
    // that arrives while the app is in the foreground, so without this the push
    // would simply vanish. A snackbar is shown instead — visible, dismissible,
    // and it needs no extra local-notifications dependency.
    FirebaseMessaging.onMessage.listen(_showForeground);
  }

  void _showForeground(RemoteMessage message) {
    final notification = message.notification;
    if (notification == null) return;
    final title = notification.title ?? 'Unify';
    final body = notification.body ?? '';
    messengerKey.currentState?.showSnackBar(
      SnackBar(
        duration: const Duration(seconds: 6),
        content: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(title,
                style: const TextStyle(fontWeight: FontWeight.w700)),
            if (body.isNotEmpty) Text(body),
          ],
        ),
      ),
    );
  }

  /// Called on sign-out and before account deletion: this device should stop
  /// receiving pushes meant for an account it is no longer signed in to.
  /// Best-effort — a failure here must not block signing out.
  Future<void> unregisterDevice(String uid) async {
    final token = _token;
    if (token == null) return;
    try {
      await FirebaseService.instance
          .notificationTokens(uid)
          .doc(token)
          .delete();
    } catch (e) {
      debugPrint('Could not unregister device token: $e');
    }
  }
}
