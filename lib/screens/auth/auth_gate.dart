import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';

import '../../main.dart' show MainShell;
import '../../services/firebase_service.dart';
import '../../services/notification_service.dart';
import '../splash_screen.dart';
import 'onboarding_screen.dart';
import 'sign_in_screen.dart';

/// Routes by auth + profile state:
///   signed out                       -> SignInScreen
///   signed in, no users/{uid} doc    -> OnboardingScreen (creates the doc
///                                       + financialProfile in one
///                                       transaction)
///   signed in, doc exists            -> MainShell
///
/// The users/{uid} doc is the only routing signal: onboarding creates it
/// and financialProfile together, so its existence implies both. Accounts
/// created before financialProfile existed are test/dev-only — deleted and
/// recreated via fresh signup, not handled with a retrofit branch here.
class AuthGate extends StatefulWidget {
  const AuthGate({super.key});

  @override
  State<AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<AuthGate> {
  String? _lastActiveTouchedForUid;

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      stream: FirebaseService.instance.authStateChanges,
      builder: (context, authSnap) {
        if (authSnap.connectionState == ConnectionState.waiting) {
          return const SplashScreen();
        }
        final user = authSnap.data;
        if (user == null) {
          _lastActiveTouchedForUid = null;
          return const SignInScreen();
        }
        return StreamBuilder<DocumentSnapshot<Map<String, dynamic>>>(
          stream: FirebaseService.instance.userDocStream(user.uid),
          builder: (context, docSnap) {
            if (!docSnap.hasData) return const SplashScreen();
            if (!docSnap.data!.exists) {
              // Account exists but profile was never created (fresh sign-up,
              // or a Google SSO first sign-in) — collect it now.
              return const OnboardingScreen();
            }
            if (_lastActiveTouchedForUid != user.uid) {
              _lastActiveTouchedForUid = user.uid;
              FirebaseService.instance.touchLastActive(user.uid);
              // Same once-per-signed-in-user gate: reads the OS permission and
              // this account's toggles, and refreshes the device's FCM token if
              // permission was already granted. Never prompts — the prompt
              // belongs to the Settings toggle (see NotificationService).
              NotificationService.instance.initialize(user.uid);
            }
            return const MainShell();
          },
        );
      },
    );
  }
}
