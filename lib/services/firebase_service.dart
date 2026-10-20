import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_sign_in/google_sign_in.dart';

import '../models/app_user.dart';
import '../models/user_financial_profile.dart';

/// Central access point for Firebase Auth and Firestore.
///
/// Firestore layout implemented so far (see Section 8 of the project brief):
///   users/{uid}                          — profile document
///   users/{uid}/financialProfile/profile — single-doc subcollection; the
///     client writes income once, the backend owns the derived fields
///   users/{uid}/insurancePolicies/{id}   — written by the backend parser
///     endpoint; the client only ever reads it.
class FirebaseService {
  FirebaseService._();
  static final FirebaseService instance = FirebaseService._();

  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _db = FirebaseFirestore.instance;

  // ---- Auth ----

  Stream<User?> get authStateChanges => _auth.authStateChanges();
  User? get currentUser => _auth.currentUser;

  Future<UserCredential> signUpWithEmail(String email, String password) {
    return _auth.createUserWithEmailAndPassword(email: email, password: password);
  }

  Future<UserCredential> signInWithEmail(String email, String password) {
    return _auth.signInWithEmailAndPassword(email: email, password: password);
  }

  /// Returns null if the user cancelled the Google account picker.
  Future<UserCredential?> signInWithGoogle() async {
    final googleUser = await GoogleSignIn().signIn();
    if (googleUser == null) return null;
    final googleAuth = await googleUser.authentication;
    final credential = GoogleAuthProvider.credential(
      accessToken: googleAuth.accessToken,
      idToken: googleAuth.idToken,
    );
    return _auth.signInWithCredential(credential);
  }

  Future<void> signOut() async {
    await GoogleSignIn().signOut();
    await _auth.signOut();
  }

  /// True when the signed-in account has an email/password credential.
  /// Google-only accounts have no password to reauthenticate against, so
  /// Settings > Password & Security explains that instead of offering a
  /// change form that could only ever fail.
  bool get hasPasswordProvider =>
      _auth.currentUser?.providerData
          .any((p) => p.providerId == 'password') ??
      false;

  /// Reauthenticates with the current password. Firebase requires a recent
  /// login before updatePassword; doing this explicitly turns an opaque
  /// `requires-recent-login` failure into a wrong-password error the user
  /// can actually act on.
  Future<void> reauthenticateWithPassword(String currentPassword) async {
    final user = _auth.currentUser;
    final email = user?.email;
    if (user == null || email == null) {
      throw FirebaseAuthException(
        code: 'no-current-user',
        message: 'You are signed out. Sign in again.',
      );
    }
    await user.reauthenticateWithCredential(
      EmailAuthProvider.credential(email: email, password: currentPassword),
    );
  }

  /// Reauthentication for a Google-only account — the counterpart to
  /// [reauthenticateWithPassword], not a replacement for it.
  ///
  /// Password & Security (Pass 1) never needed this: a Google account has no
  /// Unify password, so that screen correctly explains there is nothing to
  /// change. Delete Account does need it, because a Google user must be able
  /// to delete their account and Firebase requires a recent login for it. The
  /// account picker is the reauthentication — re-signing in with the same
  /// credential is exactly what `reauthenticateWithCredential` verifies.
  ///
  /// Returns false if the user cancelled the picker, so a cancelled
  /// reauthentication reads as "not confirmed" rather than as a failure.
  Future<bool> reauthenticateWithGoogle() async {
    final user = _auth.currentUser;
    if (user == null) {
      throw FirebaseAuthException(
        code: 'no-current-user',
        message: 'You are signed out. Sign in again.',
      );
    }
    final googleUser = await GoogleSignIn().signIn();
    if (googleUser == null) return false;
    final googleAuth = await googleUser.authentication;
    await user.reauthenticateWithCredential(
      GoogleAuthProvider.credential(
        accessToken: googleAuth.accessToken,
        idToken: googleAuth.idToken,
      ),
    );
    return true;
  }

  /// Call only after [reauthenticateWithPassword] has succeeded.
  Future<void> updatePassword(String newPassword) async {
    final user = _auth.currentUser;
    if (user == null) {
      throw FirebaseAuthException(
        code: 'no-current-user',
        message: 'You are signed out. Sign in again.',
      );
    }
    await user.updatePassword(newPassword);
  }

  /// Firebase ID token to send to the FastAPI backend as
  /// `Authorization: Bearer <token>`.
  Future<String?> getIdToken() => _auth.currentUser?.getIdToken() ?? Future.value(null);

  // ---- Firestore: users/{uid} ----

  DocumentReference<Map<String, dynamic>> userDoc(String uid) =>
      _db.collection('users').doc(uid);

  Stream<DocumentSnapshot<Map<String, dynamic>>> userDocStream(String uid) =>
      userDoc(uid).snapshots();

  /// Creates users/{uid} and users/{uid}/financialProfile/profile exactly
  /// once, in one transaction, at the end of onboarding — the profile and
  /// the income that personalises its score are a single create step, not
  /// two write paths. Throws StateError if the user document already exists.
  Future<void> createUserProfile(
    AppUser user, {
    required double monthlyIncomeSgd,
  }) async {
    final ref = userDoc(user.uid);
    await _db.runTransaction((tx) async {
      final snap = await tx.get(ref);
      if (snap.exists) {
        throw StateError('users/${user.uid} already exists');
      }
      tx.set(ref, {
        ...user.toCreateMap(),
        'created_at': FieldValue.serverTimestamp(),
        'last_active': FieldValue.serverTimestamp(),
      });
      tx.set(financialProfileDoc(user.uid),
          UserFinancialProfile.createMap(monthlyIncomeSgd));
    });
  }

  Future<void> touchLastActive(String uid) =>
      userDoc(uid).update({'last_active': FieldValue.serverTimestamp()});

  /// Settings > Edit Profile. Writes only the three fields that form exposes:
  /// `email` is owned by Firebase Auth and `created_at` is left untouched
  /// rather than round-tripped through the UI. Changing `date_of_birth`
  /// shifts the age band the scoring layers use, and `dependants` selects
  /// the tone of the Layer 1 checklist copy — the backend picks both up on
  /// its next score run, not here.
  Future<void> updateUserProfile(
    String uid, {
    required String name,
    required DateTime dateOfBirth,
    required int dependants,
  }) {
    return userDoc(uid).update({
      'name': name,
      'date_of_birth': Timestamp.fromDate(dateOfBirth),
      'dependants': dependants,
    });
  }

  // ---- Firestore: users/{uid}/financialProfile ----

  /// Section 8 specifies financialProfile as a single document; Firestore
  /// requires documents to live in a collection, so it is a one-document
  /// subcollection with the fixed ID "profile". Created only inside
  /// createUserProfile's transaction — onboarding is the sole client
  /// write path for this document.
  DocumentReference<Map<String, dynamic>> financialProfileDoc(String uid) =>
      userDoc(uid).collection('financialProfile').doc('profile');

  /// Settings > Edit Profile. The only client write to this document after
  /// [createUserProfile] creates it — and deliberately a single field.
  ///
  /// Until this existed, onboarding was the one and only place
  /// `monthly_income_sgd` was ever written, which left a student or NS user
  /// (Section 8 allows income = 0 for exactly those cases) with no way to
  /// update it after starting work — and income = 0 makes the whole score
  /// uncomputable, so the dead end was permanent.
  ///
  /// `income_source` is not written: it is `'user_entered'` for every user
  /// today (Plaid cannot derive salary reliably — Section 8), so a manual edit
  /// cannot change what it says. If a `'plaid_derived'` path is ever added,
  /// this write must start setting it back to `'user_entered'`.
  ///
  /// Uses `update`, not `set`: the document is created at onboarding and a
  /// caller reaching here without one is a real anomaly, not a case to paper
  /// over by creating a half-formed profile.
  Future<void> updateMonthlyIncome(String uid, double monthlyIncomeSgd) {
    return financialProfileDoc(uid)
        .update({'monthly_income_sgd': monthlyIncomeSgd});
  }

  // ---- Firestore: users/{uid}/insurancePolicies ----

  /// Path the upcoming insurance parser endpoint writes to. No documents
  /// exist yet — exposed so the Insurance screen can stream it later.
  CollectionReference<Map<String, dynamic>> insurancePolicies(String uid) =>
      userDoc(uid).collection('insurancePolicies');

  // ---- Firestore: users/{uid}/plaidAccounts (backend-written) ----

  CollectionReference<Map<String, dynamic>> plaidAccounts(String uid) =>
      userDoc(uid).collection('plaidAccounts');

  CollectionReference<Map<String, dynamic>> plaidTransactions(
          String uid, String accountId) =>
      plaidAccounts(uid).doc(accountId).collection('transactions');

  // ---- Firestore: users/{uid}/notification* (Settings Pass 2) ----

  /// The four Notifications & Alerts toggles. A single-document subcollection
  /// with the fixed ID "preferences", the same shape financialProfile uses —
  /// Firestore requires collection/document alternation, so a bare
  /// `users/{uid}/notificationSettings` document is not structurally possible.
  ///
  /// In Firestore rather than shared_preferences because all four gate pushes
  /// the BACKEND decides to send (the premium-due job, the score-change hook
  /// and the risk-alert hook all read this document from Python).
  /// shared_preferences is device-local and unreadable from the server, so it
  /// can only hold a purely client-side preference — Theme and Hide Balances
  /// are, and correctly stay there; none of these four is.
  DocumentReference<Map<String, dynamic>> notificationSettingsDoc(String uid) =>
      userDoc(uid).collection('notificationSettings').doc('preferences');

  /// One document per device, keyed BY the FCM token, so re-registering the
  /// same token rewrites one document instead of accumulating duplicates.
  /// A single `users/{uid}.fcmToken` field would let a second device silently
  /// knock the first one off notifications — see notifications/service.py.
  CollectionReference<Map<String, dynamic>> notificationTokens(String uid) =>
      userDoc(uid).collection('notificationTokens');

  // ---- Firestore: users/{uid}/scoreHistory (backend-written, read-only) ----

  CollectionReference<Map<String, dynamic>> scoreHistory(String uid) =>
      userDoc(uid).collection('scoreHistory');

  // ---- Firestore: users/{uid}/investmentAccounts (backend-written) ----

  CollectionReference<Map<String, dynamic>> investmentAccounts(String uid) =>
      userDoc(uid).collection('investmentAccounts');

  CollectionReference<Map<String, dynamic>> investmentTransactions(
          String uid, String accountId) =>
      investmentAccounts(uid).doc(accountId).collection('transactions');
}
