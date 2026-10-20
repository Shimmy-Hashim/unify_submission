import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';

import '../../services/account_service.dart';
import '../../services/firebase_service.dart';
import '../../services/notification_service.dart';
import '../../utils/app_styles.dart';
import '../../widgets/primary_button.dart';
import '../auth/auth_ui.dart';

/// The exact phrase the user has to type. Uppercase and checked case-sensitively
/// — the friction is the point, and a case-insensitive check would let an
/// autocorrected "delete" through.
const _kConfirmPhrase = 'DELETE';

/// Settings > Account Actions > Delete Account.
///
/// THREE INDEPENDENT GATES, because this is the one irreversible thing the app
/// can do and no single one of them is enough on its own:
///
///   1. Reading what is actually destroyed — listed by name, not "your data".
///   2. Typing DELETE — defeats a mis-tap and a muscle-memory confirm dialog.
///   3. Real Firebase Auth reauthentication — proves the person holding the
///      unlocked phone is the account owner, not someone who picked it up.
///
/// Gate 3 reuses the helpers Pass 1 built for Password & Security
/// (reauthenticateWithPassword) rather than a lighter confirmation, which is
/// the whole reason that helper is on FirebaseService instead of inside
/// ChangePasswordScreen. A Google-only account has no password to check, so it
/// reauthenticates through the account picker instead — same guarantee, the
/// only mechanism Firebase offers for that provider.
class DeleteAccountScreen extends StatefulWidget {
  const DeleteAccountScreen({super.key});

  @override
  State<DeleteAccountScreen> createState() => _DeleteAccountScreenState();
}

class _DeleteAccountScreenState extends State<DeleteAccountScreen> {
  final _formKey = GlobalKey<FormState>();
  final _passwordController = TextEditingController();
  final _phraseController = TextEditingController();
  bool _busy = false;
  bool _showPassword = false;

  bool get _hasPassword => FirebaseService.instance.hasPasswordProvider;

  @override
  void initState() {
    super.initState();
    // Enables/disables the button as the phrase is typed.
    _phraseController.addListener(() => setState(() {}));
  }

  @override
  void dispose() {
    _passwordController.dispose();
    _phraseController.dispose();
    super.dispose();
  }

  bool get _phraseMatches => _phraseController.text.trim() == _kConfirmPhrase;

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    if (!_phraseMatches) return;

    // Last stop before anything irreversible happens.
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Delete account permanently?'),
        content: const Text(
            'This cannot be undone. Your profile, connected accounts, '
            'transactions, investments and insurance policies will be erased.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            style: TextButton.styleFrom(
                foregroundColor:
                    AppTheme.alertRed(dialogContext, AppColors.danger)),
            child: const Text('Delete Forever'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;

    setState(() => _busy = true);
    final uid = FirebaseService.instance.currentUser?.uid;
    try {
      // --- Gate 3: real reauthentication, immediately before the call.
      if (_hasPassword) {
        await FirebaseService.instance
            .reauthenticateWithPassword(_passwordController.text);
      } else {
        final reauthed =
            await FirebaseService.instance.reauthenticateWithGoogle();
        if (!reauthed) {
          // Picker dismissed — not a failure, just not confirmed.
          if (mounted) setState(() => _busy = false);
          return;
        }
      }

      // This device should stop receiving pushes for the account either way;
      // doing it first means a delete that fails halfway does not leave a
      // token registered against an account being torn down. The backend
      // deletes the whole collection anyway — this is belt and braces.
      if (uid != null) {
        await NotificationService.instance.unregisterDevice(uid);
      }

      await AccountService.instance.deleteAccount();

      // The backend has already deleted the Auth user, so this mostly clears
      // local credential state; AuthGate reacts to the auth-state change and
      // returns to Sign In on its own.
      await FirebaseService.instance.signOut();
      // No navigation here: AuthGate owns routing, and popping this screen
      // during a sign-out teardown races it.
    } on FirebaseAuthException catch (e) {
      _showMessage(_authMessage(e));
      if (mounted) setState(() => _busy = false);
    } on AccountServiceException catch (e) {
      _showMessage(e.message);
      if (mounted) setState(() => _busy = false);
    } catch (_) {
      _showMessage('Could not delete your account. Please try again.');
      if (mounted) setState(() => _busy = false);
    }
  }

  /// Same wording rationale as ChangePasswordScreen: Firebase's own credential
  /// messages describe the token rather than what the user did.
  static String _authMessage(FirebaseAuthException e) => switch (e.code) {
        'wrong-password' || 'invalid-credential' =>
          'That password is incorrect.',
        'too-many-requests' =>
          'Too many attempts. Wait a moment and try again.',
        'requires-recent-login' =>
          'For security, sign in again before deleting your account.',
        _ => e.message ?? 'Could not delete your account (${e.code}).',
      };

  void _showMessage(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Delete Account')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(28, 20, 28, 40),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // --- Gate 1: what actually gets destroyed, by name.
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    // Same 8% wash in both modes: in dark the red underneath
                    // it is the lifted one, so the panel still reads as a warm
                    // step up from #121212 without the heading losing the box.
                    color: AppTheme.alertRed(context, AppColors.danger)
                        .withValues(alpha: 0.08),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Icon(Icons.warning_amber_rounded,
                              color:
                                  AppTheme.alertRed(context, AppColors.danger)),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              'This cannot be undone',
                              style: AppTextStyles.fieldLabel.copyWith(
                                  color: AppTheme.alertRed(
                                      context, AppColors.danger)),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      Text('Deleting your account erases:',
                          style: AppTextStyles.placeholder),
                      const SizedBox(height: 6),
                      ...const [
                        'Your profile and income details',
                        'Every connected bank and brokerage',
                        'All synced transactions and holdings',
                        'All uploaded insurance policies',
                        'Your Financial Health Score history',
                      ].map((line) => Padding(
                            padding: const EdgeInsets.only(bottom: 2),
                            child: Text('•  $line',
                                style: AppTextStyles.placeholder),
                          )),
                    ],
                  ),
                ),
                const SizedBox(height: 28),

                // --- Gate 2: type the phrase.
                Text('Type $_kConfirmPhrase to confirm',
                    style: AppTextStyles.fieldLabel
                        .copyWith(color: AppTheme.primaryText(context))),
                const SizedBox(height: 8),
                UnifyTextField(
                  label: '',
                  hint: _kConfirmPhrase,
                  controller: _phraseController,
                  textCapitalization: TextCapitalization.characters,
                ),
                const SizedBox(height: 26),

                // --- Gate 3: reauthentication.
                if (_hasPassword)
                  UnifyTextField(
                    label: 'Confirm your password',
                    hint: 'Enter your password',
                    controller: _passwordController,
                    obscureText: !_showPassword,
                    autofillHints: const [AutofillHints.password],
                    validator: (v) => (v == null || v.isEmpty)
                        ? 'Enter your password to confirm'
                        : null,
                    suffix: IconButton(
                      icon: Icon(
                        _showPassword
                            ? Icons.visibility_outlined
                            : Icons.visibility_off_outlined,
                        color: AppColors.placeholder,
                      ),
                      onPressed: () =>
                          setState(() => _showPassword = !_showPassword),
                    ),
                  )
                else
                  Text(
                    'You signed in with Google. Tapping below opens the Google '
                    'account picker to confirm it is you before anything is '
                    'deleted.',
                    style: AppTextStyles.placeholder,
                  ),
                const SizedBox(height: 40),

                PrimaryButton(
                  label: 'Delete My Account',
                  busy: _busy,
                  // Stays inert until the phrase matches exactly, so the
                  // button cannot be hit reflexively.
                  onPressed: _phraseMatches ? _submit : null,
                ),
                const SizedBox(height: 12),
                Center(
                  child: TextButton(
                    onPressed: _busy ? null : () => Navigator.of(context).pop(),
                    child: const Text('Keep my account'),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
