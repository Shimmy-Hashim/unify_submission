import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';

import '../../services/firebase_service.dart';
import '../../utils/app_styles.dart';
import '../../widgets/primary_button.dart';
import '../auth/auth_ui.dart';

/// Settings > Account > Password & Security (Figma node 134:446, second row).
///
/// Firebase requires a recent login before updatePassword, so the current
/// password is collected and reauthenticated explicitly (a wrong password
/// then surfaces as a wrong-password error rather than an opaque
/// `requires-recent-login` on save). The 8-character minimum matches Sign Up.
class ChangePasswordScreen extends StatefulWidget {
  const ChangePasswordScreen({super.key});

  @override
  State<ChangePasswordScreen> createState() => _ChangePasswordScreenState();
}

class _ChangePasswordScreenState extends State<ChangePasswordScreen> {
  final _formKey = GlobalKey<FormState>();
  final _currentController = TextEditingController();
  final _newController = TextEditingController();
  final _confirmController = TextEditingController();
  bool _busy = false;
  bool _showCurrent = false;
  bool _showNew = false;

  @override
  void dispose() {
    _currentController.dispose();
    _newController.dispose();
    _confirmController.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _busy = true);
    try {
      await FirebaseService.instance
          .reauthenticateWithPassword(_currentController.text);
      await FirebaseService.instance.updatePassword(_newController.text);
      if (mounted) {
        Navigator.of(context).pop();
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Password updated')),
        );
      }
    } on FirebaseAuthException catch (e) {
      _showMessage(_authMessage(e));
    } catch (_) {
      _showMessage('Could not change password. Please try again.');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  /// Firebase's own messages are serviceable except for the credential
  /// errors, where the wording ("malformed or has expired") describes the
  /// token rather than what the user actually did.
  static String _authMessage(FirebaseAuthException e) => switch (e.code) {
        'wrong-password' ||
        'invalid-credential' =>
          'That current password is incorrect.',
        'weak-password' => 'Pick a stronger password (at least 8 characters).',
        'too-many-requests' =>
          'Too many attempts. Wait a moment and try again.',
        _ => e.message ?? 'Could not change password (${e.code}).',
      };

  void _showMessage(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    // Google-only accounts have no password credential to reauthenticate
    // against, so there is nothing to change here.
    if (!FirebaseService.instance.hasPasswordProvider) {
      return Scaffold(
        appBar: AppBar(title: const Text('Password & Security')),
        body: Padding(
          padding: const EdgeInsets.fromLTRB(28, 40, 28, 28),
          child: Column(
            children: [
              const Icon(Icons.lock_outline, size: 40),
              const SizedBox(height: 12),
              Text(
                'You signed in with Google',
                textAlign: TextAlign.center,
                style: AppTextStyles.fieldLabel
                    .copyWith(color: AppTheme.primaryText(context)),
              ),
              const SizedBox(height: 8),
              Text(
                'This account has no Unify password to change. Manage it '
                'from your Google account instead.',
                textAlign: TextAlign.center,
                style: AppTextStyles.placeholder,
              ),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Password & Security')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(28, 20, 28, 40),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                UnifyTextField(
                  label: 'Current Password',
                  hint: 'Enter Current Password',
                  controller: _currentController,
                  obscureText: !_showCurrent,
                  autofillHints: const [AutofillHints.password],
                  validator: (v) => (v == null || v.isEmpty)
                      ? 'Enter your current password'
                      : null,
                  suffix: IconButton(
                    icon: Icon(
                      _showCurrent
                          ? Icons.visibility_outlined
                          : Icons.visibility_off_outlined,
                      color: AppColors.placeholder,
                    ),
                    onPressed: () =>
                        setState(() => _showCurrent = !_showCurrent),
                  ),
                ),
                const SizedBox(height: 26),
                UnifyTextField(
                  label: 'New Password',
                  hint: 'Enter New Password',
                  controller: _newController,
                  obscureText: !_showNew,
                  autofillHints: const [AutofillHints.newPassword],
                  validator: (v) => (v == null || v.length < 8)
                      ? 'Password must be at least 8 characters'
                      : null,
                  suffix: IconButton(
                    icon: Icon(
                      _showNew
                          ? Icons.visibility_outlined
                          : Icons.visibility_off_outlined,
                      color: AppColors.placeholder,
                    ),
                    onPressed: () => setState(() => _showNew = !_showNew),
                  ),
                ),
                const SizedBox(height: 26),
                UnifyTextField(
                  label: 'Confirm New Password',
                  hint: 'Re-enter New Password',
                  controller: _confirmController,
                  obscureText: !_showNew,
                  validator: (v) => v != _newController.text
                      ? 'Passwords do not match'
                      : null,
                ),
                const SizedBox(height: 40),
                PrimaryButton(
                  label: 'Update Password',
                  busy: _busy,
                  onPressed: _submit,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
