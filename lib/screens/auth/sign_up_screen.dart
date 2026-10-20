import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';

import '../../services/firebase_service.dart';
import '../../utils/app_styles.dart';
import '../../widgets/primary_button.dart';
import 'auth_ui.dart';

/// Sign Up (Figma node 3:1025) — email + password only, mirroring the
/// design. Creates the Firebase Auth account only; the users/{uid} Firestore
/// document is created once, at the end of onboarding (AuthGate routes
/// there next).
class SignUpScreen extends StatefulWidget {
  const SignUpScreen({super.key});

  @override
  State<SignUpScreen> createState() => _SignUpScreenState();
}

class _SignUpScreenState extends State<SignUpScreen> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _busy = false;
  bool _showPassword = false;

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  /// [action] returns whether a user is now signed in. A cancelled Google
  /// picker returns false and leaves this screen as it is, with no message,
  /// the same as Sign In.
  Future<void> _run(Future<bool> Function() action) async {
    setState(() => _busy = true);
    try {
      final signedIn = await action();
      // AuthGate now shows OnboardingScreen; unwind to the root below it.
      if (signedIn && mounted) {
        Navigator.of(context).popUntil((r) => r.isFirst);
      }
    } on FirebaseAuthException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(e.message ?? 'Sign-up failed (${e.code})')),
        );
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  void _signUp() {
    if (!_formKey.currentState!.validate()) return;
    _run(() async {
      final credential = await FirebaseService.instance.signUpWithEmail(
        _emailController.text.trim(),
        _passwordController.text,
      );
      return credential.user != null;
    });
  }

  @override
  Widget build(BuildContext context) {
    // Pinned light, not inherited. This flow has no dark variant in Figma and
    // hardcodes a light #F2F3F5 background, so without this a user who had
    // selected Dark and then signed out would get the now theme-aware
    // UnifyTextField drawing white labels onto a light screen.
    //
    // Split into a second method purely so the wrap costs one line instead of
    // re-indenting the whole tree.
    return Theme(data: AppTheme.lightTheme, child: _buildScaffold(context));
  }

  Widget _buildScaffold(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            children: [
              const UnifyLogoBanner(),
              const SizedBox(height: 14),
              Text('Sign Up', style: AppTextStyles.heroTitle),
              const SizedBox(height: 24),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 42),
                child: Form(
                  key: _formKey,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      UnifyTextField(
                        label: 'Email',
                        hint: 'Enter Email',
                        controller: _emailController,
                        keyboardType: TextInputType.emailAddress,
                        autofillHints: const [AutofillHints.email],
                        validator: (v) => (v == null || !v.contains('@'))
                            ? 'Enter a valid email'
                            : null,
                      ),
                      const SizedBox(height: 26),
                      UnifyTextField(
                        label: 'Password',
                        hint: 'Enter Password',
                        controller: _passwordController,
                        obscureText: !_showPassword,
                        autofillHints: const [AutofillHints.newPassword],
                        validator: (v) => (v == null || v.length < 8)
                            ? 'Password must be at least 8 characters'
                            : null,
                        suffix: IconButton(
                          icon: Icon(
                            _showPassword
                                ? Icons.visibility_off
                                : Icons.remove_red_eye,
                            color: Colors.black,
                          ),
                          onPressed: () =>
                              setState(() => _showPassword = !_showPassword),
                        ),
                      ),
                      const SizedBox(height: 30),
                      const OrLoginWithDivider(),
                      const SizedBox(height: 30),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          SocialIconButton(
                            asset: 'assets/icons/google.svg',
                            semanticLabel: 'Sign up with Google',
                            onTap: _busy
                                ? null
                                : () => _run(() async {
                                      // Null when the picker was cancelled.
                                      final credential = await FirebaseService
                                          .instance
                                          .signInWithGoogle();
                                      return credential?.user != null;
                                    }),
                          ),
                        ],
                      ),
                      const SizedBox(height: 40),
                      SwitchModeLink(
                        prefix: 'Already have an account? ',
                        action: 'Login',
                        onTap: () => Navigator.of(context).pop(),
                      ),
                      const SizedBox(height: 12),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 11),
                        child: PrimaryButton(
                          label: 'Register',
                          busy: _busy,
                          onPressed: _signUp,
                        ),
                      ),
                      const SizedBox(height: 40),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
