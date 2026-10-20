import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';

import '../../services/firebase_service.dart';
import '../../utils/app_styles.dart';
import '../../widgets/primary_button.dart';
import 'auth_ui.dart';
import 'sign_up_screen.dart';

/// Sign In (Figma node 3:243). The design labels the first field "Username";
/// auth is email-based, so it is labelled Email here — same styling.
class SignInScreen extends StatefulWidget {
  const SignInScreen({super.key});

  @override
  State<SignInScreen> createState() => _SignInScreenState();
}

class _SignInScreenState extends State<SignInScreen> {
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

  Future<void> _run(Future<void> Function() action) async {
    setState(() => _busy = true);
    try {
      await action();
      // Navigation is handled by AuthGate reacting to auth state.
    } on FirebaseAuthException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(e.message ?? 'Sign-in failed (${e.code})')),
        );
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  void _signInWithEmail() {
    if (!_formKey.currentState!.validate()) return;
    _run(() => FirebaseService.instance.signInWithEmail(
          _emailController.text.trim(),
          _passwordController.text,
        ));
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
              Text('Sign In', style: AppTextStyles.heroTitle),
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
                        autofillHints: const [AutofillHints.password],
                        validator: (v) => (v == null || v.isEmpty)
                            ? 'Enter your password'
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
                            semanticLabel: 'Sign in with Google',
                            onTap: _busy
                                ? null
                                : () => _run(() => FirebaseService.instance
                                    .signInWithGoogle()),
                          ),
                        ],
                      ),
                      const SizedBox(height: 40),
                      SwitchModeLink(
                        prefix: "Don't have an account? ",
                        action: 'Register',
                        onTap: () => Navigator.of(context).push(
                          MaterialPageRoute(
                              builder: (_) => const SignUpScreen()),
                        ),
                      ),
                      const SizedBox(height: 12),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 11),
                        child: PrimaryButton(
                          label: 'Login',
                          busy: _busy,
                          onPressed: _signInWithEmail,
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
