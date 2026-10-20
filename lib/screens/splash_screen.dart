import 'package:flutter/material.dart';

/// Splash (Figma node 3:3): white background, full-width Unify logo banner
/// vertically centered. Shown by AuthGate while the Firebase auth state /
/// profile document are being resolved.
class SplashScreen extends StatelessWidget {
  const SplashScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: Image.asset(
          'assets/images/unify_logo.png',
          width: double.infinity,
          fit: BoxFit.fitWidth,
        ),
      ),
    );
  }
}
