import 'package:flutter/material.dart';

import '../../models/app_user.dart';
import '../../models/user_financial_profile.dart';
import '../../services/firebase_service.dart';
import '../../utils/app_styles.dart';
import '../../widgets/primary_button.dart';
import 'auth_ui.dart';

/// One-time profile form shown after account creation (Section 8:
/// "asked once and never again"). Submitting creates users/{uid} exactly
/// once, inside a transaction (see FirebaseService.createUserProfile).
///
/// No Figma frame exists for this screen — it follows the design language
/// of Sign In / Sign Up (nodes 3:243 / 3:1025).
class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final _formKey = GlobalKey<FormState>();
  late final TextEditingController _nameController;
  final _dobController = TextEditingController();
  final _incomeController = TextEditingController();
  DateTime? _dateOfBirth;
  int _dependants = 0;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    // Prefill from Google display name when available.
    _nameController = TextEditingController(
      text: FirebaseService.instance.currentUser?.displayName ?? '',
    );
  }

  @override
  void dispose() {
    _nameController.dispose();
    _dobController.dispose();
    _incomeController.dispose();
    super.dispose();
  }

  Future<void> _pickDateOfBirth() async {
    final now = DateTime.now();
    final picked = await showDatePicker(
      context: context,
      initialDate: _dateOfBirth ?? DateTime(now.year - 25, now.month, now.day),
      firstDate: DateTime(now.year - 100),
      lastDate: now,
      helpText: 'Date of birth',
    );
    if (picked != null) {
      setState(() {
        _dateOfBirth = picked;
        _dobController.text =
            '${picked.day.toString().padLeft(2, '0')}/${picked.month.toString().padLeft(2, '0')}/${picked.year}';
      });
    }
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    final authUser = FirebaseService.instance.currentUser;
    if (authUser == null) return; // AuthGate will show sign-in.

    setState(() => _busy = true);
    try {
      await FirebaseService.instance.createUserProfile(
        AppUser(
          uid: authUser.uid,
          name: _nameController.text.trim(),
          email: authUser.email ?? '',
          dateOfBirth: _dateOfBirth!,
          dependants: _dependants,
        ),
        monthlyIncomeSgd:
            UserFinancialProfile.parseIncomeInput(_incomeController.text),
      );
      // No derived-field refresh needed here: a fresh signup cannot have
      // Plaid data yet (Connect Bank sits behind MainShell), and the Plaid
      // sync endpoints refresh financialProfile server-side on every sync.
      // AuthGate's users/{uid} stream now emits the new doc -> MainShell.
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not save profile: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
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
              Text('About You', style: AppTextStyles.heroTitle),
              const SizedBox(height: 8),
              Text(
                'A few details to personalise your score.\nWe only ask this once.',
                textAlign: TextAlign.center,
                style: AppTextStyles.placeholder,
              ),
              const SizedBox(height: 24),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 42),
                child: Form(
                  key: _formKey,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      UnifyTextField(
                        label: 'Name',
                        hint: 'Enter Full Name',
                        controller: _nameController,
                        textCapitalization: TextCapitalization.words,
                        validator: (v) => (v == null || v.trim().isEmpty)
                            ? 'Enter your name'
                            : null,
                      ),
                      const SizedBox(height: 26),
                      UnifyTextField(
                        label: 'Date of Birth',
                        hint: 'Select Date of Birth',
                        controller: _dobController,
                        readOnly: true,
                        onTap: _busy ? null : _pickDateOfBirth,
                        validator: (v) => (v == null || v.isEmpty)
                            ? 'Select your date of birth'
                            : null,
                        suffix: const Icon(Icons.calendar_today_outlined,
                            size: 20, color: Colors.black),
                      ),
                      const SizedBox(height: 26),
                      UnifyTextField(
                        label: 'Monthly Income (SGD)',
                        hint: 'e.g. 4500',
                        controller: _incomeController,
                        keyboardType: const TextInputType.numberWithOptions(
                            decimal: true),
                        validator: UserFinancialProfile.validateIncomeInput,
                      ),
                      const SizedBox(height: 26),
                      DependantsStepper(
                        value: _dependants,
                        onChanged: (v) => setState(() => _dependants = v),
                        enabled: !_busy,
                      ),
                      const SizedBox(height: 40),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 11),
                        child: PrimaryButton(
                          label: 'Continue',
                          busy: _busy,
                          onPressed: _submit,
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
