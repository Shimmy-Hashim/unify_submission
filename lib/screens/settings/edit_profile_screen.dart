import 'dart:async';

import 'package:flutter/material.dart';

import '../../models/app_user.dart';
import '../../models/user_financial_profile.dart';
import '../../services/firebase_service.dart';
import '../../services/health_score_service.dart';
import '../../utils/app_styles.dart';
import '../../widgets/primary_button.dart';
import '../auth/auth_ui.dart';

/// Settings > Account > Edit Profile (Figma node 134:446, first row).
///
/// Edits the three editable users/{uid} fields — name, date of birth, and
/// dependants — reusing the onboarding form's field widgets so the two
/// profile forms stay visually identical. Email is not edited here (it is
/// a Firebase Auth credential, not a profile field), and
/// the design's "photo" has no backing store: Firebase Storage is
/// deliberately not provisioned in this project.
///
/// MONTHLY INCOME IS A FOURTH FIELD ON A DIFFERENT DOCUMENT.
/// The other three live on users/{uid}; income lives on
/// users/{uid}/financialProfile/profile (Section 8), so it is loaded and
/// saved separately — see [_loadIncome] and [_save]. It is edited here rather
/// than on a screen of its own because it is profile data the user was asked
/// for once at onboarding, alongside these three.
class EditProfileScreen extends StatefulWidget {
  const EditProfileScreen({super.key, required this.user});

  final AppUser user;

  @override
  State<EditProfileScreen> createState() => _EditProfileScreenState();
}

class _EditProfileScreenState extends State<EditProfileScreen> {
  final _formKey = GlobalKey<FormState>();
  late final TextEditingController _nameController;
  late final TextEditingController _dobController;
  final _incomeController = TextEditingController();
  late DateTime _dateOfBirth;
  late int _dependants;
  bool _busy = false;

  /// The income as it currently stands on financialProfile, or null while the
  /// read is in flight or after it failed. Kept so [_save] can tell an actual
  /// edit from an untouched field, and so a failed read can never be written
  /// back over a real value.
  double? _loadedIncome;
  bool _incomeReady = false;
  bool _incomeLoadFailed = false;

  /// The income field's hint after a failed read, and also its validation
  /// error: Save is refused rather than saving the other three fields alone.
  static const _incomeLoadFailedText = 'Could not load. Try again later';

  @override
  void initState() {
    super.initState();
    _nameController = TextEditingController(text: widget.user.name);
    _dateOfBirth = widget.user.dateOfBirth;
    _dependants = widget.user.dependants;
    _dobController = TextEditingController(text: _formatDob(_dateOfBirth));
    _loadIncome();
  }

  @override
  void dispose() {
    _nameController.dispose();
    _dobController.dispose();
    _incomeController.dispose();
    super.dispose();
  }

  /// One-shot read of the income the user last set. Fetched here rather than
  /// passed in, so nothing outside this screen has to change to make the field
  /// editable.
  Future<void> _loadIncome() async {
    try {
      final snap =
          await FirebaseService.instance.financialProfileDoc(widget.user.uid).get();
      final income = UserFinancialProfile.fromDoc(snap).monthlyIncomeSgd;
      if (!mounted) return;
      setState(() {
        _loadedIncome = income;
        _incomeController.text = _formatIncome(income);
        _incomeReady = true;
      });
    } catch (error) {
      debugPrint('edit profile: could not read monthly income ($error)');
      if (!mounted) return;
      // The other three fields stay editable. Income does not, because writing
      // a value we could not read first risks overwriting a real one.
      setState(() => _incomeLoadFailed = true);
    }
  }

  static String _formatDob(DateTime d) =>
      '${d.day.toString().padLeft(2, '0')}/${d.month.toString().padLeft(2, '0')}/${d.year}';

  /// "4500" for a whole amount, "4500.50" otherwise — the same shape as
  /// onboarding's "e.g. 4500" hint, so the field reads back the way it was
  /// entered.
  static String _formatIncome(double income) => income == income.roundToDouble()
      ? income.toStringAsFixed(0)
      : income.toStringAsFixed(2);

  Future<void> _pickDateOfBirth() async {
    final now = DateTime.now();
    final picked = await showDatePicker(
      context: context,
      initialDate: _dateOfBirth,
      firstDate: DateTime(now.year - 100),
      lastDate: now,
      helpText: 'Date of birth',
    );
    if (picked != null) {
      setState(() {
        _dateOfBirth = picked;
        _dobController.text = _formatDob(picked);
      });
    }
  }

  /// Nudges the backend to recompute the score after an income change.
  ///
  /// Income is the denominator of every ratio in the pipeline, and income = 0
  /// is the single most likely reason a score was unavailable — so the user
  /// should see the effect of fixing it without first having to discover
  /// pull-to-refresh on Home.
  ///
  /// Non-fatal by design, and silent: the income write has already succeeded
  /// and is what the user asked for, the four background triggers recompute
  /// anyway, and Home's own pull-to-refresh remains. Surfacing a failure here
  /// would report an error for something that is not the user's operation.
  Future<void> _refreshScoreNonFatal() async {
    try {
      await HealthScoreService.instance.requestRefresh();
    } catch (error) {
      debugPrint('edit profile: score refresh after income change failed '
          '($error)');
    }
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;
    // All or nothing: the income validator already refuses a failed read, and
    // a read still in flight has no value to save yet, so nothing is written
    // until income is known. Parsed before the first write for the same
    // reason, so a bad value can never leave the profile half-saved.
    if (!_incomeReady) return;
    final income =
        UserFinancialProfile.parseIncomeInput(_incomeController.text);
    setState(() => _busy = true);
    try {
      await FirebaseService.instance.updateUserProfile(
        widget.user.uid,
        name: _nameController.text.trim(),
        dateOfBirth: _dateOfBirth,
        dependants: _dependants,
      );

      // Second document, so a second write (Section 8: income lives on
      // financialProfile/profile, the other three on users/{uid}). Skipped
      // unless the value actually changed.
      if (income != _loadedIncome) {
        await FirebaseService.instance
            .updateMonthlyIncome(widget.user.uid, income);
        // Deliberately not awaited: the save is done and the user should not
        // wait on a convenience recompute.
        unawaited(_refreshScoreNonFatal());
      }

      if (mounted) {
        Navigator.of(context).pop();
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Profile updated')),
        );
      }
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
    return Scaffold(
      appBar: AppBar(title: const Text('Edit Profile')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(28, 20, 28, 40),
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
                  suffix: Icon(Icons.calendar_today_outlined,
                      size: 20, color: AppTheme.primaryText(context)),
                ),
                const SizedBox(height: 26),
                // Same label, hint, keyboard and validator as onboarding's
                // income field — the rules (0 allowed for students/NS,
                // negatives and > $1M rejected) live on the model, so the two
                // forms cannot drift apart.
                UnifyTextField(
                  label: 'Monthly Income (SGD)',
                  hint: _incomeLoadFailed ? _incomeLoadFailedText : 'e.g. 4500',
                  controller: _incomeController,
                  keyboardType:
                      const TextInputType.numberWithOptions(decimal: true),
                  readOnly: !_incomeReady || _busy,
                  // A failed read fails the whole form, so the other three
                  // fields are never saved without income. A read still in
                  // flight is refused by _save itself.
                  validator: (value) => _incomeReady
                      ? UserFinancialProfile.validateIncomeInput(value)
                      : _incomeLoadFailed
                          ? _incomeLoadFailedText
                          : null,
                ),
                const SizedBox(height: 26),
                DependantsStepper(
                  value: _dependants,
                  onChanged: (v) => setState(() => _dependants = v),
                  enabled: !_busy,
                ),
                const SizedBox(height: 40),
                PrimaryButton(
                  label: 'Save Changes',
                  busy: _busy,
                  onPressed: _save,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
