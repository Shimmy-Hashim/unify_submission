import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

import '../../utils/app_styles.dart';

/// Shared building blocks for the auth/onboarding screens, matching the
/// Figma Sign In / Sign Up frames (nodes 3:243 / 3:1025).
///
/// THESE ARE NO LONGER AUTH-ONLY, WHICH IS WHY THEY RESOLVE COLOUR.
/// Settings > Edit Profile, Password & Security and Delete Account all reuse
/// [UnifyTextField] and [DependantsStepper], and those screens DO follow the
/// user's theme preference — so a hardcoded black label rendered as
/// black-on-black the moment Dark mode existed.
///
/// The auth screens themselves stay light by design (no dark Figma variant,
/// and they hardcode a light #F2F3F5 background). They now say so explicitly
/// by wrapping their own subtree in `Theme(data: AppTheme.lightTheme, …)`
/// rather than relying on every leaf widget being hardcoded — which is what
/// lets these widgets follow the theme without the auth flow inheriting Dark.

/// Full-width Unify logo banner shown at the top of auth screens.
class UnifyLogoBanner extends StatelessWidget {
  const UnifyLogoBanner({super.key});

  @override
  Widget build(BuildContext context) {
    return Image.asset(
      'assets/images/unify_logo.png',
      width: double.infinity,
      fit: BoxFit.fitWidth,
    );
  }
}

/// Space Grotesk Bold label + outlined input box (#818181 border, 20px
/// radius, 60px tall).
class UnifyTextField extends StatelessWidget {
  const UnifyTextField({
    super.key,
    required this.label,
    required this.hint,
    this.controller,
    this.obscureText = false,
    this.suffix,
    this.validator,
    this.keyboardType,
    this.autofillHints,
    this.textCapitalization = TextCapitalization.none,
    this.readOnly = false,
    this.onTap,
    this.style,
  });

  final String label;
  final String hint;
  final TextEditingController? controller;
  final bool obscureText;
  final Widget? suffix;
  final String? Function(String?)? validator;
  final TextInputType? keyboardType;
  final Iterable<String>? autofillHints;
  final TextCapitalization textCapitalization;
  final bool readOnly;
  final VoidCallback? onTap;
  final TextStyle? style;

  static OutlineInputBorder _border(Color color, [double width = 1]) =>
      OutlineInputBorder(
        borderRadius: BorderRadius.circular(20),
        borderSide: BorderSide(color: color, width: width),
      );

  @override
  Widget build(BuildContext context) {
    // AppTextStyles hardcodes black and is shared with the light-pinned auth
    // screens, so the ink is overridden here rather than changed at source.
    final ink = AppTheme.primaryText(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: AppTextStyles.fieldLabel.copyWith(color: ink)),
        const SizedBox(height: 4),
        TextFormField(
          controller: controller,
          obscureText: obscureText,
          validator: validator,
          keyboardType: keyboardType,
          autofillHints: autofillHints,
          textCapitalization: textCapitalization,
          readOnly: readOnly,
          onTap: onTap,
          style: style ?? AppTextStyles.input.copyWith(color: ink),
          decoration: InputDecoration(
            hintText: hint,
            // #9C9C9C clears 5.9:1 on the dark background and is the Figma
            // value on light, so the placeholder needs no variant.
            hintStyle: AppTextStyles.placeholder,
            contentPadding:
                const EdgeInsets.symmetric(horizontal: 14, vertical: 19.5),
            // #818181 reads as an outline against both backgrounds — the
            // resting border is the one colour here that needs no variant.
            enabledBorder: _border(AppColors.inputBorder),
            focusedBorder: _border(ink, 1.5),
            errorBorder: _border(Theme.of(context).colorScheme.error),
            focusedErrorBorder: _border(Theme.of(context).colorScheme.error, 1.5),
            suffixIcon: suffix,
          ),
        ),
      ],
    );
  }
}

/// Dependants stepper — the same label + outlined 60px box as
/// [UnifyTextField], with −/+ controls instead of a keyboard. Lifted out of
/// OnboardingScreen when Settings > Edit Profile gained the field, so the two
/// profile forms stay one implementation rather than two look-alikes.
class DependantsStepper extends StatelessWidget {
  const DependantsStepper({
    super.key,
    required this.value,
    required this.onChanged,
    this.enabled = true,
  });

  /// Upper bound carried over from onboarding. `dependants` is a tone-only
  /// modifier in the score pipeline, so the cap only exists to keep the
  /// stepper sane.
  static const int maxDependants = 10;

  final int value;
  final ValueChanged<int> onChanged;
  final bool enabled;

  @override
  Widget build(BuildContext context) {
    final ink = AppTheme.primaryText(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Dependants',
            style: AppTextStyles.fieldLabel.copyWith(color: ink)),
        const SizedBox(height: 4),
        Container(
          height: 60,
          padding: const EdgeInsets.only(left: 14, right: 6),
          decoration: BoxDecoration(
            border: Border.all(color: AppColors.inputBorder),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Row(
            children: [
              Expanded(
                child: Text('$value',
                    style: AppTextStyles.input.copyWith(color: ink)),
              ),
              IconButton(
                icon: Icon(Icons.remove, color: ink),
                onPressed: !enabled || value <= 0
                    ? null
                    : () => onChanged(value - 1),
              ),
              IconButton(
                icon: Icon(Icons.add, color: ink),
                onPressed: !enabled || value >= maxDependants
                    ? null
                    : () => onChanged(value + 1),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

/// "Or Login With" divider — rule lines either side of muted bold text.
class OrLoginWithDivider extends StatelessWidget {
  const OrLoginWithDivider({super.key});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        const Expanded(child: Divider(color: AppColors.mutedText, height: 1)),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12),
          child: Text('Or Login With', style: AppTextStyles.muted),
        ),
        const Expanded(child: Divider(color: AppColors.mutedText, height: 1)),
      ],
    );
  }
}

/// Bordered square social sign-in button (55x58, #C9C9C9 border, 10px
/// radius). Disabled buttons render at full opacity to match the design;
/// the tap handler decides what "disabled" means (e.g. a coming-soon toast).
class SocialIconButton extends StatelessWidget {
  const SocialIconButton({
    super.key,
    required this.asset,
    required this.semanticLabel,
    this.onTap,
  });

  final String asset;
  final String semanticLabel;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: semanticLabel,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(10),
        child: Container(
          width: 55,
          height: 58,
          decoration: BoxDecoration(
            border: Border.all(color: AppColors.socialBorder),
            borderRadius: BorderRadius.circular(10),
          ),
          child: Center(
            child: SvgPicture.asset(asset, width: 44, height: 44),
          ),
        ),
      ),
    );
  }
}

/// Bottom switch-mode link: muted bold text with the action word in black.
class SwitchModeLink extends StatelessWidget {
  const SwitchModeLink({
    super.key,
    required this.prefix,
    required this.action,
    required this.onTap,
  });

  final String prefix; // e.g. "Don't have an account? "
  final String action; // e.g. "Register"
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Text.rich(
        TextSpan(
          style: AppTextStyles.muted,
          children: [
            TextSpan(text: prefix),
            TextSpan(
                text: action,
                style: AppTextStyles.muted.copyWith(color: Colors.black)),
            const TextSpan(text: ' here!'),
          ],
        ),
        textAlign: TextAlign.center,
      ),
    );
  }
}
