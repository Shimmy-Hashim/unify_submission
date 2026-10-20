import 'package:flutter/material.dart';

import '../utils/app_styles.dart';

/// The design-system "Button" component (variant Primary, size Medium) used
/// at the bottom of Sign In / Sign Up — dark pill, white label. First Flutter
/// implementation of this component; reuse for any primary action.
class PrimaryButton extends StatelessWidget {
  const PrimaryButton({
    super.key,
    required this.label,
    required this.onPressed,
    this.busy = false,
  });

  final String label;
  final VoidCallback? onPressed;
  final bool busy;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 62,
      child: FilledButton(
        onPressed: busy ? null : onPressed,
        style: FilledButton.styleFrom(
          backgroundColor: AppColors.buttonDark,
          disabledBackgroundColor: AppColors.buttonDark.withValues(alpha: 0.6),
          shape: const StadiumBorder(),
        ),
        child: busy
            ? const SizedBox(
                height: 22,
                width: 22,
                child: CircularProgressIndicator(
                    strokeWidth: 2, color: Colors.white),
              )
            : Text(label, style: AppTextStyles.buttonLabel),
      ),
    );
  }
}
