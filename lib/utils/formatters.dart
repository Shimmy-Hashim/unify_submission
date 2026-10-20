/// Display formatting shared by the dashboard screens. Kept dependency-free
/// (no intl package) — SGD amounts and short dates only.
///
/// Every money figure in the app goes through one of the three formatters
/// below, which is what makes Settings > "Hide Balances by Default" a
/// single-point change: when the preference is on they all return
/// [kMaskedAmount] instead of the real number.
library;

import '../services/app_preferences.dart';

/// Rendered in place of any SGD figure while Hide Balances is on.
const kMaskedAmount = '••••';

bool get _hidden => AppPreferences.instance.hideBalances;

/// "$245,847" — rounds to whole dollars, comma thousands separators.
String formatSgd(num amount) {
  if (_hidden) return kMaskedAmount;
  final rounded = amount.round();
  final sign = rounded < 0 ? '-' : '';
  final digits = rounded.abs().toString();
  final buffer = StringBuffer(sign)..write('\$');
  for (var i = 0; i < digits.length; i++) {
    if (i > 0 && (digits.length - i) % 3 == 0) buffer.write(',');
    buffer.write(digits[i]);
  }
  return buffer.toString();
}

/// "$12.34" — two-decimal variant for unit prices and per-share figures.
String formatSgdExact(num amount) =>
    _hidden ? kMaskedAmount : '\$${amount.toStringAsFixed(2)}';

/// "-$12.34" / "+$12.34" — signed transaction amounts. [negative] follows
/// each screen's own sign convention (Plaid's differs between the banking
/// and investments products), so callers pass it explicitly.
String formatSignedSgdExact(num amount, {required bool negative}) => _hidden
    ? kMaskedAmount
    : '${negative ? '-' : '+'}\$${amount.abs().toStringAsFixed(2)}';

/// "↑ +$1,234" / "↓ -$1,234" for a period-over-period change. Masked to the
/// bare [kMaskedAmount]: keeping the arrow and sign would leak the direction
/// of an amount the user asked to hide.
String formatSgdChange(num change) => _hidden
    ? kMaskedAmount
    : '${change >= 0 ? '↑ +' : '↓ -'}${formatSgd(change.abs())}';

const _monthsShort = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

/// "1 Jan 2027" — matches the Figma policy-card date style.
String formatShortDate(DateTime date) =>
    '${date.day} ${_monthsShort[date.month - 1]} ${date.year}';

/// How fresh the synced data on a hero card is, in the device's local time:
/// "Updated just now" under a minute (or a few seconds in the future, from
/// clock skew), "Updated 5 min ago" under an hour, "Updated today 3:42 pm"
/// later the same day, "Updated 9 Oct" on an earlier day this year, and
/// "Updated 9 Oct 2025" in an earlier year.
String updatedLabel(DateTime synced, {required DateTime now}) {
  final local = synced.toLocal();
  final current = now.toLocal();
  final age = current.difference(local);
  if (age.inMinutes < 1) return 'Updated just now';
  if (age.inMinutes < 60) return 'Updated ${age.inMinutes} min ago';
  if (local.year == current.year &&
      local.month == current.month &&
      local.day == current.day) {
    final hour = local.hour % 12 == 0 ? 12 : local.hour % 12;
    final minute = local.minute.toString().padLeft(2, '0');
    return 'Updated today $hour:$minute ${local.hour < 12 ? 'am' : 'pm'}';
  }
  final date = '${local.day} ${_monthsShort[local.month - 1]}';
  return local.year == current.year
      ? 'Updated $date'
      : 'Updated $date ${local.year}';
}
