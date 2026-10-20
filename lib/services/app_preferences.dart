import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Device-local display preferences owned by Settings > App Preferences
/// (Figma node 134:417). Deliberately NOT stored in Firestore: theme and
/// balance masking are per-device display choices, not account data, so
/// users/{uid} stays exactly the profile document Section 8 describes.
///
/// First use of shared_preferences in the app — the dependency was already
/// declared in pubspec.yaml but unused, so this file establishes the
/// pattern rather than following one.
///
/// This is also the app's only ChangeNotifier. MaterialApp (theme) and
/// MainShell (balance masking) rebuild through ListenableBuilder when a
/// value changes, which is what makes a toggle take effect on screens that
/// are already built and alive inside the IndexedStack.
class AppPreferences extends ChangeNotifier {
  AppPreferences._();
  static final AppPreferences instance = AppPreferences._();

  static const _kThemeMode = 'pref_theme_mode';
  static const _kHideBalances = 'pref_hide_balances';

  SharedPreferences? _prefs;
  ThemeMode _themeMode = ThemeMode.system;
  bool _hideBalances = false;

  ThemeMode get themeMode => _themeMode;

  /// When true every SGD figure renders as [kMaskedAmount] (see
  /// utils/formatters.dart) instead of the real number.
  bool get hideBalances => _hideBalances;

  /// Called once from main() before runApp, so the very first frame already
  /// uses the stored theme — no flash of the wrong appearance on cold start.
  Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    _prefs = prefs;
    _themeMode = _themeModeFromName(prefs.getString(_kThemeMode));
    _hideBalances = prefs.getBool(_kHideBalances) ?? false;
    notifyListeners();
  }

  Future<void> setThemeMode(ThemeMode mode) async {
    if (mode == _themeMode) return;
    _themeMode = mode;
    // Notify first so the UI responds immediately; the disk write is not
    // something the user should wait on.
    notifyListeners();
    await _prefs?.setString(_kThemeMode, mode.name);
  }

  Future<void> setHideBalances(bool hide) async {
    if (hide == _hideBalances) return;
    _hideBalances = hide;
    notifyListeners();
    await _prefs?.setBool(_kHideBalances, hide);
  }

  static ThemeMode _themeModeFromName(String? name) => switch (name) {
        'light' => ThemeMode.light,
        'dark' => ThemeMode.dark,
        _ => ThemeMode.system,
      };

  static String themeModeLabel(ThemeMode mode) => switch (mode) {
        ThemeMode.light => 'Light',
        ThemeMode.dark => 'Dark',
        ThemeMode.system => 'System default',
      };
}
