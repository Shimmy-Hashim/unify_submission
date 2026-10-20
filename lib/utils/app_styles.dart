import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

/// Design tokens lifted from the Figma Sign In / Sign Up frames
/// (file GfH2L1bSBwBnr44Mh4yrkJ, nodes 3:243 / 3:1025).
abstract final class AppColors {
  static const background = Color(0xFFF2F3F5);
  static const inputBorder = Color(0xFF818181);
  static const placeholder = Color(0xFF9C9C9C);
  static const mutedText = Color(0xFFB1B1B1);
  static const socialBorder = Color(0xFFC9C9C9);
  static const buttonDark = Color(0xFF2C2C2C);
  /// Fill behind rounded-12 option rows in selector sheets.
  static const selectorFill = Color(0xFFF8F8F8);

  /// Destructive actions. Not a Figma token — no frame in the design set has a
  /// destructive control — so this is Material's error red, added for Settings
  /// Pass 2's Delete Account and used nowhere else.
  static const danger = Color(0xFFD32F2F);

  // ---- From the UI/UX audit frames (August 2026) ----------------------------
  // Both were already the intended brand values; neither had ever been written
  // down, so screens reached for Flutter's defaults instead.

  /// The app's darkest ink. Pure #000000 was never a deliberate choice — it is
  /// what `Colors.black` gave whoever typed it first. Settings resolves its
  /// primary text through this; the other four screens still carry
  /// [AppTheme.primaryText]'s black and move over with their own audit tasks.
  static const brandBlack = Color(0xFF202020);

  /// Lime accent for the ON state of a control the user has actively switched.
  /// Reserved for genuine actions and selections — never a passive status
  /// indicator, per the convention the audit locked in. Pair with
  /// [brandBlack] on top: lime is bright enough that light ink disappears.
  static const limeAccent = Color(0xFFC9F158);

  /// Warning tint behind a destructive action's icon badge, so Delete Account
  /// does not sit in the same neutral tile as Log Out.
  static const dangerTint = Color(0xFFFFEBEE);
}

/// Theme-resolved surface and text tokens for the four dashboard screens.
///
/// WHY THIS EXISTS AS A SHARED SET AND NOT PER-SCREEN HELPERS
/// Settings Pass 1 resolved its own palette through four file-private helpers
/// (`_background`, `_surface`, `_primaryText`, `_divider` in
/// settings_screen.dart). That works for one screen. Home, Savings,
/// Investments and Insurance share a palette — the same `_kMuted`, the same
/// card shadow, the same black hero card, the same #F8F8F8 inset — so copying
/// those helpers into each file is precisely how four screens end up with four
/// different darks. One table instead.
///
/// EVERY LIGHT-MODE VALUE IS THE LITERAL THAT WAS THERE BEFORE. Switching a
/// screen onto these tokens must be a no-op in Light mode; only the dark
/// branch is new. Three near-identical greys were collapsed in the process
/// (#FAFAFA into [inset], 0x99888888 into [linkText], 0xAAAAAAAA into
/// [faintText]) — sub-perceptual differences that existed only because the
/// screens were written at different times.
///
/// Settings deliberately keeps its own private helpers: its light values are
/// identical to these, so the two agree, but Pass 1 is working code and this
/// task does not touch it.
abstract final class AppTheme {
  static bool isDark(BuildContext context) =>
      Theme.of(context).brightness == Brightness.dark;

  static Color _pick(BuildContext context, Color light, Color dark) =>
      isDark(context) ? dark : light;

  // ---- The two ThemeDatas ----
  //
  // Defined here rather than in main.dart because MaterialApp is no longer
  // their only consumer: the auth flow is light-only by design (Figma nodes
  // 3:243 / 3:1025 have no dark variant, and the screens hardcode a light
  // #F2F3F5 background), so those screens pin themselves to [lightTheme]
  // instead of following the user's preference. Two definitions of "the light
  // theme" would let the pinned flow drift away from the real one.

  static const seedColor = Color(0xFF1A73E8);

  static ThemeData get lightTheme => ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: seedColor),
        useMaterial3: true,
      );

  static ThemeData get darkTheme => ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: seedColor,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      );

  // ---- Surfaces ----

  /// Screen background. Dark matches Settings' `_kDarkBackground`.
  static Color background(BuildContext context) =>
      _pick(context, AppColors.background, const Color(0xFF121212));

  /// Card fill. Dark matches Settings' `_kDarkSurface`.
  static Color surface(BuildContext context) =>
      _pick(context, Colors.white, const Color(0xFF1E1E1E));

  /// The black summary card at the top of Savings, Investments and Insurance.
  ///
  /// Cannot stay `Colors.black` in dark mode: black on a #121212 background
  /// loses its edges and the card stops reading as a card. #2A2A2A sits above
  /// both the background and [surface], so the hero still reads as the most
  /// prominent thing on the screen — and its white-on-dark text, which is what
  /// the card was designed around, keeps working untouched.
  static Color heroSurface(BuildContext context) =>
      _pick(context, Colors.black, const Color(0xFF2A2A2A));

  /// Recessed fill *inside* a card — the emergency-fund need/have box, the
  /// upload drop zones. Must read as lower than [surface] in both modes.
  static Color inset(BuildContext context) =>
      _pick(context, const Color(0xFFF8F8F8), const Color(0xFF262626));

  /// Small rounded tile behind an icon or emoji. Dark matches the tile in
  /// Settings' `_RowIcon`.
  static Color iconTile(BuildContext context) =>
      _pick(context, AppColors.background, const Color(0xFF2A2A2A));

  // ---- Text, in descending prominence ----

  static Color primaryText(BuildContext context) =>
      _pick(context, Colors.black, Colors.white);

  static Color secondaryText(BuildContext context) =>
      _pick(context, const Color(0xFF555555), const Color(0xFFB5B5B5));

  static Color tertiaryText(BuildContext context) =>
      _pick(context, const Color(0xFF888888), const Color(0xFFA6A6A6));

  /// Body copy inside the small horizontal cards (Insights, Risk Alerts,
  /// Coverage Gaps), which carried a translucent grey rather than a solid one.
  static Color subtleBody(BuildContext context) =>
      _pick(context, const Color(0xAA555555), const Color(0xFFA8A8A8));

  /// "See all transactions →" and the other quiet inline actions.
  static Color linkText(BuildContext context) =>
      _pick(context, const Color(0x88888888), const Color(0xFF8A8A8A));

  /// The quietest text on screen — transaction date/category lines. Needs to
  /// go *down* in dark mode, not up: #BBBBBB would otherwise outrank the
  /// [secondaryText] title above it and invert the hierarchy.
  static Color faintText(BuildContext context) =>
      _pick(context, const Color(0xFFBBBBBB), const Color(0xFF7C7C7C));

  /// Constant in both modes, as in Settings (`_kMuted`). #AAAAAA clears 6.9:1
  /// on the dark card, so the screens' many `_kMuted` labels need no change.
  static const muted = Color(0xFFAAAAAA);

  // ---- Lines, tracks and edges ----

  static Color divider(BuildContext context) =>
      _pick(context, const Color(0xFFEEEEEE), const Color(0xFF2E2E2E));

  /// Divider expressed as an alpha over the card rather than a solid grey
  /// (insurance card). Stays alpha-based so it tracks whatever it sits on.
  static Color hairline(BuildContext context) =>
      _pick(context, const Color(0x14000000), const Color(0x1FFFFFFF));

  static Color dashedBorder(BuildContext context) =>
      _pick(context, const Color(0xFFDDDDDD), const Color(0xFF3A3A3A));

  /// Unfilled portion of a progress bar.
  static Color track(BuildContext context) =>
      _pick(context, const Color(0xFFEEEEEE), const Color(0xFF3A3A3A));

  /// The TierSegmentBar's "not assessed" half.
  ///
  /// This one has a three-way constraint rather than a two-way one: it has to
  /// stay distinguishable from the card behind it AND from the healthy green
  /// and weak red, in both modes. #D9D9D9 satisfies that on white; on a
  /// #1E1E1E card it would be the brightest thing in the row and read as
  /// *filled*, which is the opposite of what a neutral half means. #4A4A4F is
  /// a clear step up from the card without competing with either tier colour.
  static Color neutralSegment(BuildContext context) =>
      _pick(context, const Color(0xFFD9D9D9), const Color(0xFF4A4A4F));

  /// Cards keep their drop shadow in dark mode but need it deeper — a 25%
  /// black shadow is invisible against #121212.
  static BoxShadow cardShadow(BuildContext context) => BoxShadow(
        color: _pick(context, const Color(0x40000000), const Color(0x80000000)),
        offset: const Offset(0, 4),
        blurRadius: 4,
      );

  // ---- Chips ----

  /// Resolves one of the app's many pastel status chips (risk pills, policy
  /// type badges, "Parsed", "No data", the renewal strip) for the active
  /// theme.
  ///
  /// Derived rather than tabulated on purpose: there are eighteen of these
  /// pairs across Investments, Insights and the insurance card, and hand-
  /// picking eighteen more dark values is eighteen more chances to pick one
  /// that clashes. A pale-fill/dark-ink chip inverts cleanly — the hue is
  /// carried by lifting the ink to a light tint of itself and dropping the
  /// fill to a low-alpha wash of that tint, so every chip keeps its identity
  /// (red = risk, amber = caution, green = fine) without a second table.
  static ({Color background, Color foreground}) chip(
    BuildContext context, {
    required Color background,
    required Color foreground,
  }) {
    if (!isDark(context)) return (background: background, foreground: foreground);
    final ink = HSLColor.fromColor(foreground).withLightness(0.72).toColor();
    return (background: ink.withValues(alpha: 0.16), foreground: ink);
  }

  /// A chip's fill used on its own, with no paired ink — the coloured tile
  /// behind an institution or policy-type emoji.
  static Color tintedTile(BuildContext context, Color light) =>
      _pick(context, light, const Color(0xFF2A2A2A));

  // ---- Alerts ----

  /// The app's alert / destructive red, lifted for dark surfaces.
  ///
  /// The dark branch is what needed centralising. Both callers' light reds sit
  /// around 4.3:1 on the dark surfaces they land on — technically AA, but the
  /// weakest red in the app, and the insurance card had already picked a lift
  /// for itself. A second screen hand-picking its own lift is exactly how two
  /// screens end up with two different "alert reds", so the lift lives here.
  /// #FF8A80 clears 9:1 on #1E1E1E and 10:1 on #121212.
  ///
  /// [light] stays the caller's rather than being tabulated, as in
  /// [tintedTile]: the two light reds predate this token, differ by one
  /// Material step (#E53935 on the insurance card, [AppColors.danger] #D32F2F
  /// on Delete Account), and never appear on the same screen — collapsing them
  /// would break this table's rule that Light mode is left untouched.
  static Color alertRed(BuildContext context, Color light) =>
      _pick(context, light, const Color(0xFFFF8A80));
}

abstract final class AppTextStyles {
  /// Hero title ("Sign In" / "Sign Up"): Space Grotesk Bold 48, -1.44 tracking.
  static TextStyle get heroTitle => GoogleFonts.spaceGrotesk(
      fontWeight: FontWeight.w700,
      fontSize: 48,
      height: 1.2,
      letterSpacing: -1.44,
      color: Colors.black);

  /// Field label above each input box: Space Grotesk Bold 16.
  static TextStyle get fieldLabel => GoogleFonts.spaceGrotesk(
      fontWeight: FontWeight.w700,
      fontSize: 16,
      letterSpacing: -0.48,
      color: Colors.black);

  /// Placeholder inside inputs: Inter Regular 15 #9C9C9C.
  static TextStyle get placeholder =>
      GoogleFonts.inter(fontSize: 15, color: AppColors.placeholder);

  /// Typed input text.
  static TextStyle get input =>
      GoogleFonts.inter(fontSize: 15, color: Colors.black);

  /// "Or Login With" / switch-mode link base: Space Grotesk Bold 16 #B1B1B1.
  static TextStyle get muted => GoogleFonts.spaceGrotesk(
      fontWeight: FontWeight.w700,
      fontSize: 16,
      letterSpacing: -0.48,
      color: AppColors.mutedText);

  /// Primary button label.
  static TextStyle get buttonLabel => GoogleFonts.inter(
      fontSize: 20, fontWeight: FontWeight.w500, color: Colors.white);
}
