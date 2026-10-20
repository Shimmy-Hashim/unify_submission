import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../models/app_user.dart';
import '../models/investment_holding.dart';
import '../models/plaid_account.dart';
import '../services/app_preferences.dart';
import '../services/firebase_service.dart';
import '../services/notification_service.dart';
import '../services/plaid_link_service.dart';
import '../services/plaid_service.dart';
import '../utils/app_styles.dart';
import 'settings/change_password_screen.dart';
import 'settings/delete_account_screen.dart';
import 'settings/edit_profile_screen.dart';

const _kDarkSurface = Color(0xFF1E1E1E);
const _kDarkBackground = Color(0xFF121212);

/// Shown in the version footer. Static by design — reading the real build
/// number needs package_info_plus, which is not a dependency.
const _kVersionLabel = 'Unify v1.0.0';

/// Settings tab — Figma node 134:417, restyled against the audit frame
/// "Settings — UI/UX Fixes Applied".
///
/// All six sections are built: Profile, Account, Connections, Notifications &
/// Alerts, App Preferences and Account Actions, plus the version footer.
///
/// The frame's Financial Health Score preview card is deliberately NOT here,
/// and is not pending either. It was deferred through Pass 1/2, and the audit
/// then removed it from the design outright: the score, its "Re-run" action
/// and its last-calculated timestamp all live on Home, and a second copy in
/// Settings was redundant. Do not re-add it.
///
/// The audit's own changes, so a later reader can tell intent from accident:
/// secondary text is #6B6B6B in light mode for WCAG AA (the old #AAAAAA
/// measured ~2.3:1 on white); primary ink is [AppColors.brandBlack], not pure
/// black; every row badge carries a real Material icon rather than an emoji,
/// on one shared neutral tile — except Delete Account, which keeps a red tint
/// because a destructive action should not look like Log Out.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen>
    with WidgetsBindingObserver {
  bool _connecting = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  /// The user can revoke (or grant) notification permission in system settings
  /// while Unify sits in the background. Re-reading it on resume is what stops
  /// the master toggle from showing a state the OS no longer agrees with.
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      NotificationService.instance.refreshOsStatus();
    }
  }

  // ---- Connections: the same Plaid Link chain Savings and Investments use.
  // One Item can carry both banking and investment accounts, and the link
  // token already requests both products, so there is no separate broker
  // flow to build — hence the single merged section in the frame.

  /// PlaidLinkService owns the Link lifecycle; the result comes back only to
  /// the screen that started it, so Savings and Investments stay silent even
  /// though they are alive in the IndexedStack.
  Future<void> _connect() async {
    if (_connecting) return;
    setState(() => _connecting = true);
    final result = await PlaidLinkService.instance.connect();
    if (!mounted) return;
    setState(() => _connecting = false);
    if (result.status == PlaidConnectStatus.connected) {
      final institution = result.institutionName ?? 'your account';
      _showMessage('Connected $institution, accounts synced.');
    } else if (result.status == PlaidConnectStatus.failed) {
      _showMessage(result.message ??
          'The connection was interrupted. Please try again.');
    }
    // cancelled: the user backed out. busy: another screen owns the session.
  }

  void _showMessage(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message)));
  }

  // ---- App preferences ----

  Future<void> _pickTheme() async {
    final selected = await showModalBottomSheet<ThemeMode>(
      context: context,
      backgroundColor: _surface(context),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (sheetContext) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Theme',
                style: AppTextStyles.fieldLabel
                    .copyWith(color: _primaryText(context)),
              ),
              const SizedBox(height: 12),
              // Rounded-12 option rows on the sheet's own surface. The fill
              // comes from AppTheme.inset rather than AppColors.selectorFill:
              // the token's light value IS #F8F8F8, so light mode is
              // unchanged, but the raw constant stayed white in dark mode and
              // turned these into bright pills carrying black text.
              //
              // The near-invisible contrast against the sheet is the design,
              // not an accident — #F8F8F8 on white is a 7/255 step, and
              // #262626 on #1E1E1E is 8/255. The rows are meant to be barely
              // differentiated in both modes.
              for (final mode in ThemeMode.values)
                Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Material(
                    color: AppTheme.inset(context),
                    borderRadius: BorderRadius.circular(12),
                    child: InkWell(
                      borderRadius: BorderRadius.circular(12),
                      onTap: () => Navigator.of(sheetContext).pop(mode),
                      child: Padding(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 16, vertical: 16),
                        child: Row(
                          children: [
                            Expanded(
                              child: Text(
                                AppPreferences.themeModeLabel(mode),
                                // AppTextStyles.input hardcodes black — it is
                                // shared with the auth screens, so it is
                                // overridden here rather than changed there.
                                style: AppTextStyles.input
                                    .copyWith(color: _primaryText(context)),
                              ),
                            ),
                            if (AppPreferences.instance.themeMode == mode)
                              Icon(Icons.check,
                                  size: 20, color: _primaryText(context)),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
    if (selected != null) {
      await AppPreferences.instance.setThemeMode(selected);
    }
  }

  // ---- Account actions ----

  Future<void> _confirmLogOut() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Log out?'),
        content: const Text(
            'You will need to sign in again to see your accounts.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            child: const Text('Log Out'),
          ),
        ],
      ),
    );
    if (confirmed ?? false) {
      // AuthGate reacts to the auth-state change and shows Sign In.
      await FirebaseService.instance.signOut();
    }
  }

  @override
  Widget build(BuildContext context) {
    final uid = FirebaseService.instance.currentUser?.uid;
    return Scaffold(
      backgroundColor: _background(context),
      body: uid == null
          ? const Center(child: Text('Not signed in'))
          : SafeArea(
              child: StreamBuilder<DocumentSnapshot<Map<String, dynamic>>>(
                stream: FirebaseService.instance.userDocStream(uid),
                builder: (context, snap) {
                  final doc = snap.data;
                  final user = (doc != null && doc.exists)
                      ? AppUser.fromDoc(doc)
                      : null;
                  return ListView(
                    padding: const EdgeInsets.fromLTRB(10, 12, 10, 24),
                    children: [
                      _ProfileCard(
                        user: user,
                        onTap: user == null
                            ? null
                            : () => _openEditProfile(user),
                      ),
                      const SizedBox(height: 20),
                      const _SectionLabel('ACCOUNT'),
                      const SizedBox(height: 8),
                      _accountCard(user),
                      const SizedBox(height: 20),
                      const _SectionLabel('CONNECTIONS'),
                      const SizedBox(height: 8),
                      _ConnectionsCard(
                        uid: uid,
                        connecting: _connecting,
                        onConnect: _connect,
                      ),
                      const SizedBox(height: 20),
                      const _SectionLabel('NOTIFICATIONS & ALERTS'),
                      const SizedBox(height: 8),
                      _NotificationsCard(uid: uid),
                      const SizedBox(height: 20),
                      const _SectionLabel('APP PREFERENCES'),
                      const SizedBox(height: 8),
                      _preferencesCard(),
                      const SizedBox(height: 20),
                      const _SectionLabel('ACCOUNT ACTIONS'),
                      const SizedBox(height: 8),
                      _SettingsCard(
                        rows: [
                          _SettingsRow(
                            leading: const _RowIcon(Icons.logout),
                            title: 'Log Out',
                            onTap: _confirmLogOut,
                          ),
                          _SettingsRow(
                            leading: const _RowIcon(Icons.delete_outline,
                                danger: true),
                            title: 'Delete Account',
                            subtitle: 'Permanently erase your data',
                            titleColor:
                                AppTheme.alertRed(context, AppColors.danger),
                            onTap: () => Navigator.of(context).push(
                              MaterialPageRoute(
                                  builder: (_) => const DeleteAccountScreen()),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 24),
                      Center(
                        child: Text(
                          _kVersionLabel,
                          style: GoogleFonts.inter(
                            fontWeight: FontWeight.w500,
                            fontSize: 11,
                            color: _muted(context),
                          ),
                        ),
                      ),
                    ],
                  );
                },
              ),
            ),
    );
  }

  void _openEditProfile(AppUser user) {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => EditProfileScreen(user: user)),
    );
  }

  Widget _accountCard(AppUser? user) {
    final email = FirebaseService.instance.currentUser?.email ?? '-';
    return _SettingsCard(
      rows: [
        _SettingsRow(
          leading: const _RowIcon(Icons.person_outline),
          title: 'Edit Profile',
          subtitle: 'Name, date of birth, income, dependants',
          onTap: user == null ? null : () => _openEditProfile(user),
        ),
        _SettingsRow(
          leading: const _RowIcon(Icons.lock_outline),
          title: 'Password & Security',
          subtitle: 'Change password',
          onTap: () => Navigator.of(context).push(
            MaterialPageRoute(builder: (_) => const ChangePasswordScreen()),
          ),
        ),
        // Display only this pass: nothing else in the app anticipates an
        // email change, and doing it properly means reauthentication plus
        // a verify-before-update round trip.
        _SettingsRow(
          leading: const _RowIcon(Icons.mail_outline),
          title: 'Email Address',
          subtitle: email,
        ),
      ],
    );
  }

  Widget _preferencesCard() {
    final prefs = AppPreferences.instance;
    return _SettingsCard(
      rows: [
        _SettingsRow(
          leading: const _RowIcon(Icons.dark_mode_outlined),
          title: 'Theme',
          subtitle: 'Light, dark or system default',
          trailing: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                AppPreferences.themeModeLabel(prefs.themeMode),
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w600,
                  fontSize: 11,
                  color: _muted(context),
                ),
              ),
              Icon(Icons.chevron_right, color: _muted(context)),
            ],
          ),
          onTap: _pickTheme,
        ),
        _SettingsRow(
          leading: const _RowIcon(Icons.visibility_outlined),
          title: 'Hide Balances by Default',
          subtitle: 'Mask all values on launch',
          trailing: _LimeSwitch(
            value: prefs.hideBalances,
            onChanged: (v) => prefs.setHideBalances(v),
          ),
          onTap: () => prefs.setHideBalances(!prefs.hideBalances),
        ),
      ],
    );
  }
}

// ---- Theme-aware surfaces -------------------------------------------------
// The other four dashboard screens hardcode their light palette; Settings
// resolves its own so the Theme preference visibly takes effect. Light-mode
// values are identical to the shared tokens, so nothing shifts by default.

Color _background(BuildContext context) =>
    Theme.of(context).brightness == Brightness.dark
        ? _kDarkBackground
        : AppColors.background;

Color _surface(BuildContext context) =>
    Theme.of(context).brightness == Brightness.dark
        ? _kDarkSurface
        : Colors.white;

Color _primaryText(BuildContext context) =>
    Theme.of(context).brightness == Brightness.dark
        ? Colors.white
        : AppColors.brandBlack;

/// Secondary text — section labels, subtitles, chevrons, the version footer.
///
/// Light mode is the WCAG AA fix: #AAAAAA on white measures about 2.3:1,
/// nowhere near the 4.5:1 minimum for normal text. #6B6B6B clears it at 5.3:1.
///
/// Dark mode deliberately does NOT follow it down. This was one flat constant
/// before ([AppTheme.muted]) precisely because #AAAAAA already clears 6.9:1 on
/// the #1E1E1E card — applying the light fix to both modes would drag dark
/// mode from 6.9:1 to roughly 2.9:1 and break the very thing being fixed. The
/// constant had to become a pair for the fix to be correct in both modes.
Color _muted(BuildContext context) =>
    Theme.of(context).brightness == Brightness.dark
        ? AppTheme.muted
        : const Color(0xFF6B6B6B);

Color _divider(BuildContext context) =>
    Theme.of(context).brightness == Brightness.dark
        ? const Color(0xFF2E2E2E)
        : const Color(0xFFEEEEEE);

class _SectionLabel extends StatelessWidget {
  const _SectionLabel(this.text);

  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(left: 4),
      child: Text(
        text,
        style: GoogleFonts.inter(
          fontWeight: FontWeight.w600,
          fontSize: 11,
          letterSpacing: 0.4,
          color: _muted(context),
        ),
      ),
    );
  }
}

/// White rounded-20 container holding hairline-separated rows — the shape
/// every grouped section in the frame uses.
class _SettingsCard extends StatelessWidget {
  const _SettingsCard({required this.rows});

  final List<Widget> rows;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: _surface(context),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Column(
        children: [
          for (var i = 0; i < rows.length; i++) ...[
            if (i > 0)
              Divider(height: 1, thickness: 1, color: _divider(context)),
            rows[i],
          ],
        ],
      ),
    );
  }
}

class _SettingsRow extends StatelessWidget {
  const _SettingsRow({
    required this.leading,
    required this.title,
    this.subtitle,
    this.trailing,
    this.onTap,
    this.titleColor,
    this.subtitleMaxLines = 1,
  });

  final Widget leading;
  final String title;
  final String? subtitle;
  final Widget? trailing;
  final VoidCallback? onTap;

  /// Overrides the theme-resolved title colour. Only Delete Account sets it.
  final Color? titleColor;

  /// Every row clips its subtitle to one line, which is right for the short
  /// fixed strings the rest of Settings uses. A merged Connections row is the
  /// exception: its subtitle sums several Items AND says how many connections
  /// it stands for, and at one line the connection count — the very thing that
  /// explains why disconnect asks which one — was the part being ellipsed off
  /// the end. Opt-in, so no existing row changes.
  final int subtitleMaxLines;

  @override
  Widget build(BuildContext context) {
    final row = Padding(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      child: Row(
        children: [
          leading,
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: GoogleFonts.spaceGrotesk(
                    fontWeight: FontWeight.w700,
                    fontSize: 14,
                    color: titleColor ?? _primaryText(context),
                  ),
                ),
                if (subtitle != null)
                  Text(
                    subtitle!,
                    maxLines: subtitleMaxLines,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.inter(
                      fontWeight: FontWeight.w500,
                      fontSize: 11,
                      color: _muted(context),
                    ),
                  ),
              ],
            ),
          ),
          // A row with no action gets no chevron — the frame draws one on
          // every row, but Email Address does nothing this pass.
          trailing ??
              (onTap == null
                  ? const SizedBox.shrink()
                  : Icon(Icons.chevron_right, color: _muted(context))),
        ],
      ),
    );
    if (onTap == null) return row;
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(20),
      child: row,
    );
  }
}

/// A [Switch] whose ON state is the brand lime rather than the seed colour.
///
/// Lime is the app's active/selected accent, and a switch the user has
/// deliberately turned on is exactly the "genuine action or selection" it is
/// reserved for — not a passive status indicator, which the convention
/// excludes.
///
/// The thumb is white, sampled from the audit frame. The written spec asked
/// for a dark thumb "for contrast"; that clause turned out to describe dark
/// ink on a lime *fill*, not this control, and the frame is the authority
/// here — confirmed at #FFFFFF on the #C9F158 track.
///
/// OFF keeps Material's own resolution, which already differs correctly
/// between light and dark; only the selected state is overridden.
class _LimeSwitch extends StatelessWidget {
  const _LimeSwitch({required this.value, required this.onChanged});

  final bool value;

  /// Null disables the switch, exactly as [Switch.onChanged] does — the
  /// category rows rely on that while the master toggle is off.
  final ValueChanged<bool>? onChanged;

  @override
  Widget build(BuildContext context) {
    return Switch(
      value: value,
      onChanged: onChanged,
      // WidgetStateProperty rather than the activeColor family: that shorthand
      // is deprecated in this Flutter and would land as an analyze warning.
      thumbColor: WidgetStateProperty.resolveWith((states) =>
          states.contains(WidgetState.selected) ? Colors.white : null),
      trackColor: WidgetStateProperty.resolveWith((states) =>
          states.contains(WidgetState.selected) ? AppColors.limeAccent : null),
      // Left as-is the outline stays the seed blue on an otherwise lime track.
      trackOutlineColor: WidgetStateProperty.resolveWith((states) =>
          states.contains(WidgetState.selected)
              ? AppColors.limeAccent
              : null),
    );
  }
}

/// 32×32 rounded tile holding one row's icon.
///
/// These were emoji until the audit. Emoji render from the platform font, so
/// they carried their own colour, their own weight and their own idea of a
/// baseline — the reason the badges never quite lined up with each other. They
/// are now [Icons], the same Material set [MainShell]'s bottom nav already
/// uses, so Settings introduces no second icon system.
///
/// Every badge shares one neutral tile. [danger] is the single exception: a
/// destructive row should not be visually interchangeable with Log Out sitting
/// directly above it.
class _RowIcon extends StatelessWidget {
  const _RowIcon(this.icon, {this.danger = false});

  final IconData icon;
  final bool danger;

  @override
  Widget build(BuildContext context) {
    // Routed through AppTheme.chip so the warning tint inverts the same way
    // every other pale-fill/dark-ink badge in the app does, rather than
    // staying a near-white tile on a #1E1E1E card in dark mode.
    final tint = AppTheme.chip(
      context,
      background: AppColors.dangerTint,
      foreground: AppColors.danger,
    );
    return Container(
      width: 32,
      height: 32,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        color: danger ? tint.background : AppTheme.iconTile(context),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Icon(
        icon,
        size: 18,
        color: danger ? tint.foreground : _primaryText(context),
      ),
    );
  }
}

/// Profile card — name + email only, per the brief's Section 14 constraint.
///
/// A dark hero card, matching the audit frame: #202020 fill, white name,
/// #9A9A9A email, #898989 chevron, all sampled from it. It had drifted to a
/// plain white card at some point before the audit; the #EDEDED avatar was
/// always designed to sit on this dark fill, which is why that value looked
/// arbitrary against white.
///
/// Its colours are fixed rather than theme-resolved, since the card is dark in
/// both modes — with one exception. In dark mode the fill lifts to #2A2A2A,
/// the same value [AppTheme.heroSurface] uses for the hero cards on Savings,
/// Investments and Insurance, and for the same documented reason: #202020 on
/// the #121212 background sits a hair under the #1E1E1E of every ordinary card
/// around it, so the one card meant to stand out would recede instead. The two
/// greys still clear AA on the lifted fill (#9A9A9A ≈ 4.5:1 on #2A2A2A).
class _ProfileCard extends StatelessWidget {
  const _ProfileCard({required this.user, this.onTap});

  final AppUser? user;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final email = user?.email.isNotEmpty ?? false
        ? user!.email
        : FirebaseService.instance.currentUser?.email ?? '-';
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(20),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
        decoration: BoxDecoration(
          color: Theme.of(context).brightness == Brightness.dark
              ? const Color(0xFF2A2A2A)
              : AppColors.brandBlack,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          children: [
            // #EDEDED, sampled from the audit frame, rather than the
            // #F2F3F5 icon-tile grey this used to borrow. The audit called
            // this out as a Material 3 default purple (#EADDFF) needing a
            // neutral — that purple was never in this code, which had already
            // been re-themed, but the frame's own neutral is #EDEDED and the
            // avatar is a photo placeholder rather than an icon badge, so it
            // takes its own value instead of the shared tile.
            const CircleAvatar(
              radius: 26,
              backgroundColor: Color(0xFFEDEDED),
              child: Icon(Icons.person, color: Colors.black54, size: 28),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    user?.name ?? '-',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.spaceGrotesk(
                      fontWeight: FontWeight.w700,
                      fontSize: 18,
                      color: Colors.white,
                    ),
                  ),
                  Text(
                    email,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.inter(
                      fontWeight: FontWeight.w500,
                      fontSize: 12,
                      color: const Color(0xFF9A9A9A),
                    ),
                  ),
                ],
              ),
            ),
            const Icon(Icons.chevron_right, color: Color(0xFF898989)),
          ],
        ),
      ),
    );
  }
}

/// Asks WHICH connection, when one institution row covers more than one Plaid
/// Item. Returns the chosen Item, or null if the user backed out.
///
/// Only shown when it is genuinely needed — a single-Item institution goes
/// straight to the confirmation dialog, so the common case gains no friction.
///
/// On the detail shown: account counts alone do NOT distinguish two Items from
/// the same institution (Plaid's Sandbox gives both the same ten accounts and
/// the same masks — verified against real data), so each entry leads with its
/// last-synced date, which does differ. Entries are also numbered, because two
/// connections made minutes apart could still round to the same displayed date
/// and the user must always have something to tell them apart by.
///
/// Top-level and public for the same reason as [showDisconnectConfirmation]:
/// this screen is Firestore-backed and not widget-tested, so the picker's
/// selection contract is tested directly instead.
Future<ConnectedItem?> showDisconnectItemPicker(
  BuildContext context,
  String institutionName,
  List<ConnectedItem> items,
) {
  return showModalBottomSheet<ConnectedItem>(
    context: context,
    backgroundColor: _surface(context),
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
    ),
    builder: (sheetContext) => SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Which $institutionName connection?',
              style: AppTextStyles.fieldLabel
                  .copyWith(color: _primaryText(sheetContext)),
            ),
            const SizedBox(height: 6),
            Text(
              'You have ${items.length} separate $institutionName connections. '
              'Choose the one to disconnect. The others stay connected.',
              style: GoogleFonts.inter(
                fontSize: 12,
                height: 1.45,
                color: _muted(sheetContext),
              ),
            ),
            const SizedBox(height: 14),
            // Same rounded-12 option rows as the Theme sheet, so the two
            // bottom sheets in Settings do not look like different components.
            for (final (index, item) in items.indexed)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Material(
                  color: AppTheme.inset(sheetContext),
                  borderRadius: BorderRadius.circular(12),
                  child: InkWell(
                    borderRadius: BorderRadius.circular(12),
                    onTap: () => Navigator.of(sheetContext).pop(item),
                    child: Padding(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 14),
                      child: Row(
                        children: [
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Connection ${index + 1}'
                                  '${item.lastSynced == null ? '' : ' · synced '
                                      '${_shortDate(item.lastSynced!)}'}',
                                  style: AppTextStyles.input.copyWith(
                                    color: _primaryText(sheetContext),
                                    fontSize: 15,
                                  ),
                                ),
                                const SizedBox(height: 2),
                                Text(
                                  item.breakdown,
                                  style: GoogleFonts.inter(
                                    fontSize: 12,
                                    color: _muted(sheetContext),
                                  ),
                                ),
                              ],
                            ),
                          ),
                          Icon(Icons.chevron_right, color: _muted(sheetContext)),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    ),
  );
}

/// "3 Aug 2026" — enough to separate two connections without implying a
/// precision the underlying timestamp does not carry for the user's purposes.
String _shortDate(DateTime date) {
  const months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
  ];
  final local = date.toLocal();
  return '${local.day} ${months[local.month - 1]} ${local.year}';
}

/// The gate in front of a disconnect. Returns true only if the user explicitly
/// confirms — false on Cancel, and null if the dialog is dismissed by a barrier
/// tap or the back button, both of which callers must treat as "do nothing".
///
/// Deliberately lighter friction than Delete Account's typed "DELETE"
/// confirmation: that destroys the whole account irreversibly, while this
/// removes one connection's synced data and can be undone by reconnecting. The
/// consequences are still stated in full rather than implied.
///
/// Top-level and public so the gate itself is testable — the screen it lives on
/// is backed by Firestore streams and is not widget-tested (see
/// test/settings_disconnect_test.dart).
Future<bool?> showDisconnectConfirmation(
  BuildContext context,
  String institutionName,
) {
  return showDialog<bool>(
    context: context,
    builder: (dialogContext) => AlertDialog(
      backgroundColor: _surface(dialogContext),
      title: Text(
        'Disconnect $institutionName?',
        style: AppTextStyles.fieldLabel
            .copyWith(color: _primaryText(dialogContext), fontSize: 17),
      ),
      content: Text(
        'This will remove all accounts and transaction history from '
        '$institutionName, and revoke Unify’s access to them. '
        'This cannot be undone.\n\n'
        'You can reconnect $institutionName later.',
        style: GoogleFonts.inter(
          fontSize: 13,
          height: 1.45,
          color: _muted(dialogContext),
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(dialogContext).pop(false),
          child: Text(
            'Cancel',
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w600,
              color: _primaryText(dialogContext),
            ),
          ),
        ),
        TextButton(
          onPressed: () => Navigator.of(dialogContext).pop(true),
          child: Text(
            'Disconnect',
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w600,
              color: AppTheme.alertRed(dialogContext, AppColors.danger),
            ),
          ),
        ),
      ],
    ),
  );
}

/// One merged Connections section (frame node 134:481): every connected
/// Plaid Item with its banking-vs-investment account split, plus the single
/// Connect CTA.
///
/// plaidItems — where institution_name canonically lives — is backend-only
/// by firestore.rules, so institutions are read from the denormalized
/// `institution_name` on plaidAccounts. investmentAccounts carry item_id but
/// no institution name, so an investment-only Item falls back to its account
/// name.
class _ConnectionsCard extends StatefulWidget {
  const _ConnectionsCard({
    required this.uid,
    required this.connecting,
    required this.onConnect,
  });

  final String uid;
  final bool connecting;
  final VoidCallback onConnect;

  @override
  State<_ConnectionsCard> createState() => _ConnectionsCardState();
}

class _ConnectionsCardState extends State<_ConnectionsCard> {
  /// The Item currently being disconnected, if any. Held per-item rather than
  /// as a single bool so the spinner appears on the row the user actually
  /// tapped, and so a second row cannot be started while one is in flight.
  String? _disconnectingItemId;

  /// Disambiguate if needed, then confirm, then disconnect.
  ///
  /// The picker is a step BEFORE the existing gate, never a replacement for
  /// it: whichever Item is chosen still goes through the same confirmation
  /// dialog, the same wording and the same endpoint as a single-Item
  /// institution. A one-Item group skips the picker entirely, so nothing about
  /// the common path changed.
  Future<void> _confirmDisconnect(InstitutionGroup group) async {
    if (_disconnectingItemId != null) return;
    final candidates = group.disconnectableItems;
    if (candidates.isEmpty) return;

    final ConnectedItem item;
    if (candidates.length == 1) {
      item = candidates.first;
    } else {
      final picked = await showDisconnectItemPicker(
          context, group.institutionName, candidates);
      if (picked == null || !mounted) return; // backed out of the picker
      item = picked;
    }

    final itemId = item.itemId;
    if (itemId == null) return;

    final confirmed =
        await showDisconnectConfirmation(context, group.institutionName);
    if (confirmed != true || !mounted) return;

    setState(() => _disconnectingItemId = itemId);
    String message;
    try {
      await PlaidService.instance.disconnectItem(itemId);
      // No local list to prune: the rows come from the plaidAccounts /
      // investmentAccounts snapshots, which fire as soon as the backend's
      // deletes land, so this row (and the same data on Savings, Investments
      // and Home) disappears on its own.
      message = '${item.institutionName} disconnected.';
    } on PlaidServiceException catch (e) {
      // Every backend failure path leaves the connection intact, so the row
      // stays and the user can simply try again.
      message = e.message;
    } catch (_) {
      message = 'Could not disconnect ${item.institutionName}. Please try '
          'again.';
    }
    if (!mounted) return;
    setState(() => _disconnectingItemId = null);
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message)));
  }

  /// True while any Item inside this institution row is being disconnected.
  bool _groupIsDisconnecting(InstitutionGroup group) =>
      group.items.any((item) => item.itemId == _disconnectingItemId);

  /// The per-row disconnect affordance, or a stand-in when it should not be
  /// offered: a spinner on the row being disconnected, and nothing at all on a
  /// row with no Plaid Item behind it (a manual investment entry — there is no
  /// access_token to revoke, and its deletion is not a Plaid concern).
  Widget _disconnectRowAction(BuildContext context, InstitutionGroup group) {
    if (group.disconnectableItems.isEmpty) return const SizedBox.shrink();
    if (_groupIsDisconnecting(group)) return const _RowSpinner();
    final busy = _disconnectingItemId != null;
    return IconButton(
      icon: const Icon(Icons.link_off, size: 20),
      // Red, matching Delete Account's treatment: a destructive action should
      // not look like the neutral chevrons elsewhere in this card.
      color: AppTheme.alertRed(context, AppColors.danger),
      // Disabled, not hidden, while another row is disconnecting — a control
      // that vanishes mid-interaction is worse than one that is visibly
      // unavailable.
      onPressed: busy ? null : () => _confirmDisconnect(group),
      tooltip: group.needsDisambiguation
          ? 'Disconnect a ${group.institutionName} connection'
          : 'Disconnect ${group.institutionName}',
      visualDensity: VisualDensity.compact,
      constraints: const BoxConstraints(minWidth: 40, minHeight: 40),
      padding: EdgeInsets.zero,
    );
  }

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<QuerySnapshot<Map<String, dynamic>>>(
      stream: FirebaseService.instance.plaidAccounts(widget.uid).snapshots(),
      builder: (context, bankSnap) {
        return StreamBuilder<QuerySnapshot<Map<String, dynamic>>>(
          stream: FirebaseService.instance
              .investmentAccounts(widget.uid)
              .snapshots(),
          builder: (context, investSnap) {
            final summary = ConnectionsSummary.from(
              banking: [
                for (final d in bankSnap.data?.docs ?? [])
                  PlaidAccount.fromFirestore(d.id, d.data()),
              ],
              investments: [
                for (final d in investSnap.data?.docs ?? [])
                  InvestmentAccount.fromFirestore(d.id, d.data()),
              ],
            );
            return _SettingsCard(
              rows: [
                // No per-kind tally here. One Plaid Item usually holds both
                // banking and investment accounts, so counting banks and
                // brokerages separately described the same Items twice. The
                // badge carries the count and the rows below carry the split.
                _SettingsRow(
                  leading: const _RowIcon(Icons.attach_money),
                  title: 'Connections',
                  subtitle:
                      summary.items.isEmpty ? 'Nothing connected yet' : null,
                  // Counts Plaid Items, NOT the merged institution rows. Two
                  // connections to one bank are two live access_tokens, two
                  // things to disconnect and two things Plaid can ask the user
                  // to re-authenticate — so a badge reading "1 Active" beside
                  // a row covering two of them would be the app under-
                  // reporting what it holds. The row count is already visible;
                  // the connection count would not be recoverable anywhere
                  // else once the rows merge.
                  trailing: summary.items.isEmpty
                      ? const SizedBox.shrink()
                      : _ActiveBadge(count: summary.items.length),
                ),
                for (final group in summary.groups)
                  _SettingsRow(
                    leading: const _RowIcon(Icons.link),
                    title: group.institutionName,
                    subtitle: _groupIsDisconnecting(group)
                        ? 'Disconnecting…'
                        : [
                            group.breakdown,
                            // Says plainly that this row stands for more than
                            // one connection, so the picker is expected rather
                            // than a surprise.
                            if (group.needsDisambiguation)
                              '${group.items.length} connections',
                          ].join(' · '),
                    // A merged row carries strictly more text than a single
                    // one and was truncating exactly where it mattered.
                    subtitleMaxLines: group.needsDisambiguation ? 2 : 1,
                    // A trailing icon button rather than swipe-to-dismiss.
                    // These rows sit inside a card in a scrolling ListView,
                    // where a horizontal swipe is both undiscoverable and
                    // easy to trigger by accident on a destructive action —
                    // and _SettingsRow already carries actions in `trailing`
                    // everywhere else in this screen.
                    trailing: _disconnectRowAction(context, group),
                  ),
                _SettingsRow(
                  leading: widget.connecting
                      ? const _RowSpinner()
                      : const _RowIcon(Icons.add),
                  title: 'Connect a bank or broker',
                  subtitle: widget.connecting
                      ? 'Connecting. This can take a few seconds…'
                      : 'Link via Plaid · Secure & encrypted',
                  onTap: widget.connecting ? null : widget.onConnect,
                  trailing: widget.connecting
                      ? const SizedBox.shrink()
                      : Icon(Icons.chevron_right, color: _muted(context)),
                ),
              ],
            );
          },
        );
      },
    );
  }
}

/// Notifications & Alerts — one master toggle plus the three categories the
/// backend sends.
///
/// The master row shows the EFFECTIVE state (stored preference AND OS
/// permission), never the stored preference alone: a switch sitting in the "on"
/// position while Android is dropping every push would be the app asserting
/// something untrue about itself. When permission is missing the row explains
/// that instead, and the three category rows go disabled — they cannot do
/// anything while nothing can be delivered, and leaving them tappable would
/// imply otherwise.
class _NotificationsCard extends StatelessWidget {
  const _NotificationsCard({required this.uid});

  final String uid;

  @override
  Widget build(BuildContext context) {
    final service = NotificationService.instance;
    return ListenableBuilder(
      listenable: service,
      builder: (context, _) {
        final pushOn = service.pushEnabled;
        return _SettingsCard(
          rows: [
            _SettingsRow(
              leading: const _RowIcon(Icons.notifications_none),
              title: 'Push Notifications',
              subtitle: service.permissionBlocked
                  ? 'Blocked in system settings. Enable Unify there'
                  : pushOn
                      ? 'Alerts are on for this account'
                      : 'Turn on to receive alerts',
              trailing: _LimeSwitch(
                value: pushOn,
                onChanged: (v) => _set(context, NotificationKinds.push, v),
              ),
            ),
            _CategoryRow(
              icon: Icons.warning_amber_rounded,
              title: 'Risk Alerts',
              subtitle: 'When a tier drops or a coverage gap appears',
              kind: NotificationKinds.riskAlerts,
              masterOn: pushOn,
              onChanged: (v) =>
                  _set(context, NotificationKinds.riskAlerts, v),
            ),
            _CategoryRow(
              icon: Icons.credit_card,
              title: 'Premium Due Reminders',
              subtitle: 'A week before a policy renews',
              kind: NotificationKinds.premiumDue,
              masterOn: pushOn,
              onChanged: (v) => _set(context, NotificationKinds.premiumDue, v),
            ),
            _CategoryRow(
              icon: Icons.bar_chart,
              // Display copy only. The stored key stays `monthly_score_update`
              // (NotificationKinds.monthlyScore) — renaming the field would
              // reset the choice of every user who has already set it.
              title: 'Score Change Alerts',
              subtitle: 'When your Financial Health Score changes',
              kind: NotificationKinds.monthlyScore,
              masterOn: pushOn,
              onChanged: (v) =>
                  _set(context, NotificationKinds.monthlyScore, v),
            ),
          ],
        );
      },
    );
  }

  Future<void> _set(BuildContext context, String kind, bool value) async {
    final messenger = ScaffoldMessenger.of(context);
    final ok = await NotificationService.instance.setToggle(uid, kind, value);
    if (ok) return;
    // The only way this fails for the master is the OS refusing. Android stops
    // showing the dialog after two refusals and iOS after one, so re-prompting
    // would do nothing — the user has to be sent to system settings.
    messenger.showSnackBar(
      SnackBar(
        content: Text(kind == NotificationKinds.push
            ? 'Notifications are blocked for Unify. Turn them on in your '
                'device Settings > Apps > Unify > Notifications.'
            : 'Could not save that setting. Please try again.'),
      ),
    );
  }
}

/// One category toggle. Greyed out while the master is off — the stored
/// position is still shown, so turning the master back on restores exactly the
/// categories the user had chosen rather than resetting them.
class _CategoryRow extends StatelessWidget {
  const _CategoryRow({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.kind,
    required this.masterOn,
    required this.onChanged,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final String kind;
  final bool masterOn;
  final ValueChanged<bool> onChanged;

  @override
  Widget build(BuildContext context) {
    final stored = NotificationService.instance.storedValue(kind);
    return Opacity(
      opacity: masterOn ? 1 : 0.45,
      child: _SettingsRow(
        leading: _RowIcon(icon),
        title: title,
        subtitle: subtitle,
        trailing: _LimeSwitch(
          value: stored,
          onChanged: masterOn ? onChanged : null,
        ),
      ),
    );
  }
}

class _ActiveBadge extends StatelessWidget {
  const _ActiveBadge({required this.count});

  final int count;

  @override
  Widget build(BuildContext context) {
    // The same pale-green/dark-ink pair the rest of the app uses for a healthy
    // state (Insights' positive pill, the insurance card's "Parsed" badge), so
    // it goes through chip() rather than getting a green of its own.
    final chip = AppTheme.chip(
      context,
      background: const Color(0xFFE8F5E9),
      foreground: const Color(0xFF2E7D32),
    );
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: chip.background,
        borderRadius: BorderRadius.circular(9),
      ),
      child: Text(
        '$count Active',
        style: GoogleFonts.inter(
          fontWeight: FontWeight.w600,
          fontSize: 10,
          color: chip.foreground,
        ),
      ),
    );
  }
}

class _RowSpinner extends StatelessWidget {
  const _RowSpinner();

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 32,
      height: 32,
      child: Center(
        child: SizedBox(
          width: 16,
          height: 16,
          child: CircularProgressIndicator(
            strokeWidth: 2,
            color: _primaryText(context),
          ),
        ),
      ),
    );
  }
}

/// "2 banking accounts · 1 investment account" — omits whichever side is empty.
/// Shared by the merged institution rows and the disconnect picker's entries so
/// the two can never describe the same counts differently.
String _accountBreakdown(int banking, int investment) => [
      if (banking > 0) '$banking banking ${banking == 1 ? 'account' : 'accounts'}',
      if (investment > 0)
        '$investment investment ${investment == 1 ? 'account' : 'accounts'}',
    ].join(' · ');

/// One connected Plaid Item, counted across both account collections. Still the
/// unit DISCONNECT works on — only the display is grouped by institution.
class ConnectedItem {
  ConnectedItem(this.institutionName, this.itemId);

  String institutionName;

  /// The Plaid Item this entry represents, and the only thing Disconnect sends.
  /// Null for an investment account with no item_id — a manual entry per
  /// Section 8, which has no Plaid Item behind it to revoke.
  final String? itemId;

  int bankingCount = 0;
  int investmentCount = 0;

  /// Latest `last_synced` across this Item's banking accounts.
  ///
  /// Load-bearing for the picker, not decoration: two Items from the same
  /// institution carry identical account counts AND identical Plaid Sandbox
  /// masks, so this is the only field that actually tells them apart. Null for
  /// an investment-only Item (investmentAccounts has no client-readable
  /// timestamp), which is why the picker falls back to ordinals as well.
  DateTime? lastSynced;

  String get breakdown => _accountBreakdown(bankingCount, investmentCount);

  void countBanking(PlaidAccount account) {
    bankingCount++;
    final synced = account.lastSynced;
    if (synced != null &&
        (lastSynced == null || synced.isAfter(lastSynced!))) {
      lastSynced = synced;
    }
  }
}

/// Every Item sharing one institution name — one Connections row.
///
/// The row is the institution; the disconnect underneath is still per-Item,
/// which is the whole reason [items] stays a list rather than collapsing into
/// summed counts.
class InstitutionGroup {
  InstitutionGroup(this.institutionName);

  final String institutionName;
  final List<ConnectedItem> items = [];

  int get bankingCount =>
      items.fold(0, (total, item) => total + item.bankingCount);
  int get investmentCount =>
      items.fold(0, (total, item) => total + item.investmentCount);

  /// Counts summed across every Item in the group, so a merged row describes
  /// what the user actually has connected at that institution.
  String get breakdown => _accountBreakdown(bankingCount, investmentCount);

  /// True when disconnecting needs to ask WHICH connection first.
  bool get needsDisambiguation => disconnectableItems.length > 1;

  /// Items that can actually be revoked. A manual investment entry has no
  /// Plaid Item behind it, so it is displayed but never offered for disconnect.
  List<ConnectedItem> get disconnectableItems =>
      [for (final item in items) if (item.itemId != null) item];
}

class ConnectionsSummary {
  const ConnectionsSummary(this.groups);

  /// One entry per institution name, insertion-ordered by first appearance.
  final List<InstitutionGroup> groups;

  /// Every underlying Plaid Item, flattened. The count of real connections,
  /// which is not the same as [groups].length once two Items share a name.
  List<ConnectedItem> get items =>
      [for (final group in groups) ...group.items];

  /// Builds the per-Item tally first, then groups it by institution name.
  ///
  /// The two passes are ordered deliberately, not incidentally: institution
  /// names live ONLY on plaidAccounts (plaidItems is not client-readable, and
  /// investmentAccounts has no such field at all — brief Section 8), so every
  /// banking account must be seen before any investment account can be
  /// attached to a named group. Doing it the other way round would name a
  /// mixed Item after its brokerage account.
  factory ConnectionsSummary.from({
    required List<PlaidAccount> banking,
    required List<InvestmentAccount> investments,
  }) {
    // --- pass 1: banking accounts establish each Item AND its institution.
    final byItem = <String, ConnectedItem>{};
    for (final account in banking) {
      final item = byItem.putIfAbsent(
        account.itemId,
        () => ConnectedItem(
          account.institutionName ?? 'Connected bank',
          account.itemId,
        ),
      );
      item.countBanking(account);
    }

    // --- pass 2: investment accounts join their Item's existing group, or
    // create one. An Item reached only here has no banking account and so no
    // readable institution name — the pre-existing account-name fallback
    // stands in, unchanged.
    for (final account in investments) {
      final item = byItem.putIfAbsent(
        account.itemId ?? account.accountId,
        () => ConnectedItem(account.accountName, account.itemId),
      );
      item.investmentCount++;
    }

    // --- pass 3: merge Items sharing an institution name into one row.
    final byInstitution = <String, InstitutionGroup>{};
    for (final item in byItem.values) {
      byInstitution
          .putIfAbsent(
              item.institutionName, () => InstitutionGroup(item.institutionName))
          .items
          .add(item);
    }
    return ConnectionsSummary(byInstitution.values.toList());
  }
}
