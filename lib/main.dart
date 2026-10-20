import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/material.dart';

import 'firebase_options.dart';
import 'services/notification_service.dart';
import 'screens/auth/auth_gate.dart';
import 'screens/home_screen.dart';
import 'screens/savings_screen.dart';
import 'screens/investments_screen.dart';
import 'screens/insurance_screen.dart';
import 'screens/settings_screen.dart';
import 'services/app_preferences.dart';
import 'utils/app_styles.dart';

/// Runs in its own isolate when a push arrives with the app terminated or
/// backgrounded, so it must be a top-level function and must initialise
/// Firebase itself — the isolate shares no state with main().
///
/// Intentionally minimal: every notification this backend sends carries a
/// `notification` payload, which Android and iOS draw in the system tray
/// without the app being involved. This handler exists so the plugin has one
/// registered (it warns otherwise) and as the place background data handling
/// would go if a silent push is ever added.
@pragma('vm:entry-point')
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);
  // Awaited so the first frame already carries the stored theme choice.
  await AppPreferences.instance.load();
  runApp(const UnifyApp());
}

class UnifyApp extends StatelessWidget {
  const UnifyApp({super.key});

  @override
  Widget build(BuildContext context) {
    // Rebuilds MaterialApp when Settings changes the theme preference.
    return ListenableBuilder(
      listenable: AppPreferences.instance,
      builder: (context, _) => MaterialApp(
        title: 'Unify',
        debugShowCheckedModeBanner: false,
        // Lets a foreground push show a snackbar from NotificationService,
        // which has no BuildContext of its own.
        scaffoldMessengerKey: NotificationService.messengerKey,
        theme: AppTheme.lightTheme,
        darkTheme: AppTheme.darkTheme,
        themeMode: AppPreferences.instance.themeMode,
        home: const AuthGate(),
      ),
    );
  }
}

class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int _selectedIndex = 0;

  /// Indexes of InvestmentsScreen and InsuranceScreen in [_screens] and the
  /// nav bar.
  static const _investmentsTab = 2;
  static const _insuranceTab = 3;

  /// Rebuilt (not const) on every build so a Hide Balances toggle actually
  /// reaches the screens: a `const` child is skipped by the element tree
  /// even when its parent rebuilds. Fresh widget instances of the same type
  /// in the same slot reuse their existing State, so streams and scroll
  /// positions survive the toggle — only build() re-runs.
  List<Widget> get _screens => [
        HomeScreen(
          onOpenInsurance: () => _onItemTapped(_insuranceTab),
          onOpenInvestments: () => _onItemTapped(_investmentsTab),
        ),
        SavingsScreen(),
        InvestmentsScreen(),
        InsuranceScreen(),
        SettingsScreen(),
      ];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // IndexedStack keeps all screens alive and preserves scroll state;
      // ListenableBuilder repaints them when Hide Balances is toggled.
      body: ListenableBuilder(
        listenable: AppPreferences.instance,
        builder: (context, _) => IndexedStack(
          index: _selectedIndex,
          children: _screens,
        ),
      ),
      // bottomNavigationBar ensures content never scrolls behind the nav bar —
      // Flutter automatically adjusts MediaQuery bottom padding for the body.
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
        type: BottomNavigationBarType.fixed,
        selectedItemColor: Theme.of(context).colorScheme.primary,
        unselectedItemColor: Colors.grey,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home_outlined),
            activeIcon: Icon(Icons.home),
            label: 'Home',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.savings_outlined),
            activeIcon: Icon(Icons.savings),
            label: 'Savings',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.show_chart),
            activeIcon: Icon(Icons.show_chart),
            label: 'Invest',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.shield_outlined),
            activeIcon: Icon(Icons.shield),
            label: 'Insurance',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.settings_outlined),
            activeIcon: Icon(Icons.settings),
            label: 'Settings',
          ),
        ],
      ),
    );
  }
}
