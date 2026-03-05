import 'package:flutter/material.dart';

import 'features/input/input_screen.dart';
import 'features/landing/landing_screen.dart';
import 'shared/ui/app_theme.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Risk Predictor',
      theme: AppTheme.light(),
      initialRoute: '/',
      routes: {
        '/': (_) => const LandingScreen(),
        '/app': (_) => const InputScreen(),
      },
    );
  }
}
