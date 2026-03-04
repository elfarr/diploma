import 'package:flutter/material.dart';

import 'api/api_client.dart';
import 'features/input/input_screen.dart';
import 'features/predict/predict_screen.dart';
import 'features/result/result_screen.dart';
import 'models/predict_request.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Risk Predictor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const _HomeScreen(),
    );
  }
}

class _HomeScreen extends StatefulWidget {
  const _HomeScreen();

  @override
  State<_HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<_HomeScreen> {
  Future<void> _openInput() async {
    final result = await Navigator.of(context).push<PredictRequest>(
      MaterialPageRoute<PredictRequest>(
        builder: (_) => const InputScreen(),
      ),
    );
    if (!mounted || result == null) {
      return;
    }

    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => ResultScreen(
          request: result,
          apiClient: ApiClient(),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Risk Predictor')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            ElevatedButton(
              onPressed: () {
                Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (_) => const PredictScreen(),
                  ),
                );
              },
              child: const Text('Открыть Прогноз'),
            ),
            const SizedBox(height: 12),
            ElevatedButton(
              onPressed: _openInput,
              child: const Text('Открыть InputScreen'),
            ),
          ],
        ),
      ),
    );
  }
}
