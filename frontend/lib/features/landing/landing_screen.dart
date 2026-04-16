import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../../api/api_client.dart';
import '../../models/predict_request.dart';
import '../../shared/ui/app_layout.dart';
import '../input/input_screen.dart';
import '../result/result_screen.dart';

class LandingScreen extends StatefulWidget {
  const LandingScreen({super.key});

  @override
  State<LandingScreen> createState() => _LandingScreenState();
}

class _LandingScreenState extends State<LandingScreen> {
  static const String _repositoryUrl = 'https://github.com/elfarr/diploma';

  Future<void> _openDemo() async {
    final navigator = Navigator.of(context);
    PredictRequest? request;
    try {
      request = await navigator.pushNamed<PredictRequest>('/app');
    } catch (_) {
      request = await navigator.push<PredictRequest>(
        MaterialPageRoute<PredictRequest>(
          builder: (_) => const InputScreen(),
        ),
      );
    }
    if (!mounted || request == null) {
      return;
    }

    await navigator.push(
      MaterialPageRoute<void>(
        builder: (_) => ResultScreen(request: request!, apiClient: ApiClient()),
      ),
    );
  }

  Future<void> _openRepository() async {
    final uri = Uri.parse(_repositoryUrl);
    if (!await launchUrl(uri, mode: LaunchMode.externalApplication) && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Не удалось открыть ссылку на репозиторий')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(title: const Text('Прогноз риска')),
      body: SingleChildScrollView(
        child: AppResponsiveContainer(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              AppCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Оценка риска осложнений после трансплантации почки',
                      style: theme.textTheme.headlineSmall?.copyWith(
                        fontWeight: FontWeight.w800,
                        height: 1.2,
                      ),
                    ),
                    const SizedBox(height: 10),
                    Text(
                      'Ввод данных пациента, расчет прогноза и краткая интерпретация результата',
                      style: theme.textTheme.bodyLarge?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                ),
              ),
              AppCard(
                child: Column(
                  children: const [
                    _BulletTile(
                      icon: Icons.monitor_heart_outlined,
                      text: 'Расчет риска по клиническим данным пациента',
                    ),
                    _BulletTile(
                      icon: Icons.analytics_outlined,
                      text: 'Результат: вероятность, статус решения и факторы, на которые стоит обратить внимание',
                    ),
                  ],
                ),
              ),
              const AppWarningCard(
                text:
                    'Не является медицинским изделием. Решение принимает врач.',
              ),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: _openDemo,
                  icon: const Icon(Icons.play_arrow_rounded),
                  label: const Text('Приступить к расчету'),
                ),
              ),
              const SizedBox(height: 10),
              SizedBox(
                width: double.infinity,
                child: OutlinedButton.icon(
                  onPressed: _openRepository,
                  icon: const Icon(Icons.open_in_new_rounded),
                  label: const Text('Репозиторий'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _BulletTile extends StatelessWidget {
  const _BulletTile({required this.icon, required this.text});

  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) {
    final color = Theme.of(context).colorScheme.onSurfaceVariant;
    return ListTile(
      contentPadding: EdgeInsets.zero,
      dense: true,
      visualDensity: const VisualDensity(horizontal: -4, vertical: -3),
      leading: Icon(icon, color: color),
      title: Text(text),
    );
  }
}
