import 'dart:async';

import 'package:flutter/material.dart';

import '../../api/api_client.dart';
import '../../models/explain_item.dart';
import '../../models/predict_request.dart';
import '../../models/predict_response.dart';
import '../../shared/risk_ui.dart';
import '../../shared/ui/app_layout.dart';

class ResultScreen extends StatefulWidget {
  const ResultScreen({
    super.key,
    required this.request,
    required this.apiClient,
    this.title = 'Результат',
  });

  final PredictRequest request;
  final ApiClient apiClient;
  final String title;

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  late Future<PredictResponse> _future;

  @override
  void initState() {
    super.initState();
    _future = widget.apiClient
        .predict(widget.request)
        .timeout(const Duration(seconds: 15));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(widget.title)),
      body: FutureBuilder<PredictResponse>(
        future: _future,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }

          if (snapshot.hasError) {
            final message = _errorText(snapshot.error);
            return AppResponsiveContainer(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  AppWarningCard(text: message),
                  OutlinedButton(
                    onPressed: () => Navigator.of(context).pop(),
                    child: const Text('Назад к вводу'),
                  ),
                ],
              ),
            );
          }

          final data = snapshot.data;
          if (data == null) {
            return const Center(child: Text('Нет данных'));
          }
          return _buildSuccess(context, data);
        },
      ),
    );
  }

  Widget _buildSuccess(BuildContext context, PredictResponse data) {
    final pCal = data.probCal;
    final top = _topFactors(data.explain);
    final isUncertain =
        (data.undetermined ?? false) || (pCal != null && isUndetermined(pCal));

    return ListView(
      children: [
        AppResponsiveContainer(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              AppCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      data.klass ?? '-',
                      style: Theme.of(context).textTheme.headlineMedium
                          ?.copyWith(fontWeight: FontWeight.w800),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      pCal != null ? probText(pCal) : 'Вероятность (калибр.): -',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    if (pCal != null) ...[
                      const SizedBox(height: 12),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(99),
                        child: LinearProgressIndicator(
                          minHeight: 9,
                          value: pCal.clamp(0, 1),
                        ),
                      ),
                    ],
                    const SizedBox(height: 12),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: [
                        _ConfidenceChip(
                          text:
                              pCal != null
                                  ? confidenceBadge(pCal)
                                  : 'Уверенность: нет данных',
                        ),
                        if (data.badge != null && data.badge!.trim().isNotEmpty)
                          _ConfidenceChip(text: data.badge!),
                      ],
                    ),
                  ],
                ),
              ),
              if (isUncertain) AppWarningCard(text: undeterminedText()),
              const AppSectionTitle('Топ факторы'),
              if (top.isEmpty)
                AppCard(
                  child: Text(
                    'Нет объяснений',
                    style: Theme.of(context).textTheme.bodyLarge,
                  ),
                )
              else
                ...top.map(_factorTile),
              const SizedBox(height: 8),
              OutlinedButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text('Назад к вводу'),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _factorTile(ExplainItem item) {
    final positive = item.contribution >= 0;
    final icon = positive ? Icons.north_east_rounded : Icons.south_east_rounded;

    return AppCard(
      child: Row(
        children: [
          Icon(
            icon,
            size: 18,
            color: positive ? Colors.red.shade700 : Colors.green.shade700,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              item.name,
              style: const TextStyle(fontWeight: FontWeight.w700),
            ),
          ),
        ],
      ),
    );
  }

  List<ExplainItem> _topFactors(List<ExplainItem> source) {
    final list = List<ExplainItem>.from(source);
    list.sort((a, b) => b.contribution.abs().compareTo(a.contribution.abs()));
    return list.length > 5 ? list.sublist(0, 5) : list;
  }

  String _errorText(Object? error) {
    if (error is TimeoutException) {
      return 'Превышено время ожидания. Проверь интернет.';
    }
    if (error is ApiException) {
      if (error.statusCode == 401) {
        return 'Нет доступа (токен).';
      }
      if (error.statusCode == 422) {
        final details = error.message.trim();
        if (details.isNotEmpty) {
          return 'Ошибка данных (валидация): $details';
        }
        return 'Ошибка данных (валидация). Проверь введённые значения.';
      }
      return 'Ошибка сервера: ${error.statusCode}';
    }
    return 'Ошибка сервера.';
  }
}

class _ConfidenceChip extends StatelessWidget {
  const _ConfidenceChip({required this.text});

  final String text;

  @override
  Widget build(BuildContext context) {
    return Chip(
      label: Text(text, style: const TextStyle(fontWeight: FontWeight.w600)),
      avatar: const Icon(Icons.verified_outlined, size: 18),
    );
  }
}
