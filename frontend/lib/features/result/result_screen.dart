import 'dart:async';

import 'package:flutter/material.dart';

import '../../api/api_client.dart';
import '../../models/explain_item.dart';
import '../../models/predict_request.dart';
import '../../models/predict_response.dart';
import '../../shared/risk_ui.dart';

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
            return Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Text(
                    message,
                    style: const TextStyle(fontSize: 16),
                  ),
                  const SizedBox(height: 16),
                  ElevatedButton(
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
    final isUncertain = pCal != null && isUndetermined(pCal);

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            data.klass ?? '-',
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: 8),
          Text(
            pCal != null ? probText(pCal) : 'Вероятность (калибр.): -',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 6),
          Text(
            pCal != null ? confidenceBadge(pCal) : 'Уверенность: нет данных',
          ),
          if (isUncertain) ...[
            const SizedBox(height: 12),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.orange),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(undeterminedText()),
            ),
          ],
          const SizedBox(height: 16),
          Text(
            'Топ факторы',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 8),
          if (top.isEmpty)
            const Text('Нет объяснений')
          else
            ...top.map(_factorTile),
          const Spacer(),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('\u041d\u0430\u0437\u0430\u0434 \u043a \u0432\u0432\u043e\u0434\u0443'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _factorTile(ExplainItem item) {
    final arrow = item.contribution >= 0 ? '↑' : '↓';
    final sign = item.contribution >= 0 ? '+' : '';
    final contr = '$sign${item.contribution.toStringAsFixed(2)}';
    final val = item.value.toStringAsFixed(2);

    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Text('$arrow ${item.name} — вклад: $contr, значение: $val'),
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
