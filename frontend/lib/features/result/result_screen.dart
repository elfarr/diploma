import 'dart:async';

import 'package:flutter/material.dart';

import '../../api/api_client.dart';
import '../../models/explain_item.dart';
import '../../models/predict_request.dart';
import '../../models/predict_response.dart';
import '../../shared/report_download.dart';
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
                    child: const Text('На главную'),
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
                      _klassRu(data.klass) ?? '-',
                      style: Theme.of(context).textTheme.headlineMedium
                          ?.copyWith(fontWeight: FontWeight.w800),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      pCal != null ? probText(pCal) : 'Вероятность: -',
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
              const AppSectionTitle('Обратите внимание на факторы'),
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
              FilledButton.icon(
                onPressed: () => _downloadReport(context, data, top),
                icon: const Icon(Icons.download_rounded),
                label: const Text('Скачать отчет'),
              ),
              const SizedBox(height: 8),
              OutlinedButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text('На главную'),
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

  Future<void> _downloadReport(
    BuildContext context,
    PredictResponse data,
    List<ExplainItem> top,
  ) async {
    final report = _buildReport(data, top);
    final timestamp = _fileTimestamp(DateTime.now());
    final filename = 'kidney-risk-report-$timestamp.txt';

    try {
      await downloadTextReport(filename: filename, content: report);
      if (!mounted) {
        return;
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Файл \"$filename\" подготовлен к скачиванию')),
      );
    } catch (error) {
      if (!mounted) {
        return;
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Не удалось скачать файл: $error')),
      );
    }
  }

  String _buildReport(PredictResponse data, List<ExplainItem> top) {
    final buffer = StringBuffer();
    final now = DateTime.now();

    buffer.writeln('Отчет по оценке риска осложнений после трансплантации почки');
    buffer.writeln('Дата: ${_humanTimestamp(now)}');
    buffer.writeln();
    buffer.writeln('Результат');
    buffer.writeln('Класс риска: ${_klassRu(data.klass) ?? '-'}');
    buffer.writeln(
      'Вероятность: ${data.probCal != null ? '${(data.probCal! * 100).toStringAsFixed(0)}%' : '-'}',
    );
    if (data.badge != null && data.badge!.trim().isNotEmpty) {
      buffer.writeln('Статус: ${data.badge}');
    }

    buffer.writeln();
    buffer.writeln('Факторы, на которые стоит обратить внимание');
    if (top.isEmpty) {
      buffer.writeln('Нет данных');
    } else {
      for (final item in top) {
        final direction = item.contribution >= 0 ? 'повышает риск' : 'снижает риск';
        buffer.writeln(
          '- ${item.name}: ${_formatNumber(item.value)} ($direction)',
        );
      }
    }

    buffer.writeln();
    buffer.writeln('Введенные данные');
    for (final line in _reportInputLines()) {
      buffer.writeln('- $line');
    }

    return buffer.toString();
  }

  List<String> _reportInputLines() {
    final entries = widget.request.features.entries.toList();
    final groupCounts = <String, int>{};

    for (final entry in entries) {
      final prefix = _categoricalPrefix(entry.key);
      if (prefix == null) {
        continue;
      }
      if (!_isBinaryLike(entry.value)) {
        continue;
      }
      groupCounts[prefix] = (groupCounts[prefix] ?? 0) + 1;
    }

    final lines = <String>[];
    final seenGroups = <String>{};
    for (final entry in entries) {
      final prefix = _categoricalPrefix(entry.key);
      final isGroupedCategory = prefix != null &&
          (groupCounts[prefix] ?? 0) >= 2 &&
          _isBinaryLike(entry.value);

      if (isGroupedCategory) {
        if (seenGroups.contains(prefix)) {
          continue;
        }
        seenGroups.add(prefix);
        final selected = entries.firstWhere(
          (candidate) =>
              candidate.key.startsWith('${prefix}_') &&
              (_asDouble(candidate.value) ?? 0) >= 0.5,
          orElse: () => MapEntry<String, dynamic>('', null),
        );
        final selectedLabel = selected.key.isEmpty
            ? 'не указано'
            : selected.key.substring(prefix.length + 1);
        lines.add('$prefix: $selectedLabel');
        continue;
      }

      lines.add('${entry.key}: ${_formatValue(entry.value)}');
    }

    return lines;
  }

  String? _categoricalPrefix(String key) {
    final index = key.indexOf('_');
    if (index <= 0) {
      return null;
    }
    return key.substring(0, index);
  }

  bool _isBinaryLike(dynamic value) {
    final numeric = _asDouble(value);
    return numeric == 0.0 || numeric == 1.0;
  }

  double? _asDouble(dynamic value) {
    if (value is num) {
      return value.toDouble();
    }
    return null;
  }

  String _formatValue(dynamic value) {
    if (value is num) {
      return _formatNumber(value.toDouble());
    }
    return value?.toString() ?? '-';
  }

  String _formatNumber(double value) {
    final rounded = value.toStringAsFixed(3);
    return rounded
        .replaceFirst(RegExp(r'0+$'), '')
        .replaceFirst(RegExp(r'\.$'), '');
  }

  String _fileTimestamp(DateTime value) {
    final y = value.year.toString().padLeft(4, '0');
    final m = value.month.toString().padLeft(2, '0');
    final d = value.day.toString().padLeft(2, '0');
    final hh = value.hour.toString().padLeft(2, '0');
    final mm = value.minute.toString().padLeft(2, '0');
    final ss = value.second.toString().padLeft(2, '0');
    return '$y$m$d-$hh$mm$ss';
  }

  String _humanTimestamp(DateTime value) {
    final y = value.year.toString().padLeft(4, '0');
    final m = value.month.toString().padLeft(2, '0');
    final d = value.day.toString().padLeft(2, '0');
    final hh = value.hour.toString().padLeft(2, '0');
    final mm = value.minute.toString().padLeft(2, '0');
    final ss = value.second.toString().padLeft(2, '0');
    return '$d.$m.$y $hh:$mm:$ss';
  }

  String _errorText(Object? error) {
    if (error is TimeoutException) {
      return 'Превышено время ожидания. Проверь интернет.';
    }
    if (error is ApiException) {
      if (error.statusCode == 401) {
        return 'Нет доступа (токен).';
      }
      if (error.statusCode == 400 || error.statusCode == 422) {
        final details = error.message.trim();
        if (details.isNotEmpty) {
          return details;
        }
        return 'Проверьте введённые значения: некоторые показатели выглядят нетипично.';
      }
      return 'Ошибка сервера: ${error.statusCode}';
    }
    return 'Ошибка сервера.';
  }

  String? _klassRu(String? klass) {
    switch (klass) {
      case 'low':
        return 'Низкий риск';
      case 'high':
        return 'Высокий риск';
      default:
        return klass;
    }
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
