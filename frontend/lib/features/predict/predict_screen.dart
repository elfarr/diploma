import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../../api/api_client.dart';
import '../../api/predictor_repo.dart';
import '../../models/predict_request.dart';
import '../../models/predict_response.dart';
import '../../models/explain_item.dart';

const _lowVector = <String, double>{
  'ОХ перед ТП': 3.506107981608616,
  'ЛПНП перед ТП': 0.7876984106765977,
  'ЛПВП перед ТП': 1.9034689818347448,
  'ТГ перед ТП': 0.8653533992477742,
  'Мочевая кислота перед ТП': 206.66397105096388,
  'ЭХО ЛП перед ТП': 31.24068726837028,
  'ЭХО КДР перед ТП': 39.61329009596596,
  'ЭХО МЖП перед ТП': 8.809431353401537,
  'ЭХО ЗС перед ТП': 7.699244230414687,
  'ЭХО СДЛА перед ТП': 13.489484782994449,
  'ЭХО ФВ перед ТП': 68.62670083381416,
  'ЭХО ММЛЖ перед ТП': 117.92126056175977,
  'ЭХО ИММЛЖ перед ТП': 96.448646711405,
  'ОТТ перед ТП': 0,
  'САД перед ТП': 117.82160281805571,
  'ДАД перед ТП': 55.60091162264049,
  'relative risk': 0.13025527701626988,
  'Пол_жен': 1,
  'Пол_муж': 0,
  'Диагноз_АРМС': 0,
  'Диагноз_ДБСТ': 0,
  'Диагноз_МКБ': 0,
  'Диагноз_Периодическая болезнь': 0,
  'Диагноз_Подагра': 0,
  'Диагноз_Поликлистоз': 0,
  'Диагноз_Сахарный диабет': 0,
  'Диагноз_ХГН': 0,
  'Диагноз_Хронический тубулоинтерстицильный нефрит': 0,
  'Диагноз_Хронический тубулоинтерстицильный нефрит без ГУС': 0,
  'Диагноз_прочие': 0,
  'Диагноз_с-м Альпорта': 0,
  'Донор_От живого донора неродственная': 0,
  'Донор_От живого донора родственная': 0,
  'Донор_трупная почка': 0,
  'ИБС до ТП_есть': 0,
  'ИБС до ТП_нет': 1,
  'ХНС до ТП_есть': 0,
  'ХНС до ТП_нет': 1,
  'ОНМК до ТП_есть': 0,
  'ОНМК до ТП_нет': 1,
  'ТЭЛА до ТП_есть': 0,
  'ТЭЛА до ТП_нет': 1,
  'ИМ до ТП_есть': 0,
  'ИМ до ТП_нет': 1,
  'стадия ХСН перед ТП_1 ФК': 0,
  'стадия ХСН перед ТП_2 ФК': 0,
  'стадия ХСН перед ТП_3 ФК': 0,
  'стадия ХСН перед ТП_нет': 1,
  'КАГ до ТП_есть': 0,
  'КАГ до ТП_нет': 1,
};

const _highVector = <String, double>{
  'ОХ перед ТП': 4.2,
  'ЛПНП перед ТП': 2.4,
  'ЛПВП перед ТП': 1.3,
  'ТГ перед ТП': 1.1,
  'Мочевая кислота перед ТП': 300,
  'ЭХО ЛП перед ТП': 35,
  'ЭХО КДР перед ТП': 50,
  'ЭХО МЖП перед ТП': 10,
  'ЭХО ЗС перед ТП': 10,
  'ЭХО СДЛА перед ТП': 20,
  'ЭХО ФВ перед ТП': 50,
  'ЭХО ММЛЖ перед ТП': 160,
  'ЭХО ИММЛЖ перед ТП': 120,
  'ОТТ перед ТП': 1,
  'САД перед ТП': 150,
  'ДАД перед ТП': 95,
  'relative risk': 2,
  'Пол_жен': 0,
  'Пол_муж': 1,
  'Диагноз_АРМС': 0,
  'Диагноз_ДБСТ': 0,
  'Диагноз_МКБ': 0,
  'Диагноз_Периодическая болезнь': 0,
  'Диагноз_Подагра': 0,
  'Диагноз_Поликлистоз': 0,
  'Диагноз_Сахарный диабет': 1,
  'Диагноз_ХГН': 1,
  'Диагноз_Хронический тубулоинтерстицильный нефрит': 0,
  'Диагноз_Хронический тубулоинтерстицильный нефрит без ГУС': 0,
  'Диагноз_прочие': 0,
  'Диагноз_с-м Альпорта': 0,
  'Донор_От живого донора неродственная': 0,
  'Донор_От живого донора родственная': 0,
  'Донор_трупная почка': 1,
  'ИБС до ТП_есть': 1,
  'ИБС до ТП_нет': 0,
  'ХНС до ТП_есть': 0,
  'ХНС до ТП_нет': 1,
  'ОНМК до ТП_есть': 0,
  'ОНМК до ТП_нет': 1,
  'ТЭЛА до ТП_есть': 0,
  'ТЭЛА до ТП_нет': 1,
  'ИМ до ТП_есть': 1,
  'ИМ до ТП_нет': 0,
  'стадия ХСН перед ТП_1 ФК': 0,
  'стадия ХСН перед ТП_2 ФК': 0,
  'стадия ХСН перед ТП_3 ФК': 1,
  'стадия ХСН перед ТП_нет': 0,
  'КАГ до ТП_есть': 1,
  'КАГ до ТП_нет': 0,
};

const _demoFields = [
  _Field('ОХ перед ТП', 'ОХ перед ТП', 1.5, 10, 'ммоль/л', 2),
  _Field('ЛПНП перед ТП', 'ЛПНП перед ТП', 0.5, 6, 'ммоль/л', 2),
  _Field('ТГ перед ТП', 'Триглицериды', 0.2, 5, 'ммоль/л', 2),
  _Field('САД перед ТП', 'Сист. АД', 80, 200, 'мм рт.ст.', 0),
  _Field('ДАД перед ТП', 'Диаст. АД', 40, 120, 'мм рт.ст.', 0),
  _Field('relative risk', 'relative risk', 0, 20, '', 2),
];

class PredictScreen extends StatefulWidget {
  const PredictScreen({super.key});

  @override
  State<PredictScreen> createState() => _PredictScreenState();
}

class _PredictScreenState extends State<PredictScreen> {
  final _baseUrls = const [
    'http://localhost:8081',
    'http://10.0.2.2:8081',
  ];

  static const _defaultToken = 'change-me';
  late String _baseUrl;
  String _modelVersion = 'v2.0.0';
  bool _unitConvert = false;

  final Map<String, TextEditingController> _controllers = {
    for (final f in _demoFields)
      f.name: TextEditingController(text: f.min.toStringAsFixed(f.decimals)),
  };

  Map<String, double> _activeVector = Map.of(_lowVector);
  PredictResponse? _last;
  String? _error;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _baseUrl = kIsWeb ? Uri.base.origin : 'http://10.0.2.2:8081';
  }

  @override
  void dispose() {
    for (final c in _controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  PredictorRepo _buildRepo() {
    return PredictorRepo(ApiClient());
  }

  Future<void> _predict() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    final repo = _buildRepo();
    try {
      final full = _composePayload();
      debugPrint('POST /predict first 6: ${full.entries.take(6).toList()} (total ${full.length})');
      final resp = await repo.predict(PredictRequest(features: full));
      setState(() => _last = resp);
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _loading = false);
    }
  }

  Map<String, double> _composePayload() {
    final result = Map<String, double>.from(_activeVector);
    for (final f in _demoFields) {
      final v = double.tryParse(_controllers[f.name]?.text ?? '');
      if (v != null) {
        result[f.name] = v;
      }
    }
    return result;
  }

  void _applyPresetLow() {
    _activeVector = Map.of(_lowVector);
    _controllers.forEach((k, c) {
      if (_lowVector.containsKey(k)) {
        final decimals = _demoFields.firstWhere((f) => f.name == k).decimals;
        c.text = _lowVector[k]!.toStringAsFixed(decimals);
      }
    });
    setState(() {});
  }

  void _applyPresetHigh() {
    _activeVector = Map.of(_highVector);
    _controllers.forEach((k, c) {
      if (_highVector.containsKey(k)) {
        final decimals = _demoFields.firstWhere((f) => f.name == k).decimals;
        c.text = _highVector[k]!.toStringAsFixed(decimals);
      }
    });
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final sbpField = _demoFields.firstWhere((f) => f.name == 'САД перед ТП');
    final sbpVal = double.tryParse(_controllers[sbpField.name]?.text ?? '') ?? sbpField.min;

    return Scaffold(
      appBar: AppBar(title: const Text('Прогноз')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Wrap(
                spacing: 8,
                children: _baseUrls
                    .map(
                      (url) => ChoiceChip(
                        label: Text(url.contains('10.0.2.2') ? 'Эмулятор' : 'Локал'),
                        selected: _baseUrl == url,
                        onSelected: (_) => setState(() => _baseUrl = url),
                      ),
                    )
                    .toList(),
              ),
              const SizedBox(height: 8),
              Text('Версия модели: $_modelVersion'),
              Row(
                children: [
                  const Text('Конвертация единиц'),
                  Switch(
                    value: _unitConvert,
                    onChanged: (v) => setState(() => _unitConvert = v),
                  ),
                  const Spacer(),
              TextButton(
                onPressed: _loading ? null : _applyPresetLow,
                child: const Text('Никзкий риск'),
              ),
              TextButton(
                onPressed: _loading ? null : _applyPresetHigh,
                child: const Text('Высокий риск'),
              ),
              TextButton(
                onPressed: _loading
                    ? null
                    : () {
                        _activeVector = Map.of(_highVector);
                        _predict();
                      },
                child: const Text('Отправить высокий риск'),
              ),
                ],
              ),
              const Divider(),
              ..._demoFields.map(_buildField),
              const SizedBox(height: 12),
              _buildWhatIfSlider(sbpField, sbpVal),
              const SizedBox(height: 12),
              ElevatedButton(
                onPressed: _loading ? null : _predict,
                child: _loading
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text('Прогноз'),
              ),
              const SizedBox(height: 12),
              if (_error != null)
                Text(
                  _error!,
                  style: const TextStyle(color: Colors.red),
                ),
              if (_last != null) _buildResult(_last!),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildField(_Field f) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: TextField(
        controller: _controllers[f.name],
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        decoration: InputDecoration(
          labelText: '${f.label} (${f.unit})',
          helperText: '${f.min} – ${f.max}',
        ),
      ),
    );
  }

  Widget _buildWhatIfSlider(_Field field, double value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Что если: ${field.label}'),
        Slider(
          value: value.clamp(field.min, field.max),
          min: field.min,
          max: field.max,
          divisions: ((field.max - field.min) ~/ 1).clamp(1, 200).toInt(),
          label: value.toStringAsFixed(field.decimals),
          onChanged: (v) {
            _controllers[field.name]?.text = v.toStringAsFixed(field.decimals);
            setState(() {});
          },
          onChangeEnd: (_) => _predict(),
        ),
      ],
    );
  }

  Widget _buildResult(PredictResponse resp) {
    List<ExplainItem> explain = resp.explain;
    if (explain.length > 5) explain = explain.sublist(0, 5);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Класс: ${resp.klass ?? '-'}', style: Theme.of(context).textTheme.titleMedium),
            Text('prob_cal: ${resp.probCal?.toStringAsFixed(4) ?? '-'}'),
            if (resp.prob != null) Text('prob: ${resp.prob!.toStringAsFixed(4)}'),
            if (resp.badge != null) Text('badge: ${resp.badge}'),
            if (explain.isNotEmpty) const SizedBox(height: 8),
            if (explain.isNotEmpty) const Text('Топ факторов:'),
            ...explain.map(
              (e) => Text('${e.name}: ${e.value.toStringAsFixed(4)} (${e.contribution.toStringAsFixed(4)})'),
            ),
          ],
        ),
      ),
    );
  }
}

class _Field {
  const _Field(this.name, this.label, this.min, this.max, this.unit, this.decimals);
  final String name;
  final String label;
  final double min;
  final double max;
  final String unit;
  final int decimals;
}
