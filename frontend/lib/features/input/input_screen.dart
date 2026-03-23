import 'package:flutter/material.dart';

import '../../models/predict_request.dart';
import '../../models/signature.dart';
import '../../shared/form_storage.dart';
import '../../shared/qrisk3_calculator.dart';
import '../../shared/units.dart';
import '../../shared/validators.dart';
import 'presets.dart';

class InputScreen extends StatefulWidget {
  const InputScreen({super.key});

  @override
  State<InputScreen> createState() => _InputScreenState();
}

class _InputScreenState extends State<InputScreen> {
  static const _relativeRiskFieldName = 'relative risk';

  final _formKey = GlobalKey<FormState>();
  final _storage = FormStorage();

  SignatureSpec? _signature;
  final Map<String, TextEditingController> _controllers = {};
  final Map<String, dynamic> _values = {};
  final Map<String, String> _unitByField = {};

  final List<_CategoricalGroup> _categoricalGroups = [];
  final Map<String, String?> _selectedByGroup = {};
  final Set<String> _categoricalFieldNames = {};
  final _qriskDraft = _QriskDraft();

  bool _loading = true;
  String? _activePreset;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    final signature = await SignatureSpec.loadFromAssets('assets/signature.json');
    final saved = await _storage.loadForm();

    for (final field in signature.fields) {
      final raw = saved[field.name];
      final text = raw == null ? '' : raw.toString();
      _controllers[field.name] = TextEditingController(text: text);
      _values[field.name] = raw;

      final converter = _converterFor(field.name);
      if (converter != null) {
        _unitByField[field.name] = converter.units.first;
      }
    }

    _categoricalGroups
      ..clear()
      ..addAll(_buildCategoricalGroups(signature.fields));

    _categoricalFieldNames
      ..clear()
      ..addAll(_categoricalGroups.expand((g) => g.fields.map((f) => f.name)));

    _syncGroupSelectionsFromControllers();

    if (!mounted) {
      return;
    }
    setState(() {
      _signature = signature;
      _loading = false;
    });
  }

  @override
  void dispose() {
    for (final c in _controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  List<_CategoricalGroup> _buildCategoricalGroups(List<SignatureField> fields) {
    final byPrefix = <String, List<SignatureField>>{};
    for (final f in fields) {
      if (!f.name.contains('_')) {
        continue;
      }
      final isBinary = f.min == 0.0 && f.max == 1.0;
      if (!isBinary && (f.min != null || f.max != null)) {
        continue;
      }
      final prefix = f.name.split('_').first.trim();
      if (prefix.isEmpty) {
        continue;
      }
      byPrefix.putIfAbsent(prefix, () => <SignatureField>[]).add(f);
    }

    final groups = <_CategoricalGroup>[];
    for (final entry in byPrefix.entries) {
      if (entry.value.length < 2) {
        continue;
      }
      groups.add(
        _CategoricalGroup(
          key: entry.key,
          fields: entry.value,
        ),
      );
    }

    groups.sort((a, b) => a.key.compareTo(b.key));
    return groups;
  }

  void _syncGroupSelectionsFromControllers() {
    for (final group in _categoricalGroups) {
      String? selected;
      for (final field in group.fields) {
        final v = parseNumber(_controllers[field.name]?.text ?? '');
        if (v != null && v >= 0.5) {
          selected = field.name;
          break;
        }
      }
      _selectedByGroup[group.key] = selected;
    }
  }

  void _applyPreset(Map<String, dynamic> preset) {
    for (final entry in preset.entries) {
      final c = _controllers[entry.key];
      if (c == null) {
        continue;
      }
      c.text = entry.value.toString();
      _values[entry.key] = entry.value;
    }
    _syncGroupSelectionsFromControllers();
    setState(() {});
  }

  void _applyNamedPreset(String name) {
    switch (name) {
      case 'low':
        _applyPreset(presetLow);
        break;
      case 'high':
        _applyPreset(presetHigh);
        break;
      default:
        _applyPreset(presetUndetermined);
    }
    setState(() {
      _activePreset = name;
    });
  }

  void _selectGroupOption(_CategoricalGroup group, String selectedFieldName) {
    _selectedByGroup[group.key] = selectedFieldName;

    for (final field in group.fields) {
      final c = _controllers[field.name];
      if (c == null) {
        continue;
      }

      double value = 0.0;
      if (field.name == selectedFieldName) {
        value = 1.0;
      }
      c.text = formatNum(value, digits: 0);
      _values[field.name] = value;
    }

    setState(() {});
  }

  void _changeUnit(SignatureField field, String targetUnit) {
    final converter = _converterFor(field.name);
    if (converter == null) {
      return;
    }
    final fromUnit = _unitByField[field.name];
    if (fromUnit == null || fromUnit == targetUnit) {
      return;
    }

    final controller = _controllers[field.name];
    if (controller == null) {
      return;
    }
    final parsed = parseNumber(controller.text);
    if (parsed != null) {
      final converted = converter.convert(parsed, fromUnit, targetUnit);
      controller.text = formatNum(converted, digits: 3);
      _values[field.name] = converted;
    }

    setState(() {
      _unitByField[field.name] = targetUnit;
    });
  }

  double? _currentNumber(String fieldName) {
    return parseNumber(_controllers[fieldName]?.text ?? '');
  }

  String _initialTextForField(String fieldName, {int digits = 3}) {
    final value = _currentNumber(fieldName);
    if (value == null) {
      return '';
    }
    return formatNum(value, digits: digits);
  }

  void _ensureQriskDraftDefaults() {
    if (!_qriskDraft.seeded) {
      _qriskDraft.sex = _inferQriskSex();
      _qriskDraft.seeded = true;
    }
    if (_qriskDraft.sbpStdDevText.isEmpty) {
      _qriskDraft.sbpStdDevText = '0';
    }
    if (_qriskDraft.townsendText.isEmpty) {
      _qriskDraft.townsendText = '0';
    }
  }

  Qrisk3Sex _inferQriskSex() {
    if ((_currentNumber('Пол_жен') ?? 0) >= 0.5) {
      return Qrisk3Sex.female;
    }
    if ((_currentNumber('Пол_муж') ?? 0) >= 0.5) {
      return Qrisk3Sex.male;
    }
    return Qrisk3Sex.female;
  }

  Future<void> _openRelativeRiskCalculator() async {
    _ensureQriskDraftDefaults();

    final ageController = TextEditingController(text: _qriskDraft.ageText);
    final totalCholController = TextEditingController(text: _qriskDraft.totalCholText);
    final hdlController = TextEditingController(text: _qriskDraft.hdlText);
    final sbpController = TextEditingController(text: _qriskDraft.sbpText);
    final sbpStdDevController = TextEditingController(text: _qriskDraft.sbpStdDevText);
    final heightController = TextEditingController(text: _qriskDraft.heightText);
    final weightController = TextEditingController(text: _qriskDraft.weightText);
    final townsendController = TextEditingController(text: _qriskDraft.townsendText);

    var sex = _qriskDraft.sex;
    var ethnicity = _qriskDraft.ethnicity;
    var smoking = _qriskDraft.smoking;
    var diabetes = _qriskDraft.diabetes;
    var familyHistoryCvd = _qriskDraft.familyHistoryCvd;
    var chronicKidneyDisease = _qriskDraft.chronicKidneyDisease;
    var atrialFibrillation = _qriskDraft.atrialFibrillation;
    var bpTreatment = _qriskDraft.bpTreatment;
    var migraine = _qriskDraft.migraine;
    var rheumatoidArthritis = _qriskDraft.rheumatoidArthritis;
    var sle = _qriskDraft.sle;
    var severeMentalIllness = _qriskDraft.severeMentalIllness;
    var atypicalAntipsychotics = _qriskDraft.atypicalAntipsychotics;
    var steroids = _qriskDraft.steroids;
    var erectileDysfunction = _qriskDraft.erectileDysfunction;
    Qrisk3Result? qriskResult;
    String? calcError;

    double? parse(TextEditingController controller) => parseNumber(controller.text);

    void invalidateComputedResult() {
      qriskResult = null;
      calcError = null;
      _qriskDraft.result = null;
    }

    final result = await showDialog<double>(
      context: context,
      builder: (dialogContext) {
        return StatefulBuilder(
          builder: (context, setLocalState) {
            void recalc() {
              final age = parse(ageController);
              final totalChol = parse(totalCholController);
              final hdl = parse(hdlController);
              final sbp = parse(sbpController);
              final sbpStdDev = parse(sbpStdDevController) ?? 0.0;
              final heightCm = parse(heightController);
              final weightKg = parse(weightController);
              final townsend = parse(townsendController) ?? 0.0;

              final missingRequired = <String>[
                if (age == null) 'возраст',
                if (totalChol == null) 'общий холестерин',
                if (hdl == null) 'ЛПВП',
                if (sbp == null) 'САД',
                if (heightCm == null) 'рост',
                if (weightKg == null) 'вес',
              ];

              String? validationMessage;
              if (hdl != null && hdl <= 0) {
                validationMessage = 'ЛПВП должен быть больше 0';
              } else if (missingRequired.isNotEmpty) {
                validationMessage = 'Заполните поля: ${missingRequired.join(', ')}';
              }

              if (validationMessage != null) {
                setLocalState(() {
                  calcError = validationMessage;
                  qriskResult = null;
                  _qriskDraft.result = null;
                });
                return;
              }

              try {
                final result = Qrisk3Calculator.calculate(
                  Qrisk3Input(
                    age: age!,
                    sex: sex,
                    ethnicity: ethnicity,
                    smoking: smoking,
                    diabetes: diabetes,
                    familyHistoryCvd: familyHistoryCvd,
                    chronicKidneyDisease: chronicKidneyDisease,
                    atrialFibrillation: atrialFibrillation,
                    bpTreatment: bpTreatment,
                    migraine: migraine,
                    rheumatoidArthritis: rheumatoidArthritis,
                    sle: sle,
                    severeMentalIllness: severeMentalIllness,
                    atypicalAntipsychotics: atypicalAntipsychotics,
                    steroids: steroids,
                    erectileDysfunction: erectileDysfunction,
                    cholesterolHdlRatio: totalChol! / hdl!,
                    systolicBp: sbp!,
                    sbpStdDev: sbpStdDev,
                    heightCm: heightCm!,
                    weightKg: weightKg!,
                    townsend: townsend,
                  ),
                );
                setLocalState(() {
                  qriskResult = result;
                  calcError = null;
                  _qriskDraft.result = result;
                });
              } on ArgumentError catch (error) {
                setLocalState(() {
                  qriskResult = null;
                  calcError = error.message?.toString() ?? error.toString();
                  _qriskDraft.result = null;
                });
              } catch (error) {
                setLocalState(() {
                  qriskResult = null;
                  calcError = error.toString();
                  _qriskDraft.result = null;
                });
              }
            }

            Widget buildNumberField({
              required TextEditingController controller,
              required String label,
              required String hint,
              String? helperText,
              String? tooltipMessage,
              required ValueChanged<String> onChangedValue,
            }) {
              final tooltip = tooltipMessage == null
                  ? null
                  : Padding(
                      padding: const EdgeInsets.only(right: 8),
                      child: Tooltip(
                        message: tooltipMessage,
                        waitDuration: const Duration(milliseconds: 250),
                        child: Icon(
                          Icons.info_outline_rounded,
                          size: 18,
                          color: Theme.of(context).colorScheme.primary,
                        ),
                      ),
                    );

              return TextField(
                controller: controller,
                keyboardType: const TextInputType.numberWithOptions(decimal: true),
                decoration: InputDecoration(
                  labelText: label,
                  hintText: hint,
                  helperText: helperText,
                  border: const OutlineInputBorder(),
                  isDense: true,
                  suffixIcon: tooltip,
                  suffixIconConstraints: const BoxConstraints(minWidth: 44, minHeight: 44),
                ),
                onChanged: (value) {
                  onChangedValue(value);
                  setLocalState(() {
                    invalidateComputedResult();
                  });
                },
              );
            }

            Widget buildDropdownField<T>({
              required T value,
              required String label,
              required List<T> values,
              required String Function(T value) titleOf,
              required ValueChanged<T?> onChanged,
            }) {
              return DropdownButtonFormField<T>(
                value: value,
                decoration: InputDecoration(
                  labelText: label,
                  border: const OutlineInputBorder(),
                  isDense: true,
                ),
                items: values
                    .map(
                      (item) => DropdownMenuItem<T>(
                        value: item,
                        child: Text(titleOf(item)),
                      ),
                    )
                    .toList(),
                onChanged: (next) {
                  onChanged(next);
                  setLocalState(() {
                    invalidateComputedResult();
                  });
                },
              );
            }

            Widget buildBoolField({
              required String label,
              required bool value,
              required ValueChanged<bool> onChanged,
            }) {
              return CheckboxListTile(
                value: value,
                dense: true,
                contentPadding: EdgeInsets.zero,
                controlAffinity: ListTileControlAffinity.leading,
                title: Text(label),
                onChanged: (next) {
                  onChanged(next ?? false);
                  setLocalState(() {
                    invalidateComputedResult();
                  });
                },
              );
            }

            Widget buildResultTile({
              required String label,
              required String value,
              required String unit,
              String? tooltipMessage,
            }) {
              return Container(
                width: 180,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surfaceContainerHighest.withValues(alpha: 0.35),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            label,
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ),
                        if (tooltipMessage != null)
                          Tooltip(
                            message: tooltipMessage,
                            waitDuration: const Duration(milliseconds: 250),
                            child: Icon(
                              Icons.info_outline_rounded,
                              size: 16,
                              color: Theme.of(context).colorScheme.primary,
                            ),
                          ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text.rich(
                      TextSpan(
                        children: [
                          TextSpan(
                            text: value,
                            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                  fontWeight: FontWeight.w700,
                                ),
                          ),
                          if (unit.isNotEmpty)
                            TextSpan(
                              text: ' $unit',
                              style: Theme.of(context).textTheme.bodyMedium,
                            ),
                        ],
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ),
              );
            }

            final ageValue = parse(ageController);
            final totalCholValue = parse(totalCholController);
            final hdlValue = parse(hdlController);
            final sbpValue = parse(sbpController);
            final heightValue = parse(heightController);
            final weightValue = parse(weightController);
            final canCalculate = ageValue != null &&
                ageValue >= 25 &&
                ageValue <= 84 &&
                totalCholValue != null &&
                totalCholValue > 0 &&
                hdlValue != null &&
                hdlValue > 0 &&
                sbpValue != null &&
                sbpValue > 0 &&
                heightValue != null &&
                heightValue > 0 &&
                weightValue != null &&
                weightValue > 0;

            return AlertDialog(
              title: const Text('Калькулятор QRISK3'),
              contentPadding: const EdgeInsets.fromLTRB(24, 20, 24, 20),
              content: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 720, maxHeight: 640),
                child: SingleChildScrollView(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Для QRISK3 указываются усредненные показатели, а не разовое измерение перед трансплантацией.',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                      const SizedBox(height: 12),
                      buildDropdownField<Qrisk3Sex>(
                        value: sex,
                        label: 'Пол',
                        values: Qrisk3Sex.values,
                        titleOf: _qriskSexLabel,
                        onChanged: (value) {
                          if (value != null) {
                            sex = value;
                            _qriskDraft.sex = value;
                            if (sex == Qrisk3Sex.female) {
                              erectileDysfunction = false;
                              _qriskDraft.erectileDysfunction = false;
                            }
                          }
                        },
                      ),
                      const SizedBox(height: 12),
                      buildDropdownField<Qrisk3Ethnicity>(
                        value: ethnicity,
                        label: 'Этническая группа',
                        values: Qrisk3Ethnicity.values,
                        titleOf: _qriskEthnicityLabel,
                        onChanged: (value) {
                          if (value != null) {
                            ethnicity = value;
                            _qriskDraft.ethnicity = value;
                          }
                        },
                      ),
                      const SizedBox(height: 12),
                      buildDropdownField<Qrisk3Smoking>(
                        value: smoking,
                        label: 'Курение',
                        values: Qrisk3Smoking.values,
                        titleOf: _qriskSmokingLabel,
                        onChanged: (value) {
                          if (value != null) {
                            smoking = value;
                            _qriskDraft.smoking = value;
                          }
                        },
                      ),
                      const SizedBox(height: 12),
                      buildDropdownField<Qrisk3Diabetes>(
                        value: diabetes,
                        label: 'Сахарный диабет',
                        values: Qrisk3Diabetes.values,
                        titleOf: _qriskDiabetesLabel,
                        onChanged: (value) {
                          if (value != null) {
                            diabetes = value;
                            _qriskDraft.diabetes = value;
                          }
                        },
                      ),
                      const SizedBox(height: 12),
                      buildNumberField(
                        controller: ageController,
                        label: 'Возраст',
                        hint: '25..84',
                        helperText: 'Допустимый диапазон: 25..84 года',
                        onChangedValue: (value) => _qriskDraft.ageText = value,
                      ),
                      const SizedBox(height: 12),
                      buildNumberField(
                        controller: totalCholController,
                        label: 'Средний ОХ',
                        hint: 'ммоль/л',
                        onChangedValue: (value) => _qriskDraft.totalCholText = value,
                      ),
                      const SizedBox(height: 12),
                      buildNumberField(
                        controller: hdlController,
                        label: 'Средний ЛПВП',
                        hint: 'ммоль/л',
                        onChangedValue: (value) => _qriskDraft.hdlText = value,
                      ),
                      const SizedBox(height: 12),
                      buildNumberField(
                        controller: sbpController,
                        label: 'Средний САД',
                        hint: 'мм рт. ст.',
                        onChangedValue: (value) => _qriskDraft.sbpText = value,
                      ),
                      const SizedBox(height: 12),
                      buildNumberField(
                        controller: sbpStdDevController,
                        label: 'Стандартное отклонение САД',
                        hint: '0',
                        tooltipMessage:
                            'Показывает, насколько менялось систолическое давление между измерениями. Если данных нет, оставьте 0.',
                        onChangedValue: (value) => _qriskDraft.sbpStdDevText = value,
                      ),
                      const SizedBox(height: 12),
                      buildNumberField(
                        controller: heightController,
                        label: 'Рост',
                        hint: 'см',
                        onChangedValue: (value) => _qriskDraft.heightText = value,
                      ),
                      const SizedBox(height: 12),
                      buildNumberField(
                        controller: weightController,
                        label: 'Вес',
                        hint: 'кг',
                        onChangedValue: (value) => _qriskDraft.weightText = value,
                      ),
                      const SizedBox(height: 12),
                      buildNumberField(
                        controller: townsendController,
                        label: 'Индекс Townsend',
                        hint: '0',
                        tooltipMessage:
                            'Показатель социально-экономической депривации, используемый в формуле QRISK3. Если неизвестен, оставьте 0.',
                        onChangedValue: (value) => _qriskDraft.townsendText = value,
                      ),
                      const SizedBox(height: 12),
                      buildBoolField(
                        label: 'Семейный анамнез сердечно-сосудистых заболеваний',
                        value: familyHistoryCvd,
                        onChanged: (value) {
                          familyHistoryCvd = value;
                          _qriskDraft.familyHistoryCvd = value;
                        },
                      ),
                      buildBoolField(
                        label: 'Хроническая болезнь почек',
                        value: chronicKidneyDisease,
                        onChanged: (value) {
                          chronicKidneyDisease = value;
                          _qriskDraft.chronicKidneyDisease = value;
                        },
                      ),
                      buildBoolField(
                        label: 'Фибрилляция предсердий',
                        value: atrialFibrillation,
                        onChanged: (value) {
                          atrialFibrillation = value;
                          _qriskDraft.atrialFibrillation = value;
                        },
                      ),
                      buildBoolField(
                        label: 'Антигипертензивная терапия',
                        value: bpTreatment,
                        onChanged: (value) {
                          bpTreatment = value;
                          _qriskDraft.bpTreatment = value;
                        },
                      ),
                      buildBoolField(
                        label: 'Мигрень',
                        value: migraine,
                        onChanged: (value) {
                          migraine = value;
                          _qriskDraft.migraine = value;
                        },
                      ),
                      buildBoolField(
                        label: 'Ревматоидный артрит',
                        value: rheumatoidArthritis,
                        onChanged: (value) {
                          rheumatoidArthritis = value;
                          _qriskDraft.rheumatoidArthritis = value;
                        },
                      ),
                      buildBoolField(
                        label: 'Системная красная волчанка',
                        value: sle,
                        onChanged: (value) {
                          sle = value;
                          _qriskDraft.sle = value;
                        },
                      ),
                      buildBoolField(
                        label: 'Тяжелое психическое заболевание',
                        value: severeMentalIllness,
                        onChanged: (value) {
                          severeMentalIllness = value;
                          _qriskDraft.severeMentalIllness = value;
                        },
                      ),
                      buildBoolField(
                        label: 'Прием атипичных антипсихотиков',
                        value: atypicalAntipsychotics,
                        onChanged: (value) {
                          atypicalAntipsychotics = value;
                          _qriskDraft.atypicalAntipsychotics = value;
                        },
                      ),
                      buildBoolField(
                        label: 'Прием глюкокортикостероидов',
                        value: steroids,
                        onChanged: (value) {
                          steroids = value;
                          _qriskDraft.steroids = value;
                        },
                      ),
                      if (sex == Qrisk3Sex.male)
                        buildBoolField(
                          label: 'Эректильная дисфункция',
                          value: erectileDysfunction,
                          onChanged: (value) {
                            erectileDysfunction = value;
                            _qriskDraft.erectileDysfunction = value;
                          },
                        ),
                      const SizedBox(height: 12),
                      if (calcError != null)
                        Text(
                          calcError!,
                          style: TextStyle(color: Theme.of(context).colorScheme.error),
                        )
                      else if (qriskResult != null)
                        Wrap(
                          spacing: 12,
                          runSpacing: 12,
                          children: [
                            buildResultTile(
                              label: 'QRISK3',
                              value: formatNum(qriskResult!.qrisk3, digits: 2),
                              unit: '%',
                              tooltipMessage:
                                  'Оценка 10-летнего сердечно-сосудистого риска по шкале QRISK3.',
                            ),
                            buildResultTile(
                              label: 'healthy person risk',
                              value: formatNum(qriskResult!.healthyPersonRisk, digits: 2),
                              unit: '%',
                              tooltipMessage:
                                  'Риск для условно здорового человека того же возраста и пола.',
                            ),
                            buildResultTile(
                              label: 'relative risk',
                              value: formatNum(qriskResult!.relativeRisk, digits: 3),
                              unit: '',
                              tooltipMessage:
                                  'Отношение индивидуального риска к риску условно здорового человека.',
                            ),
                            buildResultTile(
                              label: 'QRISK age',
                              value: formatNum(qriskResult!.qriskAge, digits: 1),
                              unit: 'лет',
                              tooltipMessage:
                                  'Возраст условно здорового человека с сопоставимым уровнем риска.',
                            ),
                          ],
                        ),
                      const SizedBox(height: 16),
                      Align(
                        alignment: Alignment.centerRight,
                        child: Wrap(
                          alignment: WrapAlignment.end,
                          crossAxisAlignment: WrapCrossAlignment.center,
                          spacing: 8,
                          runSpacing: 8,
                          children: [
                            TextButton(
                              onPressed: () => Navigator.of(dialogContext).pop(),
                              child: const Text('Отмена'),
                            ),
                            OutlinedButton(
                              onPressed: canCalculate ? recalc : null,
                              child: const Text('Рассчитать'),
                            ),
                            if (qriskResult != null)
                              FilledButton(
                                onPressed: () => Navigator.of(dialogContext).pop(qriskResult!.relativeRisk),
                                child: const Text('Вставить relative risk'),
                              ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            );
          },
        );
      },
    );

    ageController.dispose();
    totalCholController.dispose();
    hdlController.dispose();
    sbpController.dispose();
    sbpStdDevController.dispose();
    heightController.dispose();
    weightController.dispose();
    townsendController.dispose();

    if (result == null) {
      return;
    }

    final controller = _controllers[_relativeRiskFieldName];
    if (controller == null) {
      return;
    }

    controller.text = formatNum(result, digits: 3);
    _values[_relativeRiskFieldName] = result;
    if (!mounted) {
      return;
    }
    setState(() {});
  }
  String? _validate(SignatureField field, String? raw) {
    final reqErr = requiredField(raw);
    if (reqErr != null) {
      return reqErr;
    }

    final range = _displayRange(field);
    if (range != null) {
      return numberInRange(
        raw,
        min: range.min,
        max: range.max,
        required: true,
      );
    }

    if (raw == null || raw.trim().isEmpty) {
      return 'Обязательное поле';
    }
    if (parseNumber(raw) == null) {
      return 'Введите число';
    }
    return null;
  }

  _FieldRange? _displayRange(SignatureField field) {
    if (field.name == _relativeRiskFieldName) {
      return null;
    }
    if (field.min == null || field.max == null) {
      return null;
    }

    var min = field.min!;
    var max = field.max!;
    final converter = _converterFor(field.name);
    final selectedUnit = _unitByField[field.name];
    if (converter != null && selectedUnit != null) {
      final baseUnit = converter.units.first;
      min = converter.convert(min, baseUnit, selectedUnit);
      max = converter.convert(max, baseUnit, selectedUnit);
    }

    if (min > max) {
      final t = min;
      min = max;
      max = t;
    }
    return _FieldRange(min: min, max: max);
  }

  Future<void> _submit() async {
    if (!(_formKey.currentState?.validate() ?? false)) {
      return;
    }

    final fields = _signature?.fields ?? const <SignatureField>[];
    final features = <String, dynamic>{};
    for (final field in fields) {
      final raw = _controllers[field.name]?.text ?? '';
      final parsed = parseNumber(raw);
      if (parsed != null) {
        features[field.name] = parsed;
      } else if (_categoricalFieldNames.contains(field.name)) {
        // For one-hot categorical features, an empty value means "not selected".
        // Backend still expects the feature key, so send explicit zero.
        features[field.name] = 0.0;
      }
    }

    await _storage.saveForm(features);
    if (!mounted) {
      return;
    }
    Navigator.pop(context, PredictRequest(features: features));
  }

  @override
  Widget build(BuildContext context) {
    final fields = _signature?.fields ?? const <SignatureField>[];
    final numericFields = fields.where((f) => !_categoricalFieldNames.contains(f.name)).toList();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Ввод данных'),
        actions: [
          PopupMenuButton<String>(
            tooltip: 'Дополнительно',
            icon: const Icon(Icons.more_vert_rounded),
            onSelected: _applyNamedPreset,
            itemBuilder: (context) => const [
              PopupMenuItem<String>(
                value: 'low',
                child: Text('Пресет: низкий риск'),
              ),
              PopupMenuItem<String>(
                value: 'high',
                child: Text('Пресет: высокий риск'),
              ),
            ],
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : Padding(
              padding: const EdgeInsets.fromLTRB(16, 6, 16, 16),
              child: Form(
                key: _formKey,
                child: ListView(
                  padding: EdgeInsets.zero,
                  children: [
                    ..._categoricalGroups.map(_buildCategoricalField),
                    ...numericFields.map(_buildNumericField),
                    const SizedBox(height: 8),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        onPressed: _submit,
                        child: const Text('Рассчитать'),
                      ),
                    ),
                  ],
                ),
              ),
            ),
    );
  }

  Widget _buildCategoricalField(_CategoricalGroup group) {
    final items = <DropdownMenuItem<String>>[
      ...group.fields.map(
        (f) => DropdownMenuItem<String>(
          value: f.name,
          child: Text(_optionLabel(group.key, f.name)),
        ),
      ),
    ];

    final selected = _selectedByGroup[group.key];
    final dropdownValue = selected;

    return _SectionCard(
      child: DropdownButtonFormField<String>(
        value: dropdownValue,
        borderRadius: BorderRadius.circular(14),
        dropdownColor: Theme.of(context).colorScheme.surface,
        decoration: InputDecoration(
          labelText: group.key,
          filled: true,
          fillColor: Theme.of(context).colorScheme.surface,
          contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: BorderSide(color: Theme.of(context).colorScheme.outlineVariant),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: BorderSide(color: Theme.of(context).colorScheme.outlineVariant),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: BorderSide(color: Theme.of(context).colorScheme.primary, width: 1.5),
          ),
        ),
        items: items,
        validator: (_) => null,
        onChanged: (value) {
          if (value != null) {
            _selectGroupOption(group, value);
          }
        },
      ),
    );
  }

  Widget _buildNumericField(SignatureField field) {
    final converter = _converterFor(field.name);
    final selectedUnit = _unitByField[field.name];
    final isRelativeRiskField = field.name == _relativeRiskFieldName;
    final displayRange = _displayRange(field);
    final helperText = displayRange == null
        ? null
        : 'Диапазон: ${formatNum(displayRange.min, digits: 3)}..${formatNum(displayRange.max, digits: 3)}';

    return _SectionCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          TextFormField(
            controller: _controllers[field.name],
            keyboardType: const TextInputType.numberWithOptions(decimal: true),
            decoration: InputDecoration(
              labelText: field.name,
              border: const OutlineInputBorder(),
              isDense: true,
              helperText: helperText,
              suffixIcon: (converter != null && selectedUnit != null)
                  ? Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 8),
                      child: DecoratedBox(
                        decoration: BoxDecoration(
                          color: Theme.of(context).colorScheme.surface,
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
                        ),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 8),
                          child: DropdownButtonHideUnderline(
                            child: DropdownButton<String>(
                              value: selectedUnit,
                              borderRadius: BorderRadius.circular(12),
                              isDense: true,
                              items: converter.units
                                  .map(
                                    (u) => DropdownMenuItem<String>(
                                      value: u,
                                      child: Text(u),
                                    ),
                                  )
                                  .toList(),
                              onChanged: (u) {
                                if (u != null) {
                                  _changeUnit(field, u);
                                }
                              },
                            ),
                          ),
                        ),
                      ),
                    )
                  : null,
              suffixIconConstraints: const BoxConstraints(minWidth: 0, minHeight: 0),
            ),
            validator: (v) => _validate(field, v),
            onChanged: (v) => _values[field.name] = parseNumber(v),
          ),
          if (isRelativeRiskField)
            Align(
              alignment: Alignment.centerRight,
              child: TextButton.icon(
                onPressed: _openRelativeRiskCalculator,
                icon: const Icon(Icons.calculate_outlined),
                label: const Text('Калькулятор QRISK3'),
              ),
            ),
        ],
      ),
    );
  }
}

class _SectionCard extends StatelessWidget {
  const _SectionCard({
    this.title,
    required this.child,
  });

  final String? title;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.35),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: theme.colorScheme.outlineVariant),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (title != null) ...[
            Text(
              title!,
              style: theme.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 8),
          ],
          child,
        ],
      ),
    );
  }
}

class _CategoricalGroup {
  const _CategoricalGroup({
    required this.key,
    required this.fields,
  });

  final String key;
  final List<SignatureField> fields;
}

class _FieldRange {
  const _FieldRange({
    required this.min,
    required this.max,
  });

  final double min;
  final double max;
}

class _QriskDraft {
  bool seeded = false;

  String ageText = '';
  String totalCholText = '';
  String hdlText = '';
  String sbpText = '';
  String sbpStdDevText = '';
  String heightText = '';
  String weightText = '';
  String townsendText = '';

  Qrisk3Sex sex = Qrisk3Sex.female;
  Qrisk3Ethnicity ethnicity = Qrisk3Ethnicity.whiteOrNotStated;
  Qrisk3Smoking smoking = Qrisk3Smoking.nonSmoker;
  Qrisk3Diabetes diabetes = Qrisk3Diabetes.none;

  bool familyHistoryCvd = false;
  bool chronicKidneyDisease = false;
  bool atrialFibrillation = false;
  bool bpTreatment = false;
  bool migraine = false;
  bool rheumatoidArthritis = false;
  bool sle = false;
  bool severeMentalIllness = false;
  bool atypicalAntipsychotics = false;
  bool steroids = false;
  bool erectileDysfunction = false;

  Qrisk3Result? result;
}

String _optionLabel(String groupKey, String fullFieldName) {
  final prefix = '${groupKey}_';
  if (fullFieldName.startsWith(prefix)) {
    return fullFieldName.substring(prefix.length);
  }
  return fullFieldName;
}

String _qriskSexLabel(Qrisk3Sex sex) {
  switch (sex) {
    case Qrisk3Sex.female:
      return 'Женский';
    case Qrisk3Sex.male:
      return 'Мужской';
  }
}

String _qriskSmokingLabel(Qrisk3Smoking smoking) {
  switch (smoking) {
    case Qrisk3Smoking.nonSmoker:
      return 'Не курит';
    case Qrisk3Smoking.exSmoker:
      return 'Бывший курильщик';
    case Qrisk3Smoking.lightSmoker:
      return 'Курит < 10 сиг./день';
    case Qrisk3Smoking.moderateSmoker:
      return 'Курит 10-19 сиг./день';
    case Qrisk3Smoking.heavySmoker:
      return 'Курит >= 20 сиг./день';
  }
}

String _qriskDiabetesLabel(Qrisk3Diabetes diabetes) {
  switch (diabetes) {
    case Qrisk3Diabetes.none:
      return 'Нет';
    case Qrisk3Diabetes.type1:
      return '1 тип';
    case Qrisk3Diabetes.type2:
      return '2 тип';
  }
}

String _qriskEthnicityLabel(Qrisk3Ethnicity ethnicity) {
  switch (ethnicity) {
    case Qrisk3Ethnicity.whiteOrNotStated:
      return 'Белая / не указана';
    case Qrisk3Ethnicity.indian:
      return 'Индийская';
    case Qrisk3Ethnicity.pakistani:
      return 'Пакистанская';
    case Qrisk3Ethnicity.bangladeshi:
      return 'Бангладешская';
    case Qrisk3Ethnicity.otherAsian:
      return 'Другая азиатская';
    case Qrisk3Ethnicity.blackCaribbean:
      return 'Чернокожая карибская';
    case Qrisk3Ethnicity.blackAfrican:
      return 'Чернокожая африканская';
    case Qrisk3Ethnicity.chinese:
      return 'Китайская';
    case Qrisk3Ethnicity.otherEthnicGroup:
      return 'Другая';
  }
}

class _UnitConverter {
  const _UnitConverter({
    required this.units,
    required this.convert,
  });

  final List<String> units;
  final double Function(double value, String from, String to) convert;
}

_UnitConverter? _converterFor(String fieldName) {
  final key = _normFieldName(fieldName);

  // Glucose: mmol/L <-> mg/dL
  if (key.contains('glucose') || key.contains('глюк')) {
    return _UnitConverter(
      units: const ['mmol/L', 'mg/dL'],
      convert: (value, from, to) {
        if (from == to) return value;
        if (from == 'mmol/L' && to == 'mg/dL') {
          return mmolLToMgDl(value, factor: glucoseFactor);
        }
        return mgDlToMmolL(value, factor: glucoseFactor);
      },
    );
  }

  if (key.contains('ох') ||
      key.contains('chol') ||
      key.contains('totalchol') ||
      key.contains('лпнп') ||
      key.contains('ldl') ||
      key.contains('лпвп') ||
      key.contains('hdl')) {
    return _UnitConverter(
      units: const ['mmol/L', 'mg/dL'],
      convert: (value, from, to) {
        if (from == to) return value;
        if (from == 'mmol/L' && to == 'mg/dL') {
          return mmolLToMgDl(value, factor: cholesterolFactor);
        }
        return mgDlToMmolL(value, factor: cholesterolFactor);
      },
    );
  }

  if (key.contains('тг') || key.contains('triglycer') || key == 'tg') {
    return _UnitConverter(
      units: const ['mmol/L', 'mg/dL'],
      convert: (value, from, to) {
        if (from == to) return value;
        if (from == 'mmol/L' && to == 'mg/dL') {
          return mmolLToMgDl(value, factor: triglycerideFactor);
        }
        return mgDlToMmolL(value, factor: triglycerideFactor);
      },
    );
  }

  if (key.contains('мочева') || key.contains('uric')) {
    return _UnitConverter(
      units: const ['µmol/L', 'mg/dL'],
      convert: (value, from, to) {
        if (from == to) return value;
        if (from == 'µmol/L' && to == 'mg/dL') {
          return umolLToMgDl(value, factor: uricAcidFactor);
        }
        return mgDlToUmolL(value, factor: uricAcidFactor);
      },
    );
  }

  if (key.contains('sbp') || key.contains('сад') || key.contains('dbp') || key.contains('дад')) {
    return _UnitConverter(
      units: const ['mmHg', 'kPa'],
      convert: (value, from, to) {
        if (from == to) return value;
        if (from == 'mmHg' && to == 'kPa') {
          return mmhgToKpa(value);
        }
        return kpaToMmhg(value);
      },
    );
  }

  return null;
}

String _normFieldName(String value) {
  return value.toLowerCase().replaceAll(' ', '');
}

