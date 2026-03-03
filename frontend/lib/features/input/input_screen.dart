import 'package:flutter/material.dart';

import '../../models/predict_request.dart';
import '../../models/signature.dart';
import '../../shared/form_storage.dart';
import '../../shared/units.dart';
import '../../shared/validators.dart';
import 'presets.dart';

class InputScreen extends StatefulWidget {
  const InputScreen({super.key});

  @override
  State<InputScreen> createState() => _InputScreenState();
}

class _InputScreenState extends State<InputScreen> {

  final _formKey = GlobalKey<FormState>();
  final _storage = FormStorage();

  SignatureSpec? _signature;
  final Map<String, TextEditingController> _controllers = {};
  final Map<String, dynamic> _values = {};
  final Map<String, String> _unitByField = {};

  final List<_CategoricalGroup> _categoricalGroups = [];
  final Map<String, String?> _selectedByGroup = {};
  final Set<String> _categoricalFieldNames = {};

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

  String? _validate(SignatureField field, String? raw) {
    final reqErr = requiredField(raw);
    if (reqErr != null) {
      return reqErr;
    }

    if (field.min != null && field.max != null) {
      return numberInRange(
        raw,
        min: field.min!,
        max: field.max!,
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
      appBar: AppBar(title: const Text('Ввод данных')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  _SectionCard(
                    title: 'Пресеты',
                    child: Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: [
                        ChoiceChip(
                          label: const Text('Low'),
                          selected: _activePreset == 'low',
                          onSelected: (_) => _applyNamedPreset('low'),
                        ),
                        ChoiceChip(
                          label: const Text('High'),
                          selected: _activePreset == 'high',
                          onSelected: (_) => _applyNamedPreset('high'),
                        ),
                        ChoiceChip(
                          label: const Text('Undetermined'),
                          selected: _activePreset == 'undetermined',
                          onSelected: (_) => _applyNamedPreset('undetermined'),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 12),
                  Expanded(
                    child: Form(
                      key: _formKey,
                      child: ListView(
                        children: [
                          if (_categoricalGroups.isNotEmpty)
                            const Padding(
                              padding: EdgeInsets.only(bottom: 8),
                              child: Text(
                                'Категориальные поля',
                                style: TextStyle(fontWeight: FontWeight.w600),
                              ),
                            ),
                          ..._categoricalGroups.map(_buildCategoricalField),
                          if (numericFields.isNotEmpty)
                            const Padding(
                              padding: EdgeInsets.only(bottom: 8, top: 8),
                              child: Text(
                                'Числовые поля',
                                style: TextStyle(fontWeight: FontWeight.w600),
                              ),
                            ),
                          ...numericFields.map(_buildNumericField),
                        ],
                      ),
                    ),
                  ),
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
        decoration: InputDecoration(
          labelText: group.key,
          border: const OutlineInputBorder(),
          isDense: true,
        ),
        items: items,
        validator: (value) {
          if (value == null) {
            return 'Выберите значение';
          }
          return null;
        },
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

    return _SectionCard(
      child: TextFormField(
        controller: _controllers[field.name],
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        decoration: InputDecoration(
          labelText: field.name,
          border: const OutlineInputBorder(),
          isDense: true,
          helperText: (field.min != null && field.max != null)
              ? 'Диапазон: ${field.min}..${field.max}'
              : null,
          suffixIcon: (converter != null && selectedUnit != null)
              ? Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 8),
                  child: DropdownButtonHideUnderline(
                    child: DropdownButton<String>(
                      value: selectedUnit,
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
                )
              : null,
          suffixIconConstraints: const BoxConstraints(minWidth: 0, minHeight: 0),
        ),
        validator: (v) => _validate(field, v),
        onChanged: (v) => _values[field.name] = parseNumber(v),
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

String _optionLabel(String groupKey, String fullFieldName) {
  final prefix = '${groupKey}_';
  if (fullFieldName.startsWith(prefix)) {
    return fullFieldName.substring(prefix.length);
  }
  return fullFieldName;
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
