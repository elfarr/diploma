import 'dart:convert';

import 'package:flutter/services.dart' show rootBundle;

class SignatureField {
  const SignatureField({
    required this.name,
    required this.type,
    this.min,
    this.max,
  });

  final String name;
  final String type;
  final double? min;
  final double? max;

  factory SignatureField.fromJson(Map<String, dynamic> json) {
    return SignatureField(
      name: (json['name'] as String?) ?? '',
      type: (json['type'] as String?) ?? '',
      min: _asDouble(json['min']),
      max: _asDouble(json['max']),
    );
  }

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'name': name,
      'type': type,
      'min': min,
      'max': max,
    };
  }
}

class SignatureSpec {
  const SignatureSpec({required this.fields});

  final List<SignatureField> fields;

  factory SignatureSpec.fromJson(Map<String, dynamic> json) {
    final rawFields = _extractRawFields(json);
    final fields = rawFields is List
        ? rawFields
            .whereType<Map<String, dynamic>>()
            .map(SignatureField.fromJson)
            .where((f) => f.name.isNotEmpty)
            .toList()
        : <SignatureField>[];
    return SignatureSpec(fields: fields);
  }

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'fields': fields.map((f) => f.toJson()).toList(),
    };
  }

  static Future<SignatureSpec> loadFromAssets(String path) async {
    final raw = await rootBundle.loadString(path);
    final decoded = jsonDecode(raw);
    if (decoded is Map<String, dynamic>) {
      return SignatureSpec.fromJson(decoded);
    }
    return const SignatureSpec(fields: <SignatureField>[]);
  }
}

List<dynamic>? _extractRawFields(Map<String, dynamic> json) {
  final direct = json['fields'];
  if (direct is List) {
    return direct;
  }

  final input = json['input'];
  if (input is Map<String, dynamic>) {
    final nested = input['features'];
    if (nested is List) {
      return nested;
    }
  }
  return null;
}

double? _asDouble(dynamic value) {
  if (value == null) {
    return null;
  }
  if (value is num) {
    return value.toDouble();
  }
  if (value is String) {
    return double.tryParse(value);
  }
  return null;
}
