import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

class FormStorage {
  static const String _storageKey = 'predict_form_v1';

  Future<void> saveForm(Map<String, dynamic> values) async {
    final prefs = await SharedPreferences.getInstance();
    final encoded = jsonEncode(values);
    await prefs.setString(_storageKey, encoded);
  }

  Future<Map<String, dynamic>> loadForm() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_storageKey);
    if (raw == null || raw.isEmpty) {
      return <String, dynamic>{};
    }
    try {
      final decoded = jsonDecode(raw);
      if (decoded is Map<String, dynamic>) {
        return decoded;
      }
    } catch (_) {
     
    }
    return <String, dynamic>{};
  }
}
