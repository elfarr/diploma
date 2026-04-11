String? requiredField(String? s, {String message = 'Обязательное поле'}) {
  if (s == null || s.trim().isEmpty) {
    return message;
  }
  return null;
}

double? parseNumber(String input) {
  final normalized = input.replaceAll(' ', '').replaceAll(',', '.').trim();
  if (normalized.isEmpty) {
    return null;
  }
  return double.tryParse(normalized);
}

String? numberInRange(
  String? s, {
  required double min,
  required double max,
  required bool required,
}) {
  final value = s?.trim() ?? '';
  if (value.isEmpty) {
    if (required) {
      return 'Обязательное поле';
    }
    return null;
  }

  final parsed = parseNumber(value);
  if (parsed == null) {
    return 'Введите число';
  }

  if (parsed < min || parsed > max) {
    return 'Проверьте значение: показатель выглядит нетипичным.';
  }

  return null;
}
