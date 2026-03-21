String confidenceBadge(double pCal) {
  final d = (pCal - 0.5).abs();
  if (d >= 0.30) {
    return 'Высокая уверенность';
  }
  if (d >= 0.15) {
    return 'Средняя уверенность';
  }
  return 'Низкая уверенность';
}

bool isUndetermined(double pCal, {double tLow = 0.55, double tHigh = 0.85}) {
  return pCal >= tLow && pCal <= tHigh;
}

String undeterminedText() {
  return 'Результат находится в промежуточной зоне. Нужна дополнительная оценка.';
}

String probText(double pCal) {
  final percent = (pCal * 100).toStringAsFixed(0);
  return 'Вероятность: $percent%';
}
