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

String probText(double pCal) {
  final percent = (pCal * 100).toStringAsFixed(0);
  return 'Вероятность: $percent%';
}
