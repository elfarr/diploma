class ExplainItem {
  const ExplainItem({
    required this.name,
    required this.value,
    required this.contribution,
  });

  final String name;
  final double value;
  final double contribution;

  factory ExplainItem.fromJson(Map<String, dynamic> json) {
    final rawContribution = json['contribution'];
    final rawImpact = json['impact'];
    final direction = (json['direction'] as String?) ?? '';
    double contribution = 0.0;
    if (rawContribution is num) {
      contribution = rawContribution.toDouble();
    } else if (rawImpact is num) {
      final impact = rawImpact.toDouble();
      contribution = direction == 'down' ? -impact : impact;
    }

    return ExplainItem(
      name: (json['name'] as String?) ?? (json['feature'] as String?) ?? '',
      value: (json['value'] as num?)?.toDouble() ?? 0.0,
      contribution: contribution,
    );
  }

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'name': name,
      'value': value,
      'contribution': contribution,
    };
  }
}
