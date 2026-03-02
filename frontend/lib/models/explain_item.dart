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
    return ExplainItem(
      name: (json['name'] as String?) ?? '',
      value: (json['value'] as num?)?.toDouble() ?? 0.0,
      contribution: (json['contribution'] as num?)?.toDouble() ?? 0.0,
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
