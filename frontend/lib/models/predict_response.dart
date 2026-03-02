import 'explain_item.dart';

class PredictResponse {
  const PredictResponse({
    this.klass,
    this.prob,
    this.probCal,
    this.badge,
    this.undetermined,
    this.explain = const <ExplainItem>[],
  });

  final String? klass;
  final double? prob;
  final double? probCal;
  final String? badge;
  final bool? undetermined;
  final List<ExplainItem> explain;

  factory PredictResponse.fromJson(Map<String, dynamic> json) {
    final rawExplain = json['explain'];
    final explainList = rawExplain is List
        ? rawExplain
            .whereType<Map<String, dynamic>>()
            .map(ExplainItem.fromJson)
            .toList()
        : <ExplainItem>[];

    return PredictResponse(
      klass: (json['klass'] as String?) ?? (json['class'] as String?),
      prob: (json['prob'] as num?)?.toDouble(),
      probCal: ((json['prob_cal'] ?? json['p_cal']) as num?)?.toDouble(),
      badge: json['badge'] as String?,
      undetermined: json['undetermined'] as bool?,
      explain: explainList,
    );
  }

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'klass': klass,
      'prob': prob,
      'prob_cal': probCal,
      'badge': badge,
      'undetermined': undetermined,
      'explain': explain.map((e) => e.toJson()).toList(),
    };
  }
}
