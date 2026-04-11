import 'explain_item.dart';

class PredictResponse {
  const PredictResponse({
    this.klass,
    this.prob,
    this.probCal,
    this.badge,
    this.explain = const <ExplainItem>[],
  });

  final String? klass;
  final double? prob;
  final double? probCal;
  final String? badge;
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
      explain: explainList,
    );
  }

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'klass': klass,
      'prob': prob,
      'prob_cal': probCal,
      'badge': badge,
      'explain': explain.map((e) => e.toJson()).toList(),
    };
  }
}
