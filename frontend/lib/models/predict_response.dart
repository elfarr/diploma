import 'package:json_annotation/json_annotation.dart';

import 'explain_item.dart';

part 'predict_response.g.dart';

@JsonSerializable(explicitToJson: true)
class PredictResponse {
  const PredictResponse({
    required this.classLabel,
    required this.probCal,
    required this.confidence,
    required this.thresholds,
    this.explain,
    this.ood,
    this.modelVersion,
    this.schemaVersion,
    this.timingMs,
    this.pRaw,
  });

  @JsonKey(name: 'class')
  final String classLabel;

  @JsonKey(name: 'p_cal')
  final double probCal;

  final double confidence;

  final Map<String, dynamic> thresholds;

  final List<ExplainItem>? explain;

  final dynamic ood;

  @JsonKey(name: 'model_version')
  final String? modelVersion;

  @JsonKey(name: 'schema_version')
  final String? schemaVersion;

  @JsonKey(name: 'timing_ms')
  final int? timingMs;

  @JsonKey(name: 'p_raw')
  final double? pRaw;

  factory PredictResponse.fromJson(Map<String, dynamic> json) =>
      _$PredictResponseFromJson(json);

  Map<String, dynamic> toJson() => _$PredictResponseToJson(this);
}
