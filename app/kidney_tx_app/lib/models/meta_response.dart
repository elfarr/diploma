import 'package:json_annotation/json_annotation.dart';

part 'meta_response.g.dart';

@JsonSerializable(explicitToJson: true)
class MetaResponse {
  const MetaResponse({
    required this.modelVersion,
    required this.schemaVersion,
    required this.thresholds,
    required this.features,
    this.ranges,
  });

  @JsonKey(name: 'model_version')
  final String modelVersion;

  @JsonKey(name: 'schema_version')
  final String schemaVersion;

  final Map<String, dynamic> thresholds;

  final List<String> features;

  /// Optional map with per-feature ranges: {feature: {low: , high: }}
  final Map<String, dynamic>? ranges;

  factory MetaResponse.fromJson(Map<String, dynamic> json) =>
      _$MetaResponseFromJson(json);

  Map<String, dynamic> toJson() => _$MetaResponseToJson(this);
}
