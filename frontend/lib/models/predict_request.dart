import 'package:json_annotation/json_annotation.dart';

part 'predict_request.g.dart';

@JsonSerializable()
class PredictRequest {
  const PredictRequest({
    required this.features,
    required this.unitConvert,
  });

  final Map<String, double> features;

  @JsonKey(name: 'unit_convert')
  final bool unitConvert;

  factory PredictRequest.fromJson(Map<String, dynamic> json) =>
      _$PredictRequestFromJson(json);

  Map<String, dynamic> toJson() => _$PredictRequestToJson(this);
}
