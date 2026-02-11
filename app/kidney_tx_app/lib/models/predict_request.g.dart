// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'predict_request.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

PredictRequest _$PredictRequestFromJson(Map<String, dynamic> json) =>
    PredictRequest(
      features: (json['features'] as Map<String, dynamic>).map(
        (k, e) => MapEntry(k, (e as num).toDouble()),
      ),
      unitConvert: json['unit_convert'] as bool,
    );

Map<String, dynamic> _$PredictRequestToJson(PredictRequest instance) =>
    <String, dynamic>{
      'features': instance.features,
      'unit_convert': instance.unitConvert,
    };
