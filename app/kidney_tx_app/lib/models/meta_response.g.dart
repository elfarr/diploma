// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'meta_response.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

MetaResponse _$MetaResponseFromJson(Map<String, dynamic> json) => MetaResponse(
  modelVersion: json['model_version'] as String,
  schemaVersion: json['schema_version'] as String,
  thresholds: json['thresholds'] as Map<String, dynamic>,
  features:
      (json['features'] as List<dynamic>).map((e) => e as String).toList(),
  ranges: json['ranges'] as Map<String, dynamic>?,
);

Map<String, dynamic> _$MetaResponseToJson(MetaResponse instance) =>
    <String, dynamic>{
      'model_version': instance.modelVersion,
      'schema_version': instance.schemaVersion,
      'thresholds': instance.thresholds,
      'features': instance.features,
      'ranges': instance.ranges,
    };
