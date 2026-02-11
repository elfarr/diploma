// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'predict_response.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

PredictResponse _$PredictResponseFromJson(Map<String, dynamic> json) =>
    PredictResponse(
      classLabel: json['class'] as String,
      probCal: (json['p_cal'] as num).toDouble(),
      confidence: (json['confidence'] as num).toDouble(),
      thresholds: json['thresholds'] as Map<String, dynamic>,
      explain:
          (json['explain'] as List<dynamic>?)
              ?.map((e) => ExplainItem.fromJson(e as Map<String, dynamic>))
              .toList(),
      ood: json['ood'],
      modelVersion: json['model_version'] as String?,
      schemaVersion: json['schema_version'] as String?,
      timingMs: (json['timing_ms'] as num?)?.toInt(),
      pRaw: (json['p_raw'] as num?)?.toDouble(),
    );

Map<String, dynamic> _$PredictResponseToJson(PredictResponse instance) =>
    <String, dynamic>{
      'class': instance.classLabel,
      'p_cal': instance.probCal,
      'confidence': instance.confidence,
      'thresholds': instance.thresholds,
      'explain': instance.explain?.map((e) => e.toJson()).toList(),
      'ood': instance.ood,
      'model_version': instance.modelVersion,
      'schema_version': instance.schemaVersion,
      'timing_ms': instance.timingMs,
      'p_raw': instance.pRaw,
    };
