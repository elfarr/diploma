// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'explain_item.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

ExplainItem _$ExplainItemFromJson(Map<String, dynamic> json) => ExplainItem(
  feature: json['feature'] as String,
  impact: (json['impact'] as num).toDouble(),
  direction: json['direction'] as String,
);

Map<String, dynamic> _$ExplainItemToJson(ExplainItem instance) =>
    <String, dynamic>{
      'feature': instance.feature,
      'impact': instance.impact,
      'direction': instance.direction,
    };
