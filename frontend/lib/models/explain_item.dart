import 'package:json_annotation/json_annotation.dart';

part 'explain_item.g.dart';

@JsonSerializable()
class ExplainItem {
  const ExplainItem({
    required this.feature,
    required this.impact,
    required this.direction,
  });

  final String feature;
  final double impact;
  final String direction;

  factory ExplainItem.fromJson(Map<String, dynamic> json) =>
      _$ExplainItemFromJson(json);

  Map<String, dynamic> toJson() => _$ExplainItemToJson(this);
}
