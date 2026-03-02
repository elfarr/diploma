class PredictRequest {
  const PredictRequest({
    required this.features,
  });

  final Map<String, dynamic> features;

  Map<String, dynamic> toJson() => features;
}
