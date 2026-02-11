class ApiConfig {
  const ApiConfig({
    required this.baseUrl,
    this.bearerToken,
    this.modelVersion,
  });

  final String baseUrl;
  final String? bearerToken;
  final String? modelVersion;

  ApiConfig copyWith({
    String? baseUrl,
    String? bearerToken,
    String? modelVersion,
  }) {
    return ApiConfig(
      baseUrl: baseUrl ?? this.baseUrl,
      bearerToken: bearerToken ?? this.bearerToken,
      modelVersion: modelVersion ?? this.modelVersion,
    );
  }
}
