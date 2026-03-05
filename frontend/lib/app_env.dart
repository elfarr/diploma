class AppEnv {
  const AppEnv._();

  static const String apiBaseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: '',
  );

  static const bool useProxy = bool.fromEnvironment(
    'USE_PROXY',
    defaultValue: false,
  );

  static const String apiToken = String.fromEnvironment(
    'API_TOKEN',
    defaultValue: '',
  );

  static const String modelVersion = String.fromEnvironment(
    'MODEL_VERSION',
    defaultValue: 'dev',
  );

  static const String schemaVersion = String.fromEnvironment(
    'SCHEMA_VERSION',
    defaultValue: 'dev',
  );

  static const String buildDate = String.fromEnvironment(
    'BUILD_DATE',
    defaultValue: '2026-03-04',
  );

  static bool get hasBaseUrl => apiBaseUrl.isNotEmpty;
}
