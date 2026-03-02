class AppEnv {
  const AppEnv._();

  static const String apiBaseUrl =
      String.fromEnvironment('API_BASE_URL', defaultValue: '');

  static const bool useProxy =
      bool.fromEnvironment('USE_PROXY', defaultValue: false);

  static const String apiToken =
      String.fromEnvironment('API_TOKEN', defaultValue: '');

  static bool get hasBaseUrl => apiBaseUrl.isNotEmpty;
}
