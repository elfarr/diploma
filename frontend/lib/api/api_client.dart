import 'dart:convert';

import 'package:http/http.dart' as http;

import '../app_env.dart';
import '../models/predict_request.dart';
import '../models/predict_response.dart';

class ApiException implements Exception {
  ApiException(this.statusCode, this.message);

  final int statusCode;
  final String message;

  @override
  String toString() => 'ApiException($statusCode): $message';
}

class ApiClient {
  ApiClient({http.Client? client}) : _client = client ?? http.Client();

  final http.Client _client;

  Future<PredictResponse> predict(PredictRequest request) async {
    final uri = _buildUri();
    final headers = <String, String>{
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    };

    if (!AppEnv.useProxy && AppEnv.apiToken.trim().isNotEmpty) {
      headers['Authorization'] = 'Bearer ${AppEnv.apiToken}';
    }

    final response = await _client.post(
      uri,
      headers: headers,
      body: jsonEncode(<String, dynamic>{
        'features': request.toJson(),
      }),
    );

    if (response.statusCode >= 200 && response.statusCode < 300) {
      final data = (jsonDecode(response.body) as Map).cast<String, dynamic>();
      return PredictResponse.fromJson(data);
    }

    throw ApiException(response.statusCode, _readMessage(response.body));
  }

  Uri _buildUri() {
    if (AppEnv.useProxy) {
      return Uri.parse('/api/predict');
    }
    final base = AppEnv.apiBaseUrl.endsWith('/')
        ? AppEnv.apiBaseUrl.substring(0, AppEnv.apiBaseUrl.length - 1)
        : AppEnv.apiBaseUrl;
    return Uri.parse('$base/predict');
  }

  String _readMessage(String body) {
    try {
      final decoded = (jsonDecode(body) as Map).cast<String, dynamic>();
      final message = decoded['message'] ?? decoded['error'] ?? decoded['detail'];
      if (message is String && message.isNotEmpty) {
        return message;
      }
    } catch (_) {
    }
    return body;
  }
}
