import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

import 'api_config.dart';

class ApiClient {
  ApiClient(this.config)
      : _dio = Dio(
          BaseOptions(
            baseUrl: config.baseUrl,
            connectTimeout: const Duration(seconds: 10),
            sendTimeout: const Duration(seconds: 10),
            receiveTimeout: const Duration(seconds: 10),
            responseType: ResponseType.json,
          ),
        ) {
    _dio.interceptors.add(
      InterceptorsWrapper(
        onRequest: (options, handler) {
          final headers = <String, Object?>{};
          if (config.bearerToken != null && config.bearerToken!.isNotEmpty) {
            headers['Authorization'] = 'Bearer ${config.bearerToken}';
          }
          if (config.modelVersion != null && config.modelVersion!.isNotEmpty) {
            headers['X-Model-Version'] = config.modelVersion;
          }
          options.headers.addAll(headers);

          if (kDebugMode) {
            debugPrint('[API] ${options.method} ${options.uri}');
          }
          handler.next(options);
        },
        onResponse: (response, handler) {
          if (kDebugMode) {
            debugPrint('[API] ${response.statusCode} ${response.requestOptions.uri}');
          }
          handler.next(response);
        },
        onError: (dioError, handler) {
          if (kDebugMode) {
            debugPrint(
                '[API] ${dioError.response?.statusCode ?? ''} ${dioError.requestOptions.uri}: ${dioError.message}');
          }
          handler.next(dioError);
        },
      ),
    );
  }

  final ApiConfig config;
  final Dio _dio;

  Future<Response<Map<String, dynamic>>> getMeta() {
    return _dio.get<Map<String, dynamic>>('/meta');
  }

  Future<Response<Map<String, dynamic>>> postPredict(
      Map<String, dynamic> body) {
    return _dio.post<Map<String, dynamic>>('/predict', data: body);
  }
}
