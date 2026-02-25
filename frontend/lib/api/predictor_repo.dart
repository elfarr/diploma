import 'package:dio/dio.dart';

import '../models/meta_response.dart';
import '../models/predict_request.dart';
import '../models/predict_response.dart';
import 'api_client.dart';

class PredictorRepo {
  PredictorRepo(this._client);

  final ApiClient _client;

  Future<MetaResponse> meta() async {
    try {
      final resp = await _client.getMeta();
      return MetaResponse.fromJson(resp.data ?? {});
    } on DioException catch (e) {
      throw _readableError(e);
    }
  }

  Future<PredictResponse> predict(PredictRequest req) async {
    try {
      final resp = await _client.postPredict(req.toJson());
      return PredictResponse.fromJson(resp.data ?? {});
    } on DioException catch (e) {
      throw _readableError(e);
    }
  }

  Exception _readableError(DioException e) {
    final status = e.response?.statusCode;
    final msg = e.response?.data is Map
        ? (e.response?.data['error'] ?? e.response?.data.toString())
        : e.message;
    return Exception('Request failed${status != null ? ' ($status)' : ''}: $msg');
  }
}
