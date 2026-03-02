import '../models/predict_request.dart';
import '../models/predict_response.dart';
import 'api_client.dart';

class PredictorRepo {
  PredictorRepo(this._client);

  final ApiClient _client;

  Future<PredictResponse> predict(PredictRequest req) async {
    return _client.predict(req);
  }
}
