import joblib
from pathlib import Path

from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

VERSION = "v2.0.0"

BASE_DIR = Path(__file__).resolve().parent.parent
PIPELINE_PATH = BASE_DIR / "final_model.pkl"
OUT_DIR = BASE_DIR / "onnx"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pipe = joblib.load(PIPELINE_PATH)
    input_dim = getattr(pipe, "n_features_in_", 8)
    booster = pipe.get_booster()
    if booster is not None and booster.feature_names:
        booster.feature_names = [f"f{i}" for i in range(len(booster.feature_names))]
        pipe._Booster = booster

    initial_type = [("input", FloatTensorType([None, input_dim]))]

    onnx_model = convert_xgboost(pipe, initial_types=initial_type, target_opset=15)

    onnx_path = OUT_DIR / "model.onnx"
    onnx_path.write_bytes(onnx_model.SerializeToString())


if __name__ == "__main__":
    main()
