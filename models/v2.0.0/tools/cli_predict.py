import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_feature_order(model_dir: Path, signature: dict) -> list[str]:
    model_pkl = model_dir / "final_model.pkl"
    if model_pkl.exists():
        try:
            model = joblib.load(model_pkl)
            booster = model.get_booster()
            if booster is not None and booster.feature_names:
                return list(booster.feature_names)
        except Exception:
            pass
    feats = signature["input"]["features"]
    return [f["name"] for f in feats]


def validate_and_vectorize(sample: dict, feature_order: list[str], feature_meta: dict):
    vec = []
    warnings = []
    errors = []

    for name in feature_order:
        meta = feature_meta.get(name, {})
        if name not in sample:
            warnings.append(f"Признак «{name}» не найден во входных данных: подставлен 0")
            val = 0.0
        else:
            val = sample[name]

        try:
            val = float(val)
        except Exception:
            errors.append(f"Признак {name}: значение не число ({val})")
            continue

        mn, mx = meta.get("min", None), meta.get("max", None)
        if mn is not None and val < mn:
            warnings.append(f"Признак {name}: {val} < минимум {mn}")
        if mx is not None and val > mx:
            warnings.append(f"Признак {name}: {val} > максимум {mx}")

        vec.append(val)

    if errors:
        raise ValueError("Ошибка валидации входных данных:\n" + "\n".join(errors))

    arr = np.array(vec, dtype=np.float32).reshape(1, -1)
    return arr, warnings


def apply_thresholds(p: float, thr: dict) -> str:
    t_low = float(thr["t_low"])
    t_high = float(thr["t_high"])
    if p < t_low:
        return "благоприятный"
    if p > t_high:
        return "неблагоприятный"
    return "неопределено"


def run_model(model_dir: Path, x: np.ndarray) -> float:
    model = joblib.load(model_dir / "final_model.pkl")
    proba = model.predict_proba(x)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return float(proba[0, 1])
    return float(proba.ravel()[0])


def main():
    ap = argparse.ArgumentParser(description="CLI для инференса модели (sklearn)")
    ap.add_argument("--input", required=True, help="Путь к входному JSON с признаками")
    ap.add_argument("--model", required=True, help="Путь к папке модели")
    ap.add_argument("--parity", action="store_true", help="Сравнить с expected в input JSON")
    args = ap.parse_args()

    model_dir = Path(args.model)
    sample_path = Path(args.input)

    signature = load_json(model_dir / "signature.json")
    thresholds = load_json(model_dir / "thresholds.json")
    sample = load_json(sample_path)

    feature_order = build_feature_order(model_dir, signature)
    feature_meta = {f["name"]: f for f in signature["input"]["features"]}

    x, warnings = validate_and_vectorize(sample, feature_order, feature_meta)
    p = run_model(model_dir, x)
    verdict = apply_thresholds(p, thresholds)

    result = {
        "версия": signature.get("version", "unknown"),
        "вероятность_неблагоприятного": round(p, 6),
        "вердикт": verdict,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if warnings:
        print("\nПредупреждения:")
        for w in warnings:
            print(f"- {w}")

    if args.parity:
        expected = sample.get("expected")
        if not expected:
            return
        exp_p = expected.get("p_unfavorable")
        exp_v = expected.get("verdict")
        ok = True
        if exp_p is not None:
            diff = abs(float(exp_p) - float(result["вероятность_неблагоприятного"]))
            if diff > 1e-3:
                ok = False
                print(f"ожидалось={exp_p}, получили={result['вероятность_неблагоприятного']}, diff={diff}")
        if exp_v is not None:
            verdict_map = {
                "благоприятный": "favorable",
                "неблагоприятный": "unfavorable",
                "неопределено": "undetermined",
            }
            got_v_compare = verdict_map.get(result["вердикт"], result["вердикт"])
            if exp_v != got_v_compare:
                ok = False
                print(f"ожидалось={exp_v}, получили={got_v_compare}")

if __name__ == "__main__":
    main()
