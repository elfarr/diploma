import json
import numpy as np
import pandas as pd
import joblib

class ModelInference:
    def __init__(self, pack_dir: str = "inference_pack"):
        self.pack_dir = pack_dir
        with open(f"{pack_dir}/preprocess.json", "r", encoding="utf-8") as f:
            self.schema = json.load(f)

        self.model = joblib.load(f"{pack_dir}/model.pkl")

        self.t_low = float(self.schema["t_low"])
        self.t_high = float(self.schema["t_high"])
        self.raw_cols = self.schema["raw_feature_cols"]
        self.num_cols = set(self.schema["num_cols"])
        self.cat_cols = set(self.schema["cat_cols"])
        self.medians = self.schema["medians"]
        self.categories = self.schema["categories"]
        self.ohe_columns = self.schema["ohe_columns"]

    @staticmethod
    def _to_float(x):
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip().replace(" ", "").replace(",", ".")
        try:
            return float(s)
        except:
            return np.nan

    def _preprocess_one(self, payload: dict) -> pd.DataFrame:
        row = {}
        for c in self.raw_cols:
            v = payload.get(c, None)
            if c in self.num_cols:
                row[c] = self._to_float(v)
            else:
                row[c] = None if v is None else str(v)

        df = pd.DataFrame([row])
        for c in self.num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                med = float(self.medians.get(c, 0.0))
                df[c] = df[c].fillna(med)

        for c in self.cat_cols:
            if c in df.columns:
                df[c] = df[c].astype(str)
                allowed = set(self.categories.get(c, []))
                if df.loc[0, c] not in allowed:
                    df.loc[0, c] = "__UNKNOWN__"

        X = pd.get_dummies(df, columns=list(self.cat_cols), drop_first=False)
        for col in self.ohe_columns:
            if col not in X.columns:
                X[col] = 0.0

        X = X[self.ohe_columns]
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

        return X

    def predict(self, payload: dict) -> dict:
        X = self._preprocess_one(payload)
        proba = self.model.predict_proba(X)[0]
        p_good = float(proba[1])
        p_bad = float(1.0 - p_good)

        if p_good < self.t_low:
            status = "неблагоприятный исход"
            decision = 0
        elif p_good > self.t_high:
            status = "благоприятный исход"
            decision = 1
        else:
            status = "неопределённо"
            decision = None

        return {
            "p_good": p_good,
            "p_bad": p_bad,
            "decision": decision,  
            "status": status,
            "t_low": self.t_low,
            "t_high": self.t_high
        }


if __name__ == "__main__":
    inf = ModelInference("inference_pack")
    sample = {
        "Пол": "жен",
        "Диагноз": "ХГН",
        "Донор": "трупная почка",
        "САД перед ТП": 140,
        "ДАД перед ТП": "80",
        "ЛПНП перед ТП": "1,7",
        "healthy person risk": "0,2",
        "relative risk": "9,6",
        "qrisk age": 47
    }

    print(inf.predict(sample))
