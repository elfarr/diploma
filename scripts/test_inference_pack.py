import json
import joblib
import numpy as np
import pandas as pd

PACK_DIR = "inference_pack"

model = joblib.load(f"{PACK_DIR}/model.pkl")

with open(f"{PACK_DIR}/preprocess.json", "r", encoding="utf-8") as f:
    prep = json.load(f)

with open(f"{PACK_DIR}/signature.json", "r", encoding="utf-8") as f:
    sig = json.load(f)

raw_features = prep["raw_feature_cols"]
ohe_columns = prep["ohe_columns"]
num_cols = prep["num_cols"]
cat_cols = prep["cat_cols"]
medians = prep["medians"]
t_low = prep["t_low"]
t_high = prep["t_high"]

patients = pd.read_csv("reports/tables/patients_level_preds.csv")
one = patients.iloc[0]

row = {}
for f in raw_features:
    row[f] = one.get(f, None)

X_raw = pd.DataFrame([row])

for c in num_cols:
    if c in X_raw.columns:
        X_raw[c] = (
            X_raw[c].astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
        )
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")
        X_raw[c] = X_raw[c].fillna(medians.get(c, 0))

X = pd.get_dummies(X_raw, columns=[c for c in cat_cols if c in X_raw.columns], drop_first=False)
X = X.reindex(columns=ohe_columns, fill_value=0).astype(np.float32)

p = float(model.predict_proba(X)[:, 1][0])

if p < t_low:
    status = "низкий риск"
elif p > t_high:
    status = "высокий риск"
else:
    status = "неопределённо"

print("p:", round(p, 4))
print("status:", status)
