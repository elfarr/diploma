import json
import pandas as pd

path = "reports/tables/frozen_best_combo_per_outer.csv" 
df = pd.read_csv(path)

selected_features = {}

for _, row in df.iterrows():
    outer = int(row["outer"])
    feats = json.loads(row["features_json"])
    selected_features[outer] = feats


import os
os.makedirs("reports/tables", exist_ok=True)
with open("reports/tables/selected_features_by_outer.json", "w", encoding="utf-8") as f:
    json.dump(selected_features, f, ensure_ascii=False, indent=2)

