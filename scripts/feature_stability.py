import json
import os
import pandas as pd
import matplotlib.pyplot as plt

JSON_PATH = "reports/tables/selected_features_by_outer.json"
OUT_PNG = "reports/figs/feature_stability.png"
OUT_CSV = "reports/tables/feature_stability.csv"

os.makedirs("reports/figs", exist_ok=True)
os.makedirs("reports/tables", exist_ok=True)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    selected = json.load(f)

selected = {int(k): v for k, v in selected.items()}

outers = sorted(selected.keys())
all_features = sorted({feat for feats in selected.values() for feat in feats})

mat = pd.DataFrame(0, index=all_features, columns=[f"outer_{o}" for o in outers], dtype=int)

for o in outers:
    for feat in selected[o]:
        mat.loc[feat, f"outer_{o}"] = 1

mat["freq"] = mat.sum(axis=1)
mat["freq_rate"] = mat["freq"] / len(outers)

mat_sorted = mat.sort_values(["freq", "freq_rate"], ascending=False)

mat_sorted.to_csv(OUT_CSV, index=True, encoding="utf-8-sig")
heat = mat_sorted[[c for c in mat_sorted.columns if c.startswith("outer_")]]

plt.figure(figsize=(8, max(3, 0.35 * len(heat))))  # чтобы не было каши по высоте
plt.imshow(heat.values, aspect="auto")

plt.xticks(range(len(heat.columns)), heat.columns, rotation=0)
plt.yticks(range(len(heat.index)), heat.index)

plt.title("Стабильность признаков по внешним фолдам")
plt.xlabel("Внешний фолд")
plt.ylabel("Признак")

for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        plt.text(j, i, str(int(heat.iat[i, j])), ha="center", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.close()

stable = mat_sorted[mat_sorted["freq"] >= 4].index.tolist()