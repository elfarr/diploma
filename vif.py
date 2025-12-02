import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# === 1. Загружаем Excel-файл ===
file_path = "data/processed/dataset.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')
df_numeric = df.select_dtypes(include=["int64", "float64"])

# === 2. Нормализация количественных признаков ===
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

# === 3. Расчёт VIF ===
vif_data = pd.DataFrame()
vif_data["Variable"] = df_scaled.columns
vif_data["VIF"] = [variance_inflation_factor(df_scaled.values, i) for i in range(df_scaled.shape[1])]

# === 4. Сохраняем VIF в Excel ===
os.makedirs("reports/tables", exist_ok=True)
vif_data.to_excel("reports/tables/vif.xlsx", index=False)

# === 5. Строим и сохраняем график VIF ===
vif_data_sorted = vif_data.sort_values(by="VIF", ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(vif_data_sorted["Variable"], vif_data_sorted["VIF"], color="steelblue")
plt.axvline(x=5, color="orange", linestyle="--", label="VIF = 5 (порог)")
plt.axvline(x=10, color="red", linestyle="--", label="VIF = 10 (критично)")
plt.xlabel("VIF")
plt.ylabel("Признаки")
plt.title("Распределение значений VIF по количественным признакам")
plt.legend()
plt.tight_layout()

# Создаём папку для графиков, если нет
os.makedirs("reports/figures", exist_ok=True)
plt.savefig("reports/figures/vif_plot.png", dpi=300)
plt.show()
