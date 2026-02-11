import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("inner_model_selection_with_calibration_best.csv")

df_plot = df[["best_model", "brier_valid"]].copy()

df_plot["best_model"] = df_plot["best_model"].replace({
    "logreg": "Logistic Regression",
    "xgb": "XGBoost",
    "mlp": "MLP"
})

groups = [
    df_plot[df_plot["best_model"] == "Logistic Regression"]["brier_valid"],
    df_plot[df_plot["best_model"] == "XGBoost"]["brier_valid"],
    df_plot[df_plot["best_model"] == "MLP"]["brier_valid"]
]

plt.figure(figsize=(7, 5))
plt.boxplot(
    groups,
    labels=["LogReg", "XGBoost", "MLP"],
    showmeans=True
)

plt.ylabel("Brier Score")
plt.xlabel("Семейство моделей")
plt.title("Распределение значений Brier Score\nдля различных семейств моделей (inner-fold)")

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

plt.show()
