import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

df = pd.read_csv("reports/tables/preds_mlp.csv", sep=";", header=None)
df.columns = ["y_true", "y_pred_class", "y_prob_raw"]
df["y_prob"] = df["y_pred_class"].astype(str).str.replace(",", ".").astype(float)

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            acc = np.mean(y_true[in_bin])
            conf = np.mean(y_prob[in_bin])
            ece += (bin_size / len(y_true)) * abs(acc - conf)
    return ece

def plot_calibration_curve(y_true, y_prob, model_name="Model"):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label="Модель")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Идеально")
    plt.xlabel("Предсказанная вероятность")
    plt.ylabel("Фактическая частота")
    plt.title(f"Калибровочная диаграмма ({model_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"reports/figures/calibration_{model_name}.png", dpi=300)
    plt.show()

ece = compute_ece(df["y_true"], df["y_prob"])
print(f"Expected Calibration Error (ECE): {ece:.4f}")

plot_calibration_curve(df["y_true"], df["y_prob"], model_name="MLP")
