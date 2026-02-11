import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("outer_test_predictions.csv")

y_true = df["y_true"].astype(int).values
y_prob = df["y_pred_proba"].astype(float).values

n_bins = 10
bins = np.linspace(0.0, 1.0, n_bins + 1)

bin_id = np.digitize(y_prob, bins) - 1
bin_id = np.clip(bin_id, 0, n_bins - 1)

mean_pred = []
frac_pos = []
counts = []

for b in range(n_bins):
    mask = bin_id == b
    counts.append(mask.sum())
    if mask.sum() == 0:
        mean_pred.append(np.nan)
        frac_pos.append(np.nan)
    else:
        mean_pred.append(y_prob[mask].mean())
        frac_pos.append(y_true[mask].mean())

mean_pred = np.array(mean_pred, dtype=float)
frac_pos = np.array(frac_pos, dtype=float)
counts = np.array(counts, dtype=int)

ok = ~np.isnan(mean_pred) & ~np.isnan(frac_pos)
mean_pred = mean_pred[ok]
frac_pos = frac_pos[ok]

plt.figure(figsize=(7, 5))

plt.plot([0, 1], [0, 1], linestyle="--", label="Идеальная калибровка")

plt.plot(mean_pred, frac_pos, marker="o", label="Итоговая модель (outer-test)")

plt.xlabel("Средняя предсказанная вероятность")
plt.ylabel("Доля положительных исходов")
plt.title("Калибровочная кривая (reliability plot)\nдля итоговой модели на outer-test")

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig("reliability_plot_outer_test.png", dpi=300)
plt.show()
