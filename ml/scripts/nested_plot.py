import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("nested_cv_results.csv")
plt.figure(figsize=(8, 5))
sns.boxplot(x="outer_fold", y="brier_score", data=df, color="skyblue")
sns.stripplot(x="outer_fold", y="brier_score", data=df, color="black", alpha=0.7, jitter=True)
plt.title("Распределение Brier Score по внешним фолдам")
plt.xlabel("Номер внешнего фолда")
plt.ylabel("Brier Score")
plt.grid(True)
plt.tight_layout()
plt.show()
