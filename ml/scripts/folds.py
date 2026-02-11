import pandas as pd
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv("data/processed/dataset_v1_folds.csv", sep=";")
df = df.reset_index(drop=True)
df["fold_inner"] = -1

target_col = "Исход"

for outer_fold in sorted(df["fold_outer"].unique()):
    is_train = df["fold_outer"] != outer_fold
    train_data = df[is_train].copy()
    y_train = train_data[target_col].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for inner_fold, (_, val_idx) in enumerate(skf.split(train_data, y_train), start=1):
        real_indices = train_data.iloc[val_idx].index
        df.loc[real_indices, "fold_inner"] = inner_fold

print(df["fold_inner"].value_counts())
df.to_csv("data/processed/with_inner_folds.csv", sep=";", index=False)