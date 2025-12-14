import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import brier_score_loss
df = pd.read_csv("data/processed/with_inner_folds.csv", sep=";")

df["Исход"] = df["Исход"].astype(str).str.strip().replace({
    "благоприятный": 1,
    "неблагоприятный": 0
}).astype(int)

RANDOM_STATE = 42
outer_folds = sorted(df["fold_outer"].unique())
inner_folds = sorted(df["fold_inner"].unique())

excluded_cols = ['fold_outer', 'fold_inner', 'Исход']
all_features = [col for col in df.columns if col not in excluded_cols]
numeric_features_all = [col for col in all_features if pd.api.types.is_numeric_dtype(df[col])]
categorical_features_all = [col for col in all_features if col not in numeric_features_all]

results = []

for outer in outer_folds:
    outer_train = df[df["fold_outer"] != outer]

    for inner in inner_folds:
        inner_train = outer_train[outer_train["fold_inner"] != inner]
        inner_valid = outer_train[outer_train["fold_inner"] == inner]

        X_inner_train = inner_train.drop(columns=excluded_cols)
        y_inner_train = inner_train["Исход"]
        X_inner_valid = inner_valid.drop(columns=excluded_cols)
        y_inner_valid = inner_valid["Исход"]

        numeric_selected = []
        pvals_numeric = []
        for col in numeric_features_all:
            try:
                corr, pval = pointbiserialr(X_inner_train[col].astype(float), y_inner_train)
                if not np.isnan(pval):
                    pvals_numeric.append((pval, col))
            except Exception:
                continue

        pvals_numeric.sort(key=lambda x: x[0])
        numeric_selected = [col for _, col in pvals_numeric[:12]]

        categorical_selected = []
        for col in categorical_features_all:
            try:
                table = pd.crosstab(X_inner_train[col], y_inner_train)
                chi2, p, _, _ = chi2_contingency(table)
                if p < 0.05:
                    categorical_selected.append(col)
            except Exception:
                continue

        candidate_features = numeric_selected + categorical_selected

        scaler = StandardScaler()
        X_train_num = pd.DataFrame(scaler.fit_transform(X_inner_train[numeric_selected]), columns=numeric_selected, index=X_inner_train.index)
        X_valid_num = pd.DataFrame(scaler.transform(X_inner_valid[numeric_selected]), columns=numeric_selected, index=X_inner_valid.index)

        cat_dummies_train = {}
        cat_dummies_valid = {}
        for col in categorical_selected:
            train_dum = pd.get_dummies(X_inner_train[col], drop_first=True)
            train_dum = train_dum.reindex(sorted(train_dum.columns), axis=1)
            valid_dum = pd.get_dummies(X_inner_valid[col], drop_first=True)
            valid_dum = valid_dum.reindex(columns=train_dum.columns, fill_value=0)
            cat_dummies_train[col] = train_dum
            cat_dummies_valid[col] = valid_dum

        def evaluate_features(feats):
            parts = []
            for f in feats:
                if f in numeric_selected:
                    parts.append(X_train_num[[f]])
                elif f in categorical_selected:
                    parts.append(cat_dummies_train[f])
            if not parts:
                return None
            X = pd.concat(parts, axis=1)
            model = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            probs = cross_val_predict(model, X, y_inner_train, cv=cv, method="predict_proba")[:, 1]
            return brier_score_loss(y_inner_train, probs)

        selected_features = []
        current_score = float("inf")
        improved = True

        while improved and candidate_features:
            best_feature = None
            best_score = current_score
            for feat in candidate_features:
                trial = selected_features + [feat]
                score = evaluate_features(trial)
                if score is not None and score < best_score:
                    best_score = score
                    best_feature = feat
            if best_feature:
                selected_features.append(best_feature)
                candidate_features.remove(best_feature)
                current_score = best_score
            else:
                improved = False

        X_parts_train = []
        X_parts_valid = []
        for f in selected_features:
            if f in numeric_selected:
                X_parts_train.append(X_train_num[[f]])
                X_parts_valid.append(X_valid_num[[f]])
            elif f in categorical_selected:
                X_parts_train.append(cat_dummies_train[f])
                X_parts_valid.append(cat_dummies_valid[f])

        if X_parts_train:
            X_train_final = pd.concat(X_parts_train, axis=1)
            X_valid_final = pd.concat(X_parts_valid, axis=1)
            model_final = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
            model_final.fit(X_train_final, y_inner_train)
            preds = model_final.predict_proba(X_valid_final)[:, 1]
            final_brier = brier_score_loss(y_inner_valid, preds)
        else:
            final_brier = np.nan

        results.append({
            "outer_fold": outer,
            "inner_fold": inner,
            "selected_features": selected_features,
            "brier_score": final_brier
        })

results_df = pd.DataFrame(results)
results_df.to_csv("nested_cv_results.csv", index=False)

