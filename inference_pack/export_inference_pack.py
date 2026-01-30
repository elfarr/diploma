import os
import json
import numpy as np
import pandas as pd
import joblib

SEP = ";"
DATA_PATH = "data/processed/with_inner_folds.csv"  
TARGET_COL = "Исход"                             
MODEL_IN_PATH = "models/final_model.pkl"          
OUT_DIR = "inference_pack"

T_LOW = 0.35
T_HIGH = 0.55

os.makedirs(OUT_DIR, exist_ok=True)

def to_float_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
    return pd.to_numeric(s2, errors="coerce")

def build_schema_from_dataset(df: pd.DataFrame):
    drop_cols = {"fold_outer", "fold_inner"}
    cols = [c for c in df.columns if c not in drop_cols]

    for c in ["row_index", TARGET_COL]:
        if c in cols:
            cols.remove(c)

    X_raw = df[cols].copy()

    obj_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        conv = to_float_series(X_raw[c])
        if conv.notna().mean() >= 0.5:
            X_raw[c] = conv

    cat_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_raw.columns if c not in cat_cols]

    medians = {}
    for c in num_cols:
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")
        med = float(X_raw[c].median()) if X_raw[c].notna().any() else 0.0
        medians[c] = med

    categories = {}
    for c in cat_cols:
        categories[c] = sorted(X_raw[c].dropna().astype(str).unique().tolist())

    X_ohe = X_raw.copy()
    for c in cat_cols:
        X_ohe[c] = X_ohe[c].astype(str)

    X = pd.get_dummies(X_ohe, columns=cat_cols, drop_first=False)
    ohe_columns = X.columns.tolist()

    schema = {
        "target_col": TARGET_COL,
        "t_low": T_LOW,
        "t_high": T_HIGH,
        "raw_feature_cols": cols,    
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "medians": medians,
        "categories": categories,
        "ohe_columns": ohe_columns,     
    }
    return schema

def build_signature(schema: dict):
    sig = []
    for f in schema["raw_feature_cols"]:
        f_type = "numeric" if f in schema["num_cols"] else "categorical"
        sig.append({
            "name": f,
            "type": f_type,
            "units": "",
            "valid_range": "" if f_type == "categorical" else "",
            "missing": "allowed",
            "impute": "median" if f_type == "numeric" else "unknown_category->all_zeros"
        })
    return {"features": sig}

def main():
    df = pd.read_csv(DATA_PATH, sep=SEP)
    if "row_index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "row_index"})

    schema = build_schema_from_dataset(df)

    with open(os.path.join(OUT_DIR, "preprocess.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    signature = build_signature(schema)
    with open(os.path.join(OUT_DIR, "signature.json"), "w", encoding="utf-8") as f:
        json.dump(signature, f, ensure_ascii=False, indent=2)

    if os.path.exists(MODEL_IN_PATH):
        model = joblib.load(MODEL_IN_PATH)
        joblib.dump(model, os.path.join(OUT_DIR, "model.pkl"))

if __name__ == "__main__":
    main()
