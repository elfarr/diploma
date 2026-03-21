from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


COL_SEX = "\u041f\u043e\u043b"
COL_DIAGNOSIS = "\u0414\u0438\u0430\u0433\u043d\u043e\u0437"
COL_IBS = "\u0418\u0411\u0421 \u0434\u043e \u0422\u041f"
COL_HSN_STAGE = "\u0441\u0442\u0430\u0434\u0438\u044f \u0425\u0421\u041d \u043f\u0435\u0440\u0435\u0434 \u0422\u041f"
COL_DAD = "\u0414\u0410\u0414 \u043f\u0435\u0440\u0435\u0434 \u0422\u041f"
COL_LDL = "\u041b\u041f\u041d\u041f \u043f\u0435\u0440\u0435\u0434 \u0422\u041f"
COL_REL_RISK = "relative risk"
COL_QRISK3 = "QRISK3"
COL_HEALTHY_RISK = "healthy person risk"
COL_QRISK_AGE = "qrisk age"

RISK_FAMILY_COLS = [COL_QRISK3, COL_HEALTHY_RISK, COL_REL_RISK, COL_QRISK_AGE]

VAL_MALE = "\u043c\u0443\u0436"
VAL_DIAG_HGN = "\u0425\u0413\u041d"
VAL_DIAG_DIABETES = "\u0421\u0430\u0445\u0430\u0440\u043d\u044b\u0439 \u0434\u0438\u0430\u0431\u0435\u0442"
VAL_YES = "\u0435\u0441\u0442\u044c"
VAL_HSN_2FK = "2 \u0424\u041a"

ENG_FLAG_MALE = "eng_flag_male"
ENG_FLAG_DIAG_HGN = "eng_flag_diag_hgn"
ENG_FLAG_DIAG_DIABETES = "eng_flag_diag_diabetes"
ENG_FLAG_IBS_YES = "eng_flag_ibs_yes"
ENG_FLAG_HSN_2FK = "eng_flag_hsn2fk"
ENG_RR_X_DAD = "eng_relative_risk_x_dad"
ENG_LDL_X_RR = "eng_ldlp_x_relative_risk"
ENG_LDL_X_IBS = "eng_ldlp_x_ibs_yes"
ENG_MALE_X_HGN = "eng_male_x_diag_hgn"
ENG_HSN2_X_IBS = "eng_hsn2fk_x_ibs_yes"
ENG_RR_X_HGN = "eng_relative_risk_x_diag_hgn"
ENG_DAD_X_IBS = "eng_dad_x_ibs_yes"
ENG_FP_SUBGROUP_SCORE = "eng_fp_subgroup_score"


def apply_risk_feature_mode(feature_pool: Sequence[str], mode: str) -> List[str]:
    mode_norm = str(mode).strip().lower()
    features = [str(col) for col in feature_pool]

    if mode_norm in {"all", ""}:
        return features

    keep_map = {
        "relative_only": {COL_REL_RISK},
        "qrisk_only": {COL_QRISK3},
        "qrisk3_only": {COL_QRISK3},
        "healthy_only": {COL_HEALTHY_RISK},
        "qrisk_age_only": {COL_QRISK_AGE},
        "qrisk3_plus_relative": {COL_QRISK3, COL_REL_RISK},
    }
    if mode_norm not in keep_map:
        raise ValueError(
            f"Unsupported risk feature mode: {mode}. "
            "Expected one of: all, relative_only, qrisk_only, qrisk3_only, "
            "healthy_only, qrisk_age_only, qrisk3_plus_relative."
        )

    keep = keep_map[mode_norm]
    return [col for col in features if col not in RISK_FAMILY_COLS or col in keep]


def load_preprocess_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_numeric_string_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(
            out[col].astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False),
            errors="coerce",
        )
    return out


def resolve_declared_feature_types(
    feature_pool: Sequence[str],
    preprocess_cfg: dict[str, Any] | None,
) -> Tuple[List[str], List[str]]:
    if preprocess_cfg is None:
        return [], []
    num_cols = [str(col) for col in preprocess_cfg.get("num_cols", []) if str(col) in feature_pool]
    cat_cols = [str(col) for col in preprocess_cfg.get("cat_cols", []) if str(col) in feature_pool]
    return num_cols, cat_cols


def _series_text_equals(series: pd.Series, value: str) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().eq(value.strip().lower())


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def add_fp_signal_features(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    new_cols: List[str] = []

    male = None
    diag_hgn = None
    diag_diabetes = None
    ibs_yes = None
    hsn_2fk = None
    dad = None
    ldl = None
    rel_risk = None

    if COL_SEX in out.columns:
        male = _series_text_equals(out[COL_SEX], VAL_MALE).astype(float)
        out[ENG_FLAG_MALE] = male
        new_cols.append(ENG_FLAG_MALE)
    if COL_DIAGNOSIS in out.columns:
        diag_hgn = _series_text_equals(out[COL_DIAGNOSIS], VAL_DIAG_HGN).astype(float)
        diag_diabetes = _series_text_equals(out[COL_DIAGNOSIS], VAL_DIAG_DIABETES).astype(float)
        out[ENG_FLAG_DIAG_HGN] = diag_hgn
        out[ENG_FLAG_DIAG_DIABETES] = diag_diabetes
        new_cols.extend([ENG_FLAG_DIAG_HGN, ENG_FLAG_DIAG_DIABETES])
    if COL_IBS in out.columns:
        ibs_yes = _series_text_equals(out[COL_IBS], VAL_YES).astype(float)
        out[ENG_FLAG_IBS_YES] = ibs_yes
        new_cols.append(ENG_FLAG_IBS_YES)
    if COL_HSN_STAGE in out.columns:
        hsn_2fk = _series_text_equals(out[COL_HSN_STAGE], VAL_HSN_2FK).astype(float)
        out[ENG_FLAG_HSN_2FK] = hsn_2fk
        new_cols.append(ENG_FLAG_HSN_2FK)
    if COL_DAD in out.columns:
        dad = _safe_numeric(out[COL_DAD])
    if COL_LDL in out.columns:
        ldl = _safe_numeric(out[COL_LDL])
    if COL_REL_RISK in out.columns:
        rel_risk = _safe_numeric(out[COL_REL_RISK])

    if rel_risk is not None and dad is not None:
        out[ENG_RR_X_DAD] = rel_risk * dad
        new_cols.append(ENG_RR_X_DAD)
    if ldl is not None and rel_risk is not None:
        out[ENG_LDL_X_RR] = ldl * rel_risk
        new_cols.append(ENG_LDL_X_RR)
    if ldl is not None and ibs_yes is not None:
        out[ENG_LDL_X_IBS] = ldl * ibs_yes
        new_cols.append(ENG_LDL_X_IBS)
    if male is not None and diag_hgn is not None:
        out[ENG_MALE_X_HGN] = male * diag_hgn
        new_cols.append(ENG_MALE_X_HGN)
    if hsn_2fk is not None and ibs_yes is not None:
        out[ENG_HSN2_X_IBS] = hsn_2fk * ibs_yes
        new_cols.append(ENG_HSN2_X_IBS)
    if rel_risk is not None and diag_hgn is not None:
        out[ENG_RR_X_HGN] = rel_risk * diag_hgn
        new_cols.append(ENG_RR_X_HGN)
    if dad is not None and ibs_yes is not None:
        out[ENG_DAD_X_IBS] = dad * ibs_yes
        new_cols.append(ENG_DAD_X_IBS)

    subgroup_parts = [
        out[col]
        for col in [ENG_FLAG_DIAG_HGN, ENG_FLAG_HSN_2FK, ENG_FLAG_IBS_YES, ENG_FLAG_MALE]
        if col in out.columns
    ]
    if subgroup_parts:
        subgroup_score = subgroup_parts[0].copy().astype(float)
        for series in subgroup_parts[1:]:
            subgroup_score = subgroup_score + series.astype(float)
        out[ENG_FP_SUBGROUP_SCORE] = subgroup_score
        new_cols.append(ENG_FP_SUBGROUP_SCORE)

    return out, new_cols
