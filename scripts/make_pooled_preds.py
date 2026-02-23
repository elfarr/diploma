from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


PRED_FILE_RE = re.compile(r"^preds_(?P<name>.+)_fold(?P<fold>\d+)\.csv$")


def collect_fold_files(preds_dir: Path) -> Dict[str, List[Tuple[int, Path]]]:
    if not preds_dir.exists():
        raise FileNotFoundError()

    mapping: Dict[str, List[Tuple[int, Path]]] = {}
    for path in preds_dir.glob("preds_*_fold*.csv"):
        m = PRED_FILE_RE.match(path.name)
        if not m:
            continue
        name = m.group("name")
        fold = int(m.group("fold"))
        mapping.setdefault(name, []).append((fold, path))

    return mapping


def validate_df(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    required = {"y_true", "p_cal"}

    out = df[["y_true", "p_cal"]].copy()
    if out["y_true"].isna().any() or out["p_cal"].isna().any():
        raise ValueError()

    try:
        y = out["y_true"].astype(int)
    except Exception as exc:  # noqa: BLE001
        raise ValueError() from exc

    uniq = sorted(y.unique().tolist())
    if any(v not in (0, 1) for v in uniq):
        if len(uniq) == 2:
            y = y.map({uniq[0]: 0, uniq[1]: 1}).astype(int)
        elif len(uniq) == 1 and uniq[0] not in (0, 1):
            raise ValueError()

    try:
        p = out["p_cal"].astype(float)
    except Exception as exc:  # noqa: BLE001
        raise ValueError() from exc

    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError()

    out["y_true"] = y
    out["p_cal"] = p
    return out


def build_pooled(files: List[Tuple[int, Path]]) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for fold, path in sorted(files, key=lambda x: x[0]):
        df = pd.read_csv(path)
        df_valid = validate_df(df, path)
        df_valid["outer_fold"] = fold
        parts.append(df_valid)
    pooled = pd.concat(parts, ignore_index=True)
    return pooled


def resolve_preds_dir(preds_dir: Path) -> Path:
    if any(preds_dir.glob("preds_*_fold*.csv")):
        return preds_dir
    fallback = Path("reports/tables")
    if any(fallback.glob("preds_*_fold*.csv")):
        return fallback
    return preds_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", type=Path, default=Path("reports/preds"))
    args = parser.parse_args()

    preds_dir = resolve_preds_dir(args.preds_dir)
    file_map = collect_fold_files(preds_dir)

    for name, files in sorted(file_map.items()):
        pooled = build_pooled(files)
        out_path = preds_dir / f"preds_{name}_pooled.csv"
        pooled.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()

