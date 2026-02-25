import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services.predictor import select_model


def _make_competence(
    ece_svm=None,
    ece_cat=None,
    ece_mlp=None,
    brier_svm=None,
    brier_cat=None,
    brier_mlp=None,
):
    def _arr(v, default):
        if v is None:
            return [default] * 10
        return v

    return {
        "ece": {
            "svm_rbf": _arr(ece_svm, 0.30),
            "catboost": _arr(ece_cat, 0.20),
            "mlp": _arr(ece_mlp, 0.40),
        },
        "brier": {
            "svm_rbf": _arr(brier_svm, 0.30),
            "catboost": _arr(brier_cat, 0.20),
            "mlp": _arr(brier_mlp, 0.40),
        },
    }


@pytest.mark.parametrize(
    "p_avg, expected_bin",
    [
        (0.0, 0),
        (0.1, 1),
        (0.999, 9),
        (1.0, 9),
    ],
)
def test_bin_id_boundaries(p_avg, expected_bin):
    competence = _make_competence()
    out = select_model(p_avg, p_avg, p_avg, competence)
    assert out["bin_id"] == expected_bin


def test_select_by_min_ece():
    ece_svm = [0.30] * 10
    ece_cat = [0.20] * 10
    ece_mlp = [0.40] * 10

    ece_svm[5] = 0.01
    ece_cat[5] = 0.05
    ece_mlp[5] = 0.10

    competence = _make_competence(ece_svm=ece_svm, ece_cat=ece_cat, ece_mlp=ece_mlp)
    out = select_model(0.55, 0.55, 0.55, competence)
    assert out["bin_id"] == 5
    assert out["winner"] == "svm_rbf"


def test_tie_break_by_brier():
    ece_svm = [0.30] * 10
    ece_cat = [0.20] * 10
    ece_mlp = [0.40] * 10
    brier_svm = [0.30] * 10
    brier_cat = [0.20] * 10
    brier_mlp = [0.40] * 10

    ece_svm[4] = 0.05
    ece_cat[4] = 0.05
    brier_svm[4] = 0.09
    brier_cat[4] = 0.01

    competence = _make_competence(
        ece_svm=ece_svm,
        ece_cat=ece_cat,
        ece_mlp=ece_mlp,
        brier_svm=brier_svm,
        brier_cat=brier_cat,
        brier_mlp=brier_mlp,
    )
    out = select_model(0.45, 0.45, 0.45, competence)
    assert out["bin_id"] == 4
    assert out["winner"] == "catboost"


def test_model_with_none_ece_is_excluded():
    ece_svm = [0.30] * 10
    ece_cat = [0.20] * 10
    ece_mlp = [0.40] * 10

    ece_svm[2] = None
    ece_cat[2] = 0.10
    ece_mlp[2] = 0.20

    competence = _make_competence(ece_svm=ece_svm, ece_cat=ece_cat, ece_mlp=ece_mlp)
    out = select_model(0.25, 0.25, 0.25, competence)
    assert out["bin_id"] == 2
    assert out["winner"] == "catboost"


def test_fallback_when_all_ece_none_in_bin():
    ece_svm = [0.30] * 10
    ece_cat = [0.20] * 10
    ece_mlp = [0.40] * 10

    ece_svm[7] = None
    ece_cat[7] = None
    ece_mlp[7] = None

    competence = _make_competence(ece_svm=ece_svm, ece_cat=ece_cat, ece_mlp=ece_mlp)
    out = select_model(0.75, 0.75, 0.75, competence, fallback_model="svm_rbf")
    assert out["bin_id"] == 7
    assert out["winner"] == "svm_rbf"
