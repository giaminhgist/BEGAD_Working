import numpy as np
import pytest
from cognitive_fusion.composite import SCORE_COLS, composite, composite_params, sign_vector

SIGN = sign_vector({c: -1 for c in SCORE_COLS})


def data():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(30), np.ones(20)].astype(int)
    S = rng.normal(size=(50, 5)) + 2 - y[:, None]          # CN higher than MCI in every score
    return S, y


def test_composite_matches_manual_formula():
    S, y = data()
    S[0, 1] = np.nan
    p = composite_params(S, y)
    C, nA = composite(S, p, SIGN)
    z = (S - np.nanmean(S, axis=0)) / np.nanstd(S, axis=0, ddof=1)
    assert np.allclose(p["sd"], np.nanstd(S, axis=0, ddof=1))
    assert C[1] == pytest.approx(-z[1].mean())
    assert C[0] == pytest.approx(-np.nanmean(z[0])) and nA[0] == 4
    assert C[y == 1].mean() > C[y == 0].mean()               # higher C = more impaired


def test_min_scores_rule():
    S, y = data()
    p = composite_params(S, y)
    S2 = S.copy()
    S2[0, :3] = np.nan                                          # 2 scores left
    S2[1, :2] = np.nan                                          # 3 scores left
    C, nA = composite(S2, p, SIGN, min_scores=3)
    assert np.isnan(C[0]) and nA[0] == 2 and np.isfinite(C[1])


def test_wrong_sign_stops():
    S, y = data()
    with pytest.raises(RuntimeError, match="CN mean"):
        composite_params(S, 1 - y)


def test_zero_sd_stops():
    S, y = data()
    S[:, 2] = 1.0
    with pytest.raises(RuntimeError):
        composite_params(S, y, check_sign=False)


def test_params_depend_only_on_given_rows():
    S, y = data()
    a = composite_params(S[:40], y[:40])
    S2 = S.copy()
    S2[40:] = 999
    assert np.array_equal(a["mu"], composite_params(S2[:40], y[:40])["mu"])
