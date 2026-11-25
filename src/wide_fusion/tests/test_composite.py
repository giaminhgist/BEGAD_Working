import numpy as np
import pytest
from wide_fusion import composite as W


def test_same_values_as_cognitive_fusion_composite():
    from cognitive_fusion import composite as C
    rng = np.random.default_rng(0)
    S = rng.normal(20, 5, size=(60, 5))
    S[rng.random(S.shape) < 0.15] = np.nan
    y = (rng.random(60) < 0.5).astype(int)
    S[y == 1] -= 3
    assert W.SCORE_COLS == C.SCORE_COLS
    sign = {c: -1 for c in W.SCORE_COLS}
    pw, pc = W.composite_params(S, y), C.composite_params(S, y)
    assert np.array_equal(pw["mu"], pc["mu"]) and np.array_equal(pw["sd"], pc["sd"])
    cw, nw = W.composite(S, pw, W.sign_vector(sign), 3)
    cc, nc = C.composite(S, pc, C.sign_vector(sign), 3)
    assert np.array_equal(cw, cc, equal_nan=True) and np.array_equal(nw, nc)


def test_sign_assertion_stops_when_training_CN_mean_is_not_higher():
    S = np.array([[10, 5, 20, 3, 4], [20, 7, 18, 4, 5], [30, 9, 19, 5, 6], [40, 3, 17, 6, 7]], float)
    with pytest.raises(RuntimeError):
        W.composite_params(S, np.array([0, 0, 1, 1]))
