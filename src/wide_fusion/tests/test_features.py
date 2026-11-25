import numpy as np
import pytest
from wide_fusion import features as FT


def cards(T=4, n=40, p=5, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, p))
    return [base + 0.5 * rng.normal(size=(n, p)) + t for t in range(T)]     # shifted per card: the z must remove it


def test_card_params_ddof1_and_degenerate_flags():
    Xs = cards()
    Xs[1][:, 2] = 7.0                         # constant on card 1
    Xs[2][:, 3] = np.nan                      # no value on card 2
    Xs[3][:, 4] = np.nan
    Xs[3][0, 4] = 1.0                         # a single value on card 3
    rows = np.arange(30)
    par = FT.card_params(Xs, rows)
    assert np.allclose(par["mu"][0, 0], Xs[0][rows, 0].mean()) and np.allclose(par["sd"][0, 0], Xs[0][rows, 0].std(ddof=1))
    deg = par["degenerate"]
    assert deg[1, 2] and deg[2, 3] and deg[3, 4] and deg.sum() == 3
    assert np.isnan(par["sd"][deg]).all()


def test_wide_is_per_card_z_side_by_side_with_prefixed_names():
    Xs = cards(T=3)
    Xs[1][:, 1] = 2.0
    tr, q = np.arange(0, 30), np.arange(30, 40)
    d = FT.wide_features(Xs, list("abcde"), [7, 9, 11], tr, q)
    assert d["Xtr"].shape == (30, 15) and d["Xq"].shape == (10, 15)
    assert d["names"][0] == "t7__a" and d["names"][5] == "t9__a" and d["names"][14] == "t11__e"
    z = (Xs[2][:, 3] - Xs[2][tr, 3].mean()) / Xs[2][tr, 3].std(ddof=1)
    assert np.allclose(d["Xq"][:, 10 + 3], z[q])
    assert np.isnan(d["Xtr"][:, 5 + 1]).all() and d["log"]["n_degenerate"] == 1       # degenerate card-column stays NaN


def test_wide_keeps_one_row_per_visit_in_card_order_and_nan_values_stay_nan():
    Xs = cards(T=2)
    Xs[1][33, 2] = np.nan
    tr, q = np.arange(0, 30), np.arange(30, 40)
    d = FT.wide_features(Xs, list("abcde"), [1, 3], tr, q)
    assert np.isnan(d["Xq"][3, 5 + 2]) and not np.isnan(d["Xq"][3, 2])
    assert np.isnan(d["Xq"]).sum() == 1
    z0 = (Xs[0][:, 0] - Xs[0][tr, 0].mean()) / Xs[0][tr, 0].std(ddof=1)
    assert np.allclose(d["Xtr"][:, 0], z0[tr]) and d["log"]["excluded"] == []


def test_parameters_use_training_rows_only():
    Xs = cards(seed=3)
    tr, q = np.arange(0, 30), np.arange(30, 40)
    Xs_b = [x.copy() for x in Xs]
    for x in Xs_b:
        x[q] = np.random.default_rng(9).normal(size=x[q].shape) * 100      # query rows replaced by something else entirely
    a = FT.wide_features(Xs, list("abcde"), [1, 3, 4, 5], tr, q)
    b = FT.wide_features(Xs_b, list("abcde"), [1, 3, 4, 5], tr, q)
    assert np.array_equal(a["Xtr"], b["Xtr"]) and a["log"]["degenerate"] == b["log"]["degenerate"]
    assert not np.array_equal(a["Xq"], b["Xq"])


def test_query_row_depends_only_on_its_own_card_rows():
    Xs = cards(seed=4)
    tr, q = np.arange(0, 30), np.arange(30, 40)
    a = FT.wide_features(Xs, list("abcde"), [1, 3, 4, 5], tr, q)
    Xs2 = [x.copy() for x in Xs]
    for x in Xs2:
        x[35] += 3.0                           # change one query visit only
    b = FT.wide_features(Xs2, list("abcde"), [1, 3, 4, 5], tr, q)
    other = np.arange(len(q)) != 5
    assert np.array_equal(a["Xq"][other], b["Xq"][other]) and not np.array_equal(a["Xq"][5], b["Xq"][5])
    assert a["Xq"].shape == (10, 20)                                    # exactly one row per query visit


def test_overlap_between_training_and_query_rows_is_rejected():
    with pytest.raises(AssertionError):
        FT.wide_features(cards(), list("abcde"), [1, 3, 4, 5], np.arange(0, 30), np.arange(29, 35))
