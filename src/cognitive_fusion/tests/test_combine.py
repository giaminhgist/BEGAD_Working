import numpy as np
import pandas as pd
import pytest
from cognitive_fusion import combine as C
from common.latefusion import TASKS


def fold_data(n=48, seed=0):
    rng = np.random.default_rng(seed)
    inner = np.repeat(np.arange(4), n // 4)
    tags = np.array([f"v{i}" for i in range(n)])
    return rng.normal(size=n), rng.normal(size=n), inner, tags, rng.normal(size=10), rng.normal(size=10)


def test_w0_is_z_of_logit_and_w1_is_z_of_regressor():
    l, r, inner, tags, lt, rt = fold_data()
    out = C.combine_fold(l, r, inner, tags, lt, rt, [0.0, 0.5, 1.0])
    m, s = l.mean(), l.std(ddof=1)
    assert np.allclose(out[0.0][1], (lt - m) / s)
    m, s = r.mean(), r.std(ddof=1)
    assert np.allclose(out[1.0][1], (rt - m) / s)
    assert np.allclose(out[0.5][1], 0.5 * out[0.0][1] + 0.5 * out[1.0][1])


def test_inner_rows_use_other_inner_folds_only():
    l, r, inner, tags, lt, rt = fold_data()
    out = C.combine_fold(l, r, inner, tags, lt, rt, [0.0])
    m = inner == 2
    assert np.allclose(out[0.0][0][m], (l[m] - l[~m].mean()) / l[~m].std(ddof=1))
    # changing own-fold values changes only the other folds' z of that fold's scores, not this fold's reference
    l2 = l.copy()
    l2[m] += 100
    out2 = C.combine_fold(l2, r, inner, tags, lt, rt, [0.0])
    assert np.allclose(out2[0.0][0][m], (l2[m] - l[~m].mean()) / l[~m].std(ddof=1))


def test_test_scores_do_not_use_test_values_in_the_reference():
    l, r, inner, tags, lt, rt = fold_data()
    a = C.combine_fold(l, r, inner, tags, lt, rt, [0.5])
    b = C.combine_fold(l, r, inner, tags, lt + 50, rt - 50, [0.5])
    assert np.array_equal(a[0.5][0], b[0.5][0])           # inner-OOF scores unchanged by test values


def test_audit_references_detects_contamination():
    l, r, inner, tags, lt, rt = fold_data()
    log = []
    C.combine_fold(l, r, inner, tags, lt, rt, [0.5], log)
    test_tags = [f"t{i}" for i in range(10)]
    C.audit_references(log, tags, test_tags)
    with pytest.raises(AssertionError):
        C.audit_references(log, tags, list(tags[:3]))        # a "test" visit inside the references
    log[0]["ref"].tags = log[0]["ref"].tags + ("zzz",)
    with pytest.raises(AssertionError):
        C.audit_references(log, tags, test_tags)


def test_zero_sd_raises():
    with pytest.raises(ValueError):
        C.mean_sd(np.ones(5))


def test_logit_clip():
    assert np.isfinite(C.logit(np.array([0.0, 1.0]), 1e-6)).all()
    assert C.logit(0.5) == pytest.approx(0.0)


def _frames(rng, tasks=TASKS, with_pred=True):
    pred, oof = {}, {}
    pid = {f"v{i}": f"p{i // 2}" for i in range(40)}
    for t in tasks:
        rows_p, rows_o = [], []
        for o in range(2):
            te = [f"v{i}" for i in range(40) if (i // 2) % 2 == o]
            tr = [f"v{i}" for i in range(40) if (i // 2) % 2 != o]
            for v in te:
                rows_p.append((0, o, v, pid[v], 0, float(int(v[1:]) % 3 == 0), rng.uniform(0.05, 0.95)))
            for k, v in enumerate(tr):
                rows_o.append((0, o, k % 4, v, pid[v], 0, float(int(v[1:]) % 3 == 0), rng.uniform(0.05, 0.95)))
        pred[t] = pd.DataFrame(rows_p, columns=["seed", "outer_fold", "RealTag", "ParticipantID", "IsFollowup", "y_true", "y_score"])
        oof[t] = pd.DataFrame(rows_o, columns=["seed", "outer_fold", "inner_fold", "RealTag", "ParticipantID", "IsFollowup", "y_true", "y_score"])
    return pred, oof


def test_fusion_arms_end_to_end_on_synthetic_cells():
    rng = np.random.default_rng(0)
    pred, oof = _frames(rng)
    rp = {t: d.assign(y_pred=rng.normal(size=len(d))).drop(columns="y_score") for t, d in pred.items()}
    ro = {t: d.assign(y_pred=rng.normal(size=len(d))).drop(columns="y_score") for t, d in oof.items()}
    ws = [0.0, 0.25, 0.5, 0.75, 1.0]
    cells, aux = C.fusion_arms(pred, oof, rp, ro, ws)
    assert list(cells) == ws and [C.arm_name(w) for w in ws] == [f"cogfusion_w{k:03d}" for k in (0, 25, 50, 75, 100)]
    p, o = cells[0.5]
    assert len(p) == 40 and len(o) == 40 and p["RealTag"].is_unique
    assert not any("sel" in C.arm_name(w) for w in ws)                  # no selected-weight arm
    # classifier member = mean of the per-task probabilities
    bt, pt = aux["p_te"]
    manual = pd.concat([pred[t].set_index("RealTag")["y_score"] for t in TASKS], axis=1).mean(axis=1)
    assert np.allclose(pt, manual.loc[bt["RealTag"]].to_numpy(), atol=1e-12)
    # threshold of every fold is the Youden threshold of that fold's inner OOF
    from common.metrics import youden_threshold
    for f, te in p.groupby("outer_fold"):
        tr = o[o.outer_fold == f]
        assert (te["threshold"] == youden_threshold(tr["y_true"], tr["y_score"])).all()


def test_rho_nan_rules():
    assert np.isnan(C.rho([1, 2], [1, 2])) and np.isnan(C.rho([1, 1, 1], [1, 2, 3]))
    assert C.rho([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert np.isnan(C.seed_mean([np.nan, np.nan])) and C.seed_mean([1.0, np.nan, 3.0]) == 2.0
