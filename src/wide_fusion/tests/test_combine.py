import numpy as np
import pandas as pd
import pytest
from wide_fusion import combine as CB


def synth(seed=0, n=48):
    rng = np.random.default_rng(seed)
    v = np.arange(n)
    tags = [f"V{i:03d}" for i in v]
    y = (rng.random(n) < 0.4).astype(float)
    y[:8] = [0, 1] * 4
    outer = v % 4
    pred, oof = [], []
    for o in range(4):
        for i in np.flatnonzero(outer == o):
            pred.append({"seed": 0, "outer_fold": o, "RealTag": tags[i], "ParticipantID": f"P{i}", "IsFollowup": 0,
                         "y_true": y[i]})
        tr = np.flatnonzero(outer != o)
        for k, i in enumerate(tr):
            oof.append({"seed": 0, "outer_fold": o, "inner_fold": k % 4, "RealTag": tags[i], "ParticipantID": f"P{i}",
                        "IsFollowup": 0, "y_true": y[i]})
    pred, oof = pd.DataFrame(pred), pd.DataFrame(oof)
    cls = (pred.assign(y_score=rng.uniform(0.05, 0.95, len(pred))), oof.assign(y_score=rng.uniform(0.05, 0.95, len(oof))))
    reg = (pred.assign(y_score=rng.normal(size=len(pred))), oof.assign(y_score=rng.normal(size=len(oof))))
    return cls, reg


def test_fused_score_formula():
    cls, reg = synth()
    (pred, oof), sel = CB.fuse(cls, reg, 0.5, 1e-6)
    o = 2
    lo = CB.logit(cls[1][cls[1].outer_fold == o]["y_score"])
    ro = reg[1][reg[1].outer_fold == o]["y_score"].to_numpy()
    lt = CB.logit(cls[0][cls[0].outer_fold == o]["y_score"])
    rt = reg[0][reg[0].outer_fold == o]["y_score"].to_numpy()
    want = 0.5 * (lt - lo.mean()) / lo.std(ddof=1) + 0.5 * (rt - ro.mean()) / ro.std(ddof=1)
    got = pred[pred.outer_fold == o].set_index("RealTag")["y_score"].reindex(cls[0][cls[0].outer_fold == o]["RealTag"])
    assert np.allclose(got.to_numpy(), want)
    z = sel["0"]["2"]["z_ref_test"]
    assert z["m_l"] == pytest.approx(lo.mean()) and z["s_r"] == pytest.approx(ro.std(ddof=1))


def test_test_row_perturbation_leaves_inner_oof_threshold_and_z_refs_unchanged():
    cls, reg = synth()
    (p1, o1), s1 = CB.fuse(cls, reg, 0.5, 1e-6)
    cls2 = (cls[0].assign(y_score=0.999), cls[1])
    reg2 = (reg[0].assign(y_score=reg[0]["y_score"] * -50), reg[1])
    (p2, o2), s2 = CB.fuse(cls2, reg2, 0.5, 1e-6)
    pd.testing.assert_frame_equal(o1, o2)
    assert np.array_equal(p1["threshold"], p2["threshold"])
    assert s1 == s2
    assert not np.allclose(p1["y_score"], p2["y_score"])


def test_misaligned_members_raise():
    cls, reg = synth()
    with pytest.raises(AssertionError):
        CB.fuse(cls, (reg[0].iloc[1:], reg[1]), 0.5, 1e-6)
    bad = reg[1].copy()
    bad.loc[0, "y_true"] = 1 - bad.loc[0, "y_true"]
    with pytest.raises(AssertionError):
        CB.fuse(cls, (reg[0], bad), 0.5, 1e-6)


def test_audit_references_detects_a_test_visit():
    ref = CB.ZRef(np.arange(4.0), np.arange(4.0), ["a", "b", "c", "d"], [0, 1, 2, 3])
    with pytest.raises(AssertionError):
        CB.audit_references([{"target": "test", "ref": ref, "target_tags": None}], ["a", "b", "c"], ["d"])
    with pytest.raises(AssertionError):
        CB.audit_references([{"target": "inner1", "ref": ref, "target_tags": ("b",)}], ["a", "b", "c", "d"], [])


def test_single_member_cell_uses_score_as_is():
    cls, _ = synth()
    pred, oof = CB.single_cell(cls)
    assert np.array_equal(pred.sort_values(["outer_fold", "RealTag"])["y_score"].to_numpy(),
                          cls[0].sort_values(["outer_fold", "RealTag"])["y_score"].to_numpy())
