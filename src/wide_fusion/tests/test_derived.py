import numpy as np
import pandas as pd
import pytest
from common.config import REPO_ROOT
from wide_fusion import derived as DV

KEYS = ["seed", "outer_fold", "RealTag"]
KINDS = {"cls": "wide_cls_e2", "reg": "wide_reg_C_e2"}
CFG = {"blocks": {"kd": [1, 3, 4, 5], "reading": [7, 9]}}


def synth_saved(n_seeds=1, n=12):
    tasks = [1, 3, 4, 5, 7, 9]
    rng = np.random.default_rng(0)
    rows = [(s, o, f"V{i}", f"P{i}", 0, float(i % 2)) for s in range(n_seeds) for o in range(2) for i in range(n)]
    bt = pd.DataFrame(rows, columns=["seed", "outer_fold", "RealTag", "ParticipantID", "IsFollowup", "y_true"])
    bo = pd.concat([bt.assign(inner_fold=k) for k in range(2)], ignore_index=True)
    return {"bt": bt, "bo": bo, "Pt": rng.random((len(bt), 6)), "Po": rng.random((len(bo), 6)),
            "Rt": rng.normal(size=(len(bt), 6)), "Ro": rng.normal(size=(len(bo), 6)), "tasks": tasks}


def synth_rows(saved, seed=1):
    rng = np.random.default_rng(seed)
    bt, bo = saved["bt"], saved["bo"]
    mk = lambda b: pd.concat([b[[c for c in b.columns if c in ("seed", "outer_fold", "inner_fold", "RealTag")]].assign(  # noqa: E731
        task=-1, block=name, y_row=rng.random(len(b))) for name in ("kd", "reading")], ignore_index=True)
    return mk(bt), mk(bo)


def test_kd_means_and_block_scores_are_the_stated_means():
    saved = synth_saved()
    mr = {m: synth_rows(saved, i) for i, m in enumerate(KINDS.values())}
    d = DV.build_derived(CFG, saved, mr, KINDS)
    P = saved["Pt"]
    assert np.allclose(d["kd_meanprob"][0]["y_score"], P[:, :4].mean(axis=1))
    assert np.allclose(d["kd_meanreg"][1]["y_score"], saved["Ro"][:, :4].mean(axis=1))
    rp, ro = mr["wide_cls_e2"]
    kdw = rp[rp.block == "kd"].set_index(KEYS)["y_row"].reindex(pd.MultiIndex.from_frame(saved["bt"][KEYS])).to_numpy()
    assert np.array_equal(d["kdwide_cls"][0]["y_score"].to_numpy(), kdw)
    assert set(d) == {"kd_meanprob", "kd_meanreg", "kdwide_cls", "rdwide_cls", "kdwide_reg", "rdwide_reg"}


def test_key_identity_asserts_missing_or_extra_rows():
    saved = synth_saved()
    bt = saved["bt"]
    DV.assert_same_keys({"a": bt, "b": bt.sample(frac=1, random_state=0)}, KEYS)
    with pytest.raises(AssertionError):
        DV.assert_same_keys({"a": bt, "b": bt.iloc[1:]}, KEYS)
    with pytest.raises(AssertionError):
        DV.assert_same_keys({"a": bt, "b": pd.concat([bt, bt.iloc[:1]])}, KEYS)
    with pytest.raises(AssertionError):
        DV.align(bt.iloc[1:].assign(y_pred=1.0), bt, KEYS, "y_pred")
    with pytest.raises(AssertionError):
        DV.align(bt.assign(y_pred=1.0, ParticipantID="X"), bt, KEYS, "y_pred")


def test_a_member_missing_a_row_stops_the_build():
    saved = synth_saved()
    mr = {m: synth_rows(saved, i) for i, m in enumerate(KINDS.values())}
    rp, ro = mr["wide_reg_C_e2"]
    mr["wide_reg_C_e2"] = (rp.iloc[1:], ro)
    with pytest.raises(AssertionError):
        DV.build_derived(CFG, saved, mr, KINDS)


BASE = REPO_ROOT / "output" / "baseline_2026_11_24_1207"
CF = REPO_ROOT / "output" / "cognitive_fusion_2026_11_24_1305"


@pytest.mark.skipif(not (BASE / "tabpfn_meanprob").exists() or not (CF / "cogfusion_w050").exists(), reason="finished runs missing")
def test_saved_members_reproduce_the_reference_cells(cfg):
    seeds = [0, 1]
    saved = DV.load_saved(cfg, seeds)
    out = DV.identity_check_saved(saved, cfg, seeds, cfg["w"], cfg["clip"])
    assert out["predictions.csv"]["max_abs_diff"] <= 1e-9 and out["inner_oof.csv"]["max_abs_diff"] <= 1e-9
    assert out["predictions.csv"]["rows"] == 176 * 2
    bad = dict(saved, Pt=saved["Pt"] + 1e-6)
    with pytest.raises(AssertionError):
        DV.identity_check_saved(bad, cfg, seeds, cfg["w"], cfg["clip"])
