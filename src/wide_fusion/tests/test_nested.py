import dataclasses
import numpy as np
import pandas as pd
import pytest
from wide_fusion import data as D
from wide_fusion.nested import audit_fit_log, run_member_job
from .conftest import stub_factory


def test_config_matches_the_design(cfg):
    assert cfg["w"] == 0.5 and cfg["clip"] == 1e-6
    assert list(cfg["fused_cells"]) == ["wide_cf_w050"]
    assert cfg["fused_cells"]["wide_cf_w050"] == {"classifier": "wide_cls_e2", "regressor": "wide_reg_C_e2"}
    assert cfg["tabpfn_feature_check"] is True and cfg["targets"]["C"]["min_scores"] == 3
    assert cfg["targets"]["C"]["sign"] == {"mmse": -1, "verbal": -1, "word": -1, "naming": -1, "delayed": -1}
    assert cfg["members"] == {"wide_cls_e2": {"kind": "classification", "layout": "wide", "n_estimators": 2},
                              "wide_reg_C_e2": {"kind": "regression", "layout": "wide", "n_estimators": 2, "target": "C"}}
    assert set(cfg["descriptive_cells"]) == {"kd_meanprob", "kd_meanreg", "kdwide_cls", "kdwide_reg", "rdwide_cls", "rdwide_reg"}
    assert [list(c) for c in cfg["contrasts"]] == [
        ["wide_cf_w050", "ref:cogfusion_w050"], ["kdwide_cls", "kd_meanprob"], ["kdwide_reg", "kd_meanreg"]]
    assert set(cfg["references"]) == {"tabpfn_meanprob", "cogfusion_w050"}


def test_blocks_are_wide_card_blocks_with_the_designed_widths(data, cfg):
    bl = D.build_blocks(data, cfg)
    assert [b.name for b in bl] == ["kd", "reading"] and [len(b.Xs) for b in bl] == [4, 2]
    assert [len(b.cols) for b in bl] == [63, 60]
    assert "gaze_quadrant_prop_bottom_left" not in bl[0].cols and "gaze_quadrant_prop_bottom_left" not in bl[1].cols


@pytest.fixture(scope="module")
def results(data, folds0, cfg):
    return {m: run_member_job(m, cfg["members"][m], 0, 1, data, folds0, cfg, stub_factory) for m in cfg["members"]}


@pytest.mark.parametrize("m", ["wide_cls_e2", "wide_reg_C_e2"])
def test_member_runs_audits_and_has_the_wide_widths(m, results, data, folds0):
    r = results[m]
    audit_fit_log(r["fit_log"], data, folds0)
    n_te = int((folds0["outer_fold"] == 1).sum())
    assert len(r["predictions"]) == n_te and r["predictions"]["y_score"].notna().all()
    assert len(r["inner_oof"]) == len(data.tags) - n_te and r["inner_oof"]["y_score"].notna().all()
    assert len(r["fit_log"]) == 5 * 2 and {f["block"] for f in r["fit_log"]} == {"kd", "reading"}
    assert {f["n_features"] for f in r["fit_log"] if f["block"] == "kd"} == {252}
    assert {f["n_features"] for f in r["fit_log"] if f["block"] == "reading"} == {120}
    rows = r["rows_predictions"]
    assert set(rows["block"]) == {"kd", "reading"} and (rows.groupby(["RealTag", "block"]).size() == 1).all()
    for f in r["fit_log"]:                   # z parameters come from the training visits only
        assert set(f["wide_log"]["param_tags"]) == set(f["train"]) and not set(f["wide_log"]["param_tags"]) & set(f["query"])
    te = r["predictions"].set_index("RealTag")["y_score"]       # the visit score is the mean of its KD and reading scores
    mean = rows.groupby("RealTag")["y_row"].sum() / 2
    assert np.allclose(te.reindex(mean.index).to_numpy(), mean.to_numpy())


def test_regression_rows_without_c_are_not_training_rows(results, data):
    r = results["wide_reg_C_e2"]
    ex = set(r["target_params"]["refit"]["excluded"])
    for f in r["fit_log"]:
        if f["fit"] == "refit":
            assert not (set(f["train"]) & ex)
    none = set(data.tags[(~np.isnan(data.targets["C"].S)).sum(axis=1) < 3])
    assert len(none) <= 1


def test_outer_test_perturbation_leaves_inner_oof_and_params_unchanged(results, data, folds0, cfg):
    te = folds0["outer_fold"].to_numpy() == 1
    X2 = {t: v.copy() for t, v in data.X.items()}
    for t in X2:
        X2[t][te] = 1e3
    S2, y2 = data.targets["C"].S.copy(), data.y.copy()
    S2[te] = 0.0
    y2[te] = 1 - y2[te]
    d2 = dataclasses.replace(data, X=X2, y=y2, targets={"C": dataclasses.replace(data.targets["C"], S=S2)})
    for m in cfg["members"]:
        a, b = results[m], run_member_job(m, cfg["members"][m], 0, 1, d2, folds0, cfg, stub_factory)
        pd.testing.assert_frame_equal(a["inner_oof"].drop(columns="y_true"), b["inner_oof"].drop(columns="y_true"))
        assert a["target_params"] == b["target_params"]


def test_audit_catches_injected_leaks(results, data, folds0):
    base = results["wide_reg_C_e2"]["fit_log"]
    for mutate in ("train", "z_param_tags", "target_param_tags"):
        bad = [dict(f) for f in base]
        f = next(x for x in bad if x["fit"] == "inner2")
        f[mutate] = list(f[mutate]) + [f["query"][0]]
        with pytest.raises(AssertionError):
            audit_fit_log(bad, data, folds0)
    bad = [dict(f) for f in base]
    f = next(x for x in bad if x["fit"] == "inner2")
    f["wide_log"] = {**f["wide_log"], "param_tags": f["wide_log"]["param_tags"] + [f["query"][0]]}
    with pytest.raises(AssertionError):
        audit_fit_log(bad, data, folds0)
    bad = [dict(f) for f in base]
    f = next(x for x in bad if x["fit"] == "refit")
    f["train"] = list(f["train"]) + [data.tags[folds0["outer_fold"].to_numpy() == 1][0]]
    with pytest.raises(AssertionError):
        audit_fit_log(bad, data, folds0)


def test_input_audit_rejects_forbidden_columns_and_score_copies():
    X = np.random.default_rng(0).normal(size=(30, 2))
    with pytest.raises(AssertionError):
        D.audit_inputs(["a", "MMSE"], X, {})
    with pytest.raises(AssertionError):
        D.audit_inputs(["a", "b"], X, {"score": 2 * X[:, 1] + 1})
    D.audit_inputs(["a", "b"], X, {"score": np.arange(30.0)})
