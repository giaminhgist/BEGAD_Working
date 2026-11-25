import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from common.data import load_cohort, merge_folds
from common.latefusion import TASKS
from common.make_folds import load_folds
from common.metrics import youden_threshold
from common.registry import DerivedContext
from baseline.models import REGISTRY
from common.tests.fakes import fake_task_cells as _fake_task_cells, write_source as _write_source


def run_mmse(missing="drop", seeds=(0, 1), cutoff=28):
    cfg = {"name": "mmse_cutoff", "cutoff": cutoff, "missing": missing}
    return REGISTRY["mmse_cutoff"].run(DerivedContext(cfg, None, list(seeds), False))


def test_mmse_missing_visit_is_reported_not_imputed():
    c = load_cohort()
    missing = c.loc[c["mmse"].isna(), "RealTag"].tolist()
    assert missing == ["BEGMCI-010"]
    with pytest.raises(ValueError, match="BEGMCI-010"):
        run_mmse("error")
    with pytest.raises(ValueError):
        run_mmse("impute")


def test_mmse_rule_scores_and_decisions():
    res = run_mmse("drop")
    p = res["predictions"]
    c = load_cohort().set_index("RealTag")
    mm = c.loc[p["RealTag"], "mmse"].to_numpy()
    assert (p["y_score"].to_numpy() == (mm <= 28).astype(float)).all()      # the score is the rule prediction
    assert (p["threshold"] == 0.5).all()
    assert (p["y_score"].to_numpy() == p["y_hat"].to_numpy()).all()
    assert ((p["y_hat"] == 1) == (mm <= 28)).all()            # MCI iff MMSE <= 28; 28 itself is MCI
    assert (p["y_hat"][mm == 28] == 1).all() and (p["y_hat"][mm == 29] == 0).all()
    assert "BEGMCI-010" not in set(p["RealTag"])
    assert res["selected"]["visits_without_mmse"] == ["BEGMCI-010"]


def test_mmse_cell_layout_and_no_validation():
    from baseline.models.mmse_cutoff import MmseCutoff
    res = run_mmse("drop", seeds=(0,))
    p = res["predictions"]
    assert list(p.columns) == ["seed", "outer_fold", "RealTag", "ParticipantID", "IsFollowup", "y_true", "y_score",
                               "threshold", "y_hat"]
    assert res["inner_oof"] is None and MmseCutoff.has_validation is False
    assert len(p) == 175 and p["RealTag"].is_unique
    folds = merge_folds(load_cohort(), load_folds(0)).set_index("RealTag")
    assert (folds.loc[p["RealTag"], "outer_fold"].to_numpy() == p["outer_fold"].to_numpy()).all()
    assert REGISTRY["tabpfn_meanprob"].has_validation


def test_mmse_metrics_have_null_val_auc_and_note(tmp_path):
    from common.bootstrap import load_draws
    from common.evaluate import compute_metrics, write_cell
    from common.report import write_results_md
    res = run_mmse("drop", seeds=(0, 1))
    d = load_draws(B=20)
    m = compute_metrics(res["predictions"], None, d, res["val_note"])
    p = res["predictions"]
    t = np.mean([roc_auc_score(q["y_true"], q["y_score"]) for _, q in p.groupby("seed")])
    assert m["summary"]["test_auc"]["mean"] == pytest.approx(t)
    assert m["summary"]["val_auc"] == {"mean": None, "lo": None, "hi": None}
    assert m["val_auc_note"] == "fixed rule; no model selection" and m["val"] is None
    assert m["test"]["per_seed"][0]["sens"] == pytest.approx(
        ((p[p.seed == 0].y_hat == 1) & (p[p.seed == 0].y_true == 1)).sum() / (p[p.seed == 0].y_true == 1).sum())
    write_cell(tmp_path / "mmse_cutoff" / "clinical", "baseline", "mmse_cutoff", "clinical", p, None, res["selected"], d,
               val_note=res["val_note"], footnote=res["footnote"])
    assert not (tmp_path / "mmse_cutoff" / "clinical" / "inner_oof.csv").exists()
    md = write_results_md(tmp_path, "T").read_text()
    assert "Val AUC" not in md and "validation" not in md.lower() and "Fixed-rule" not in md


def test_meanprob_is_mean_of_task_scores_with_youden_threshold(tmp_path):
    pred, oof = _fake_task_cells()
    _write_source(tmp_path, pred, oof)
    cfg = {"name": "tabpfn_meanprob", "source_model": "tabpfn", "tasks": TASKS}
    spec = REGISTRY["tabpfn_meanprob"]
    assert spec.requires(cfg) == [("tabpfn", f"task{t}") for t in TASKS]
    res = spec.run(DerivedContext(cfg, tmp_path, [0], False))
    p, o = res["predictions"], res["inner_oof"]
    assert len(p) == 176 and len(o) == 3 * 176
    key = p.set_index("RealTag")["y_score"]
    manual = pd.concat([pred[t].set_index("RealTag")["y_score"] for t in TASKS], axis=1).mean(axis=1)
    assert np.allclose(key.sort_index().to_numpy(), manual.sort_index().to_numpy(), atol=1e-12)
    for f, te in p.groupby("outer_fold"):
        tr = o[o.outer_fold == f]
        assert (te["threshold"] == youden_threshold(tr["y_true"], tr["y_score"])).all()
    assert list(p.columns) == ["seed", "outer_fold", "RealTag", "ParticipantID", "IsFollowup", "y_true", "y_score",
                               "threshold", "y_hat"]


def test_meanprob_rejects_leaky_or_missing_sources(tmp_path):
    pred, oof = _fake_task_cells()
    cfg = {"name": "tabpfn_meanprob", "source_model": "tabpfn", "tasks": TASKS}
    with pytest.raises(FileNotFoundError):
        REGISTRY["tabpfn_meanprob"].run(DerivedContext(cfg, tmp_path, [0], False))
    # a test visit also appears in the inner OOF of its own outer fold: must be refused
    bad_oof = {t: d.copy() for t, d in oof.items()}
    leak = pred[1].iloc[[0]].assign(inner_fold=0)
    for t in TASKS:
        bad_oof[t] = pd.concat([bad_oof[t], leak[bad_oof[t].columns.intersection(leak.columns)]], ignore_index=True)
    _write_source(tmp_path / "bad", pred, bad_oof)
    with pytest.raises(AssertionError):
        REGISTRY["tabpfn_meanprob"].run(DerivedContext(cfg, tmp_path / "bad", [0], False))
