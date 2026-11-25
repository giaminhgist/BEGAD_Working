import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from common import bootstrap, report
from common.evaluate import compute_metrics, write_cell
from common.metrics import auc_score, threshold_metrics, youden_threshold
from common.runinfo import command_line, output_dir_name, write_run_info
import datetime as dt


def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    s = rng.integers(0, 6, 60).astype(float)
    assert auc_score(y, s) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(auc_score(np.zeros(5), np.arange(5)))


def test_threshold_metrics_definitions():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    yh = np.array([1, 1, 0, 0, 0, 1, 0])
    m = threshold_metrics(y, yh)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (2, 1, 3, 1)
    assert m["acc"] == pytest.approx(5 / 7) and m["sens"] == pytest.approx(2 / 3) and m["spec"] == pytest.approx(3 / 4)


def test_youden_picks_best_cut():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    thr = youden_threshold(y, s)
    assert 0.3 < thr <= 0.6
    assert ((s >= thr).astype(int) == y).all()
    with pytest.raises(ValueError):
        youden_threshold(np.ones(3), np.arange(3))


def _synthetic(draws, seeds=(0, 1)):
    rng = np.random.default_rng(1)
    subj = draws["subjects"]
    rows = []
    strat = dict(zip(subj, draws["stratum"]))
    for s in seeds:
        order = rng.permutation(len(subj))
        for k, i in enumerate(order):
            for v in range(1 + (i % 2)):
                y = strat[subj[i]]
                rows.append({"seed": s, "outer_fold": k % 4, "RealTag": f"{subj[i]}_{v}", "ParticipantID": subj[i],
                             "IsFollowup": 0, "y_true": float(y), "y_score": float(rng.normal(y * 0.8, 1))})
    p = pd.DataFrame(rows)
    p["threshold"] = 0.4
    p["y_hat"] = (p["y_score"] >= p["threshold"]).astype(int)
    o = p.drop(columns=["threshold", "y_hat"]).assign(inner_fold=lambda d: d.index % 4)
    return p, o


def test_compute_metrics_summary_and_ci():
    d = bootstrap.load_draws(B=100)
    p, o = _synthetic(d)
    m = compute_metrics(p, o, d)
    for k in ["val_auc", "test_auc", "test_acc", "test_sen", "test_spec"]:
        s = m["summary"][k]
        assert set(s) == {"mean", "lo", "hi"} and s["lo"] <= s["hi"]
    # test mean = mean over seeds of the pooled-seed AUC
    manual = np.mean([roc_auc_score(q["y_true"], q["y_score"]) for _, q in p.groupby("seed")])
    assert m["test"]["mean"]["auc"] == pytest.approx(manual)
    # val = mean over outer folds of the per-fold AUC, then over seeds
    val = np.mean([np.mean([roc_auc_score(q["y_true"], q["y_score"]) for _, q in v.groupby("outer_fold")])
                   for _, v in o.groupby("seed")])
    assert m["val"]["mean"]["auc"] == pytest.approx(val)
    assert m["bootstrap"]["B"] == 100


def test_write_cell_files_and_report(tmp_path):
    d = bootstrap.load_draws(B=50)
    p, o = _synthetic(d)
    write_cell(tmp_path / "m1" / "fusion", "baseline", "m1", "fusion", p, o, {"a": 1}, d)
    write_cell(tmp_path / "m2" / "task1", "baseline", "m2", "task1", p, o, {"a": 1}, d)
    for f in ["predictions.csv", "inner_oof.csv", "selected_params.json", "metrics.json"]:
        assert (tmp_path / "m1" / "fusion" / f).exists()
    md = report.write_results_md(tmp_path, "T", ["m2", "m1"], ["task1", "fusion"]).read_text()
    rows = [l for l in md.splitlines() if l.startswith("| m")]
    assert rows[0].startswith("| m2 | task1") and rows[1].startswith("| m1 | fusion")
    assert "| Model | Feature set | Test AUC [95% CI] | Test Acc [95% CI] | Test Sen [95% CI] | Test Spec [95% CI] |" in md
    assert rows[0].count("[") == 4 and "Val AUC" not in md
    json.loads((tmp_path / "m1" / "fusion" / "metrics.json").read_text())


def test_output_dir_name_and_run_info(tmp_path):
    now = dt.datetime(2026, 11, 24, 9, 5)
    assert output_dir_name("baseline", now=now) == "baseline_2026_11_24_0905"
    assert output_dir_name("baseline", smoke=True, now=now) == "smoke_2026_11_24_0905/baseline"
    info = write_run_info(tmp_path, "baseline", "baseline.run", ["--out", "/" + "abs/elsewhere/out"], "completed", [0],
                          False, now)
    text = (tmp_path / "run_info.json").read_text()
    assert "abs/elsewhere" not in text and "--out out" in info["command"]
    assert {"command", "seeds", "versions", "code_sha256"} <= set(info)
    assert command_line("baseline.run", ["--smoke"]) == "python -m baseline.run --smoke"


def test_smoke_cognitive_fusion_goes_next_to_smoke_baseline(tmp_path):
    from common.cli import resolve_out
    base = tmp_path / "smoke_2026_11_24_0905" / "baseline"
    assert resolve_out(None, "cognitive_fusion", True, base) == base.parent / "cognitive_fusion"
    assert resolve_out(str(tmp_path / "x"), "cognitive_fusion", True, base) == tmp_path / "x"
    assert resolve_out(None, "cognitive_fusion", False, base).name.startswith("cognitive_fusion_")


def test_regenerate_results_without_refitting(tmp_path):
    import hashlib
    from common.report import regenerate
    d = bootstrap.load_draws(B=30)
    p, o = _synthetic(d)
    run = tmp_path / "smoke_x" / "baseline"
    write_cell(run / "m1" / "fusion", "baseline", "m1", "fusion", p, o, {}, d)
    write_cell(run / "rule" / "clinical", "baseline", "rule", "clinical", p.iloc[:-3], None, {}, d, val_note="fixed rule")
    (run / "run_info.json").write_text(json.dumps({"family": "baseline", "models": ["m1", "rule"]}))
    before = {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in run.rglob("*") if f.is_file()}
    out = regenerate(tmp_path / "smoke_x")                      # a parent folder is searched one level down
    assert [x.name for x in out] == ["results.md"]
    after = {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in run.rglob("*") if f.is_file() and f.name != "results.md"}
    assert all(before[f] == h for f, h in after.items())
    md = (run / "results.md").read_text()
    assert "Val AUC" not in md and "validation" not in md.lower() and "Fixed-rule" not in md
    assert "instead of" not in md and md.splitlines()[2] == "Mean over 2 seeds, 95% bootstrap CI (B = 30)."
    (run / "m1" / "fusion" / "metrics.json").unlink()
    (run / "results.md").unlink()
    regenerate(run)
    assert (run / "results.md").read_text().startswith("# ")
