import json
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from cognitive_fusion import run as crun
from common.tests.fakes import write_baseline_dir


def stub_factory(cfg):
    return lambda seed: make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=10.0))


def test_end_to_end_with_stub_regressor(tmp_path, monkeypatch):
    monkeypatch.setattr(crun, "make_tabpfn", stub_factory)
    base = write_baseline_dir(tmp_path / "base")
    out = tmp_path / "out"
    rc = crun.main(["--baseline-out", str(base), "--smoke", "--tabpfn-workers", "1", "--n-boot", "30", "--skip-pin-check",
                    "--out", str(out)])
    assert rc == 0
    arms = ["cogfusion_w000", "cogfusion_w025", "cogfusion_w050", "cogfusion_w075", "cogfusion_w100"]
    assert sorted(p.name for p in out.iterdir() if (p / "fusion").exists()) == arms
    for a in arms:
        for f in ["predictions.csv", "inner_oof.csv", "selected_params.json", "metrics.json"]:
            assert (out / a / "fusion" / f).exists()
        assert len(pd.read_csv(out / a / "fusion" / "predictions.csv")) == 176
    assert not (out / "cogfusion_wsel").exists()
    md = (out / "results.md").read_text()
    assert md.count("| cogfusion_w") == 5 and "Val AUC" not in md and "MMSE" not in md and "Fixed-rule" not in md
    info = json.loads((out / "run_info.json").read_text())
    assert info["status"] == "completed" and info["primary"] == "cogfusion_w050"
    assert "baseline_out" in info and not info["baseline_out"].startswith("/")
    assert (out / "regression" / "task1" / "predictions.csv").exists() and not (out / "_partial").exists()


def test_refuses_unfinished_or_smoke_baseline_for_full_run(tmp_path):
    base = write_baseline_dir(tmp_path / "base")
    with pytest.raises(SystemExit, match="smoke"):
        crun.main(["--baseline-out", str(base), "--skip-pin-check", "--out", str(tmp_path / "o")])
    info = json.loads((base / "run_info.json").read_text())
    (base / "run_info.json").write_text(json.dumps({**info, "status": "failed"}))
    with pytest.raises(SystemExit, match="completed"):
        crun.main(["--baseline-out", str(base), "--smoke", "--skip-pin-check", "--out", str(tmp_path / "o")])


def test_missing_baseline_out_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="baseline"):
        crun.main(["--smoke", "--out", str(tmp_path / "o")])


def test_member_must_match_reference_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(crun, "make_tabpfn", stub_factory)
    base = write_baseline_dir(tmp_path / "base")
    cell = base / "tabpfn_meanprob" / "fusion"
    cell.mkdir(parents=True)
    pred = pd.read_csv(base / "tabpfn" / "task1" / "predictions.csv").assign(y_score=0.5)
    pred.to_csv(cell / "predictions.csv", index=False)
    pd.read_csv(base / "tabpfn" / "task1" / "inner_oof.csv").assign(y_score=0.5).to_csv(cell / "inner_oof.csv", index=False)
    with pytest.raises(AssertionError, match="differs"):
        crun.main(["--baseline-out", str(base), "--smoke", "--tabpfn-workers", "1", "--n-boot", "20", "--skip-pin-check",
                   "--out", str(tmp_path / "out")])
