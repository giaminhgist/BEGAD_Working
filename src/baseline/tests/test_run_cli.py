import json
import yaml
import pytest
from baseline import run as brun

CFG = {"feature_sets": ["task1"], "models": [
    {"name": "lr_l2", "fixed": {"l1_ratio": 0.0, "solver": "lbfgs", "max_iter": 2000}, "grid": {"C": [0.01, 1], "class_weight": [None, "balanced"]}},
    {"name": "mmse_cutoff", "cutoff": 28, "missing": "drop"}]}


def test_end_to_end_tiny_run(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(CFG))
    out = tmp_path / "out"
    rc = brun.main(["--config", str(cfg), "--smoke", "--workers", "1", "--n-boot", "30", "--out", str(out)])
    assert rc == 0
    for cell in [out / "lr_l2" / "task1", out / "mmse_cutoff" / "clinical"]:
        for f in ["predictions.csv", "selected_params.json", "metrics.json"]:
            assert (cell / f).exists(), (cell, f)
        m = json.loads((cell / "metrics.json").read_text())
        assert set(m["summary"]) == {"val_auc", "test_auc", "test_acc", "test_sen", "test_spec"}
        assert m["bootstrap"]["B"] == 30
    assert (out / "lr_l2" / "task1" / "inner_oof.csv").exists()
    assert not (out / "mmse_cutoff" / "clinical" / "inner_oof.csv").exists()
    assert json.loads((out / "mmse_cutoff" / "clinical" / "metrics.json").read_text())["summary"]["val_auc"]["mean"] is None
    n = {c: json.loads((c / "metrics.json").read_text()) for c in [out / "lr_l2" / "task1", out / "mmse_cutoff" / "clinical"]}
    a, b = n[out / "lr_l2" / "task1"], n[out / "mmse_cutoff" / "clinical"]
    assert (a["n_visits"], a["n_subjects"], a["n_pos"], a["n_neg"]) == (176, 141, 54, 122)
    assert (b["n_visits"], b["n_pos"], b["n_neg"]) == (175, 53, 122)
    assert not (out / "lr_l2" / "task1" / "_partial").exists()
    md = (out / "results.md").read_text()
    assert "| lr_l2 | task1 |" in md and "| mmse_cutoff | clinical |" in md
    info = json.loads((out / "run_info.json").read_text())
    assert info["status"] == "completed" and info["seeds"] == [0] and info["command"].startswith("python -m baseline.run")
    assert "versions" in info and "code_sha256" in info
    # resume: a second launch in the same directory does not refit finished cells
    assert brun.main(["--config", str(cfg), "--smoke", "--workers", "1", "--n-boot", "30", "--out", str(out)]) == 0


def test_meanprob_without_source_cells_stops_early(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"feature_sets": ["task1"], "models": [
        {"name": "tabpfn_meanprob", "source_model": "tabpfn", "tasks": [1, 3, 4, 5, 7, 9]}]}))
    with pytest.raises(SystemExit, match="needs the cells"):
        brun.main(["--config", str(cfg), "--smoke", "--out", str(tmp_path / "o")])


def test_unknown_model_or_feature_set_rejected(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(CFG))
    with pytest.raises(SystemExit):
        brun.main(["--config", str(cfg), "--models", "nope", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        brun.main(["--config", str(cfg), "--feature-sets", "task2", "--out", str(tmp_path / "o")])
