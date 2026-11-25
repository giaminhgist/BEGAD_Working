import json
import pandas as pd
import pytest
from common.runinfo import code_hash
from wide_fusion import run as RUN
from .conftest import stub_factory

FUSED = ["wide_cf_w050"]


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(RUN, "model_factory", stub_factory)
    mp.setenv("ARMY_SEED", "3")
    o = tmp_path_factory.mktemp("wf") / "run"
    rc = RUN.main(["--smoke", "--out", str(o), "--n-boot", "30", "--tabpfn-workers", "1", "--skip-pin-check"])
    assert rc == 0
    yield o
    mp.undo()


def test_layout(out, cfg):
    for c in [*FUSED, *cfg["descriptive_cells"]]:
        d = out / c / "fusion"
        for f in ("predictions.csv", "inner_oof.csv", "selected_params.json", "metrics.json"):
            assert (d / f).exists(), (c, f)
        p = pd.read_csv(d / "predictions.csv")
        assert len(p) == 176 and p["y_score"].notna().all()
    for m in cfg["members"]:
        for f in ("predictions.csv", "inner_oof.csv", "rows_predictions.csv", "fit_log.csv", "wide_fit_log.json"):
            assert (out / "members" / m / f).exists()
    for f in ("results.md", "run_info.json", "contrasts.json", "diagnostics.json", "mmse_rules.json"):
        assert (out / f).exists()
    assert not (out / "_partial").exists()
    assert not {p.name for p in out.iterdir() if p.is_dir()} - {"members", *FUSED, *cfg["descriptive_cells"]}


def test_run_info(out):
    info = json.loads((out / "run_info.json").read_text())
    assert info["status"] == "completed" and info["family"] == "wide_fusion" and info["army_seed"] == 3
    assert len(info["code_sha256"]) == 64 and info["code_sha256"] == code_hash("wide_fusion")
    assert info["fused_cells"] == FUSED
    ident = info["saved_member_identity"]
    assert ident["predictions.csv"]["max_abs_diff"] <= 1e-9 and ident["inner_oof.csv"]["max_abs_diff"] <= 1e-9
    assert len(info["saved_member_sha256"]) == 24 and "tabpfn_feature_check" in info
    for blk in (b for m in info["tabpfn_feature_check"].values() for b in m.values()):
        assert blk["all_estimators_see_all_fitted_columns"] is True and blk["n_fits_with_a_subsampled_estimator"] == 0


def test_code_hash_covers_this_package_and_common():
    import hashlib
    from common.config import REPO_ROOT
    h = hashlib.sha256()
    for d in ("common", "wide_fusion"):
        root = REPO_ROOT / "src" / d
        for p in sorted(root.rglob("*.py")):
            rel = p.relative_to(root)
            if "tests" in rel.parts or "__pycache__" in rel.parts:
                continue
            h.update(f"{d}/{rel}".encode())
            h.update(p.read_bytes())
    assert h.hexdigest() == code_hash(*RUN.FAMILY_DIRS)


def test_contrasts(out):
    c = json.loads((out / "contrasts.json").read_text())
    assert list(c) == ["wide_cf_w050 − cogfusion_w050", "kdwide_cls − kd_meanprob", "kdwide_reg − kd_meanreg"]
    assert all("p" not in v for v in c.values()) and {v["n_visits"] for v in c.values()} == {176}


def test_results_md_rules(out):
    t = (out / "results.md").read_text()
    assert "Val AUC" not in t and "p =" not in t
    for w in ("xploratory", "EXPLORATORY", "inflation", "multiplicity"):
        assert w not in t
    assert "## MMSE rules" in t and "mmse_cutoff_lt28" in t and "## Paired contrasts" in t
    for w in ("Role", "Members", "not comparable", "Fused cells", "selected", "primary", "chosen", "Seeds:"):
        assert w not in t
    assert "wide_cf_w050" in t
    assert "val_auc" in json.loads((out / "wide_cf_w050" / "fusion" / "metrics.json").read_text())["summary"]
    row = next(l for l in t.splitlines() if l.startswith("| kdwide_cls |"))
    assert row.count("—") == 3


def test_mmse_rows_in_output(out):
    m = json.loads((out / "mmse_rules.json").read_text())
    assert set(m) == {"mmse_cutoff", "mmse_cutoff_lt28"} and m["mmse_cutoff"]["n_visits"] == 175


def test_diagnostics_name_the_target_c(out):
    d = json.loads((out / "diagnostics.json").read_text())
    r = d["inner_spearman_rbar_target"]
    assert set(r) == {"wide_reg_C_e2"} and r["wide_reg_C_e2"]["target"] == "C" and "vs C " in r["wide_reg_C_e2"]["note"]


def test_resume_refuses_partials_from_another_config(tmp_path):
    import pickle
    o = tmp_path / "r"
    p = RUN.partial_file(o, "wide_cls_e2", 0, 0)
    p.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"fingerprint": "other"}, open(p, "wb"))
    with pytest.raises(SystemExit):
        RUN.main(["--smoke", "--out", str(o), "--n-boot", "30", "--tabpfn-workers", "1", "--skip-pin-check"])


def test_subsampling_stop_halts_the_run_and_later_jobs_do_not_run(tmp_path, monkeypatch, cfg):
    from wide_fusion.tabcheck import SubsamplingStop
    ran = []

    def fake(key, payload):
        ran.append(key)
        if len(ran) == 2:
            raise SubsamplingStop("TabPFN feature subsampling ... STOP")
        return key, 0.0

    monkeypatch.setattr(RUN, "execute", fake)
    fails = RUN.run_jobs(cfg, tmp_path, [0], True, False, 1, 1, lambda m: None)
    assert len(ran) == 2 and len(fails) == 1 and "STOP" in fails[0]
    assert len(RUN.job_order(cfg)) * 4 > 2


def _stop_job(key, payload):
    from wide_fusion.tabcheck import SubsamplingStop
    if payload == "stop":
        raise SubsamplingStop("STOP")
    import time
    time.sleep(payload)
    return key, 0.0


def test_failfast_cancels_pending_futures_with_workers():
    args = [("a", "stop")] + [(f"b{i}", 1.0) for i in range(12)]
    msgs = []
    fails = RUN.run_failfast(_stop_job, args, 2, 1, msgs.append)
    assert any("STOP" in f for f in fails)
    assert sum(m.startswith("done") for m in msgs) < 12


def test_main_exits_nonzero_on_stop(tmp_path, monkeypatch):
    from wide_fusion.tabcheck import SubsamplingStop
    monkeypatch.setattr(RUN, "model_factory", stub_factory)

    def boom(key, payload):
        raise SubsamplingStop("STOP")
    monkeypatch.setattr(RUN, "execute", boom)
    with pytest.raises(SystemExit):
        RUN.main(["--smoke", "--out", str(tmp_path / "r"), "--n-boot", "30", "--tabpfn-workers", "1", "--skip-pin-check"])
    assert json.loads((tmp_path / "r" / "run_info.json").read_text())["status"] == "failed"
