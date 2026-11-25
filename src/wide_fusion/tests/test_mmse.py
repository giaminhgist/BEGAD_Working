import numpy as np
import pandas as pd
import pytest
from common.bootstrap import bootstrap_thetas, load_draws
from common.config import REPO_ROOT
from common.data import load_cohort
from common.evaluate import ci_metric_fn
from wide_fusion import mmse as MM

BASE = REPO_ROOT / "output" / "baseline_2026_11_24_1207"
SAVED = BASE / "mmse_cutoff" / "clinical" / "predictions.csv"
needs_baseline = pytest.mark.skipif(not SAVED.exists(), reason="baseline output missing")


def test_auc01_equals_balanced_accuracy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    h = rng.integers(0, 2, 200)
    m = MM.point_metrics(y, h)
    assert m["auc01"] == pytest.approx((m["sens"] + m["spec"]) / 2)


def test_toy_example():
    mmse = np.array([30, 29, 27, 28, 26, 25])
    y = np.array([0, 0, 0, 1, 1, 1])
    assert MM.point_metrics(y, mmse <= 28)["auc01"] == pytest.approx(0.8333, abs=1e-3)
    assert MM.point_metrics(y, mmse <= 27)["auc01"] == pytest.approx(0.6667, abs=1e-3)


@needs_baseline
def test_rules_use_the_saved_rows_and_the_cohort_mmse():
    saved = pd.read_csv(SAVED)
    mmse = load_cohort().set_index("RealTag")["mmse"]
    fr = MM.rule_frames(saved, mmse)
    assert (fr["mmse_cutoff"]["y_hat"].to_numpy() == saved["y_hat"].to_numpy()).all()
    v = mmse.reindex(saved["RealTag"]).to_numpy(float)
    assert np.array_equal(fr["mmse_cutoff_lt28"]["y_hat"].to_numpy(), (v < 28).astype(int))     # scores are integers: < 28 is <= 27
    assert np.array_equal(v, np.round(v)) and len(fr["mmse_cutoff"]) == len(saved)
    with pytest.raises(AssertionError):                                         # a saved score that is not the rule prediction is refused
        MM.rule_frames(saved.assign(y_score=1 - saved["y_score"]), mmse)
    with pytest.raises(AssertionError):                                         # a saved decision that is not the rule is refused
        MM.rule_frames(saved.assign(y_hat=1 - saved["y_hat"]), mmse)


@needs_baseline
def test_vectorised_bootstrap_matches_official_aggregation():
    draws = load_draws(B=30)
    pred = pd.read_csv(SAVED)
    sidx = {s: i for i, s in enumerate(draws["subjects"])}
    groups = [[{"sidx": g["ParticipantID"].map(sidx).to_numpy(), "y": g["y_true"].to_numpy(),
                "score": g["y_hat"].to_numpy().astype(float), "yhat": g["y_hat"].to_numpy()}]
              for _, g in pred.groupby("seed")]
    off = bootstrap_thetas(groups, draws["counts"], ci_metric_fn, ["auc", "sens", "spec"])
    mine = MM.boot_thetas(pred, draws)
    for a, b in [("auc", "auc01"), ("sens", "sens"), ("spec", "spec")]:
        np.testing.assert_allclose(off[a], mine[b], atol=1e-12)


@needs_baseline
@pytest.mark.slow
def test_full_run_values_of_both_rules():
    rows = MM.mmse_rows("output/baseline_2026_11_24_1207", list(range(10)) if False else sorted(pd.read_csv(SAVED).seed.unique()),
                        load_draws())
    for name, (m, lo, hi) in {"mmse_cutoff": (0.693, 0.619, 0.762), "mmse_cutoff_lt28": (0.666, 0.580, 0.751)}.items():
        r = rows[name]["auc01"]
        assert (round(r["mean"], 3), round(r["lo"], 3), round(r["hi"], 3)) == (m, lo, hi), name
    assert rows["mmse_cutoff"]["n_visits"] == 175


def test_mmse_module_reads_nothing_from_tmp():
    import inspect
    src = inspect.getsource(MM)
    assert "tmp/" not in src and "import rescore" not in src and "from rescore" not in src
