from pathlib import Path
import numpy as np
import pytest
from common.config import expand_grid, estimator_fixed, load_family_config
from common.pipelines import BalancedSampleWeight, BuildContext, wrap_pipeline
from common.registry import DerivedModel, EstimatorModel
from baseline.models import REGISTRY

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def test_config_and_model_files_agree():
    cfg = load_family_config(CONFIG)
    names = {m["name"] for m in cfg["models"]}
    files = {p.stem for p in MODELS_DIR.glob("*.py") if p.stem != "__init__"}
    assert names == files == set(REGISTRY.names())
    assert len(cfg["models"]) == 7


def test_expected_models_and_kinds():
    assert {n for n in REGISTRY.names() if REGISTRY[n].kind == "estimator"} == {"lr_l2", "linear_svm", "rf", "fnn", "tabpfn"}
    assert {n for n in REGISTRY.names() if REGISTRY[n].kind == "derived"} == {"tabpfn_meanprob", "mmse_cutoff"}
    assert REGISTRY["tabpfn"].heavy and not REGISTRY["lr_l2"].heavy
    assert REGISTRY["linear_svm"].score == "decision_function"


def test_grid_sizes_and_smoke_grids():
    cfg = {m["name"]: m for m in load_family_config(CONFIG)["models"]}
    full = {n: len(expand_grid(cfg[n], False)) for n in ["lr_l2", "linear_svm", "rf", "fnn", "tabpfn"]}
    smoke = {n: len(expand_grid(cfg[n], True)) for n in ["lr_l2", "linear_svm", "rf", "fnn", "tabpfn"]}
    assert full == {"lr_l2": 12, "linear_svm": 12, "rf": 12, "fnn": 16, "tabpfn": 1}
    assert smoke == {"lr_l2": 4, "linear_svm": 4, "rf": 8, "fnn": 8, "tabpfn": 1}
    g = expand_grid(cfg["lr_l2"], False)
    assert g[0] == {"C": 0.001, "class_weight": None} and g[1] == {"C": 0.001, "class_weight": "balanced"}
    assert expand_grid(cfg["lr_l2"], True)[-1] == {"C": 100, "class_weight": "balanced"}
    assert estimator_fixed(cfg["rf"], False)["n_estimators"] == 500 and estimator_fixed(cfg["rf"], True)["n_estimators"] == 100
    assert cfg["tabpfn"]["fixed"] == {"n_estimators": 2}


@pytest.mark.parametrize("name", ["lr_l2", "linear_svm", "rf", "fnn"])
def test_sklearn_models_fit_and_score_with_nan(name):
    cfg = {m["name"]: m for m in load_family_config(CONFIG)["models"]}[name]
    spec = REGISTRY[name]
    params = expand_grid(cfg, True)[-1]
    fixed = estimator_fixed(cfg, True)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 6))
    y = (X[:, 0] + rng.normal(scale=0.5, size=60) > 0).astype(int)
    X[3, 2] = np.nan
    X[:, 5] = 1.0                       # degenerate column is dropped
    model = wrap_pipeline(spec.build(dict(params), fixed, BuildContext(seed=0, smoke=True)), spec.pipeline)
    model.fit(X, y)
    s = model.predict_proba(X)[:, 1] if spec.score == "predict_proba" else model.decision_function(X)
    assert s.shape == (60,) and np.isfinite(s).all()
    assert np.corrcoef(s, y)[0, 1] > 0.3


def test_fnn_balanced_wraps_sample_weight():
    cfg = {m["name"]: m for m in load_family_config(CONFIG)["models"]}["fnn"]
    est = REGISTRY["fnn"].build({"alpha": 1, "hidden_layer_sizes": [32], "class_weight": "balanced"},
                                estimator_fixed(cfg, True), BuildContext(seed=0))
    assert isinstance(est, BalancedSampleWeight)
    est = REGISTRY["fnn"].build({"alpha": 1, "hidden_layer_sizes": [64, 32], "class_weight": None},
                                estimator_fixed(cfg, True), BuildContext(seed=0))
    assert est.hidden_layer_sizes == (64, 32)


def test_balanced_sample_weight_equals_balanced_class_weight():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = (X[:, 0] + rng.normal(size=50) > 1).astype(int)
    a = BalancedSampleWeight(LogisticRegression()).fit(X, y).estimator_.coef_
    b = LogisticRegression(class_weight="balanced").fit(X, y).coef_
    assert np.allclose(a, b)


def test_registering_twice_fails():
    with pytest.raises(ValueError):
        REGISTRY.register("lr_l2")(type("Dup", (EstimatorModel,), {}))


@pytest.mark.slow
def test_tabpfn_frozen_classifier_and_regressor_small_fit():
    from common.tabpfn_frozen import TabPFNFrozen
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    X[2, 1] = np.nan
    y = (X[:, 0] > 0).astype(int)
    clf = TabPFNFrozen(task="classification", n_estimators=2, random_state=0, n_threads=2).fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    assert p.shape == (40,) and ((p >= 0) & (p <= 1)).all() and np.corrcoef(p, y)[0, 1] > 0.5
    reg = TabPFNFrozen(task="regression", n_estimators=2, random_state=0, n_threads=2).fit(X, X[:, 0])
    assert np.corrcoef(reg.predict(X), X[:, 0])[0, 1] > 0.8
    assert clf.checkpoint_.startswith("tabpfn-v3.5")
