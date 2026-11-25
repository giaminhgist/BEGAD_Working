from pathlib import Path
import numpy as np
import pytest
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline

PKG = Path(__file__).resolve().parents[1]


class Stub:
    """Fast deterministic stand-in for TabPFNFrozen (same fit / predict / predict_proba interface)."""

    def __init__(self, task: str, seed: int = 0):
        self.task, self.seed = task, seed

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        assert not np.isnan(y).any()
        est = Ridge(alpha=10.0) if self.task == "regression" else LogisticRegression(C=0.1, max_iter=500)
        self.m_ = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), est).fit(X, y)
        self.n_rows_ = len(X)
        return self

    def predict(self, X):
        return self.m_.predict(np.asarray(X, float))

    def predict_proba(self, X):
        return self.m_.predict_proba(np.asarray(X, float))


def stub_factory(kind: str, n_estimators: int, n_threads: int):
    return lambda seed: Stub(kind, seed)


@pytest.fixture(scope="session")
def cfg():
    return yaml.safe_load((PKG / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def data(cfg):
    from wide_fusion.data import load_data
    return load_data(cfg)


@pytest.fixture(scope="session")
def folds0(data):
    from common.data import merge_folds
    from common.make_folds import load_folds
    return merge_folds(data.cohort, load_folds(0))
