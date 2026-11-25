"""Frozen TabPFN v3.5 (no fine-tuning): DropDegenerate -> TabPFN in-context fit.

The pretrained checkpoint is used as is; ``n_estimators`` is the only non-default setting. NaN goes
straight to TabPFN (no imputation or scaling). CPU only.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion

from .config import REPO_ROOT
from .pipelines import DropDegenerate

CHECKPOINT_NAME = "tabpfn-v3.5-20260909"


def assert_model_cache_disabled() -> None:
    """The built-model cache would hand later fits a mutated model."""
    v = os.environ.get("TABPFN_MODEL_CACHE_SIZE")
    assert v in (None, "", "0"), f"TABPFN_MODEL_CACHE_SIZE must be unset or 0, got {v!r}"


def load_tabpfn_token(env_path: Path | None = None) -> bool:
    """Load TABPFN_TOKEN from the repo .env (never printed). Returns whether it is set."""
    from dotenv import load_dotenv

    load_dotenv(env_path or REPO_ROOT / ".env", override=False)
    return bool(os.environ.get("TABPFN_TOKEN"))


__all__ = ["TabPFNFrozen", "CHECKPOINT_NAME"]


class TabPFNFrozen(BaseEstimator):
    """``random_state=None`` is treated as 0."""

    def __init__(self, task: str = "classification", n_estimators: int = 2,
                 random_state: int | None = None, n_threads: int = 4):
        self.task = task
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_threads = n_threads

    def _seed(self) -> int:
        return 0 if self.random_state is None else int(self.random_state)

    def _transform(self, X) -> np.ndarray:
        return self.drop_.transform(np.asarray(X, dtype=float))

    def fit(self, X, y):
        if self.task not in ("classification", "regression"):
            raise ValueError(f"task must be classification or regression, got {self.task!r}")
        assert_model_cache_disabled()
        load_tabpfn_token()
        torch.set_num_threads(self.n_threads)
        seed = self._seed()
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        if self.task == "classification":
            if not set(np.unique(y)) <= {0, 1}:
                raise ValueError("classification labels must be 0/1")
            y = y.astype(int)
        else:
            y = y.astype(float)
            if np.isnan(y).any():
                raise ValueError("regression target contains NaN")

        self.drop_ = DropDegenerate().fit(X)
        Xt = self._transform(X)
        self.n_features_in_ = X.shape[1]
        self.support_ = np.flatnonzero(self.drop_.support_)

        cls = TabPFNClassifier if self.task == "classification" else TabPFNRegressor
        self.model_ = cls.create_default_for_version(
            ModelVersion.V3_5, device="cpu", n_estimators=self.n_estimators, random_state=seed)
        self.model_.fit(Xt, y)
        path = Path(str(getattr(self.model_, "model_path", ""))).name
        if path and path != "auto" and not path.startswith(CHECKPOINT_NAME):
            raise RuntimeError(f"unexpected TabPFN checkpoint {path!r}, want {CHECKPOINT_NAME}")
        self.checkpoint_ = path or None
        if self.task == "classification":
            self.classes_ = np.asarray(self.model_.classes_)
        self.fit_info_ = {"fit_seconds": time.perf_counter() - t0, "n_estimators": int(self.n_estimators),
                          "n_features_in_model": int(Xt.shape[1]),
                          "checkpoint": self.checkpoint_}
        return self

    @property
    def selected_features_(self) -> list:
        check_is_fitted(self, "support_")
        if hasattr(self, "feature_names_in_"):
            return list(self.feature_names_in_[self.support_])
        return [int(i) for i in self.support_]

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification")
        torch.set_num_threads(self.n_threads)
        return self.model_.predict_proba(self._transform(X))

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        torch.set_num_threads(self.n_threads)
        return np.asarray(self.model_.predict(self._transform(X)))
