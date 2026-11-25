import warnings
import numpy as np
import pytest
from wide_fusion.tabcheck import estimator_features, fit_checked


class FakePre:
    class _S:
        num_columns = 252

    feature_schema = _S()

    def __init__(self, idx):
        self.subsample_feature_indices = idx

        class C:
            class preprocess_config:
                max_features_per_estimator = 768
        self.configs = [C(), C()]


class FakeModel:
    def __init__(self, idx, warn=None, support=None, width=252):
        self.idx, self.warn, self.width = idx, warn, width
        self.fit_info_ = {"n_features_in_model": width}
        if support is not None:
            self.support_ = np.asarray(support)

    def fit(self, X, y):
        if self.warn:
            warnings.warn(self.warn, UserWarning)
        pre = FakePre(self.idx)
        pre.feature_schema.num_columns = self.width
        self.model_ = type("M", (), {"ensemble_preprocessor_": pre})()
        return self


def test_feature_check_logs_counts_and_stops_on_subsampling():
    X, y = np.zeros((3, 252)), np.zeros(3)
    info = fit_checked(FakeModel([None, None]), X, y, 2)
    assert info == {"n_columns_offered": 252, "n_degenerate_dropped": 0, "degenerate_dropped": [], "n_features_in_model": 252,
                    "per_estimator_features": [252, 252], "max_features_per_estimator": [768], "n_estimators": 2}
    with pytest.raises(RuntimeError, match="STOP"):
        fit_checked(FakeModel([np.arange(100), None]), X, y, 2)
    with pytest.raises(UserWarning):
        fit_checked(FakeModel([None, None], warn="Running 2 estimators covers at most 100 of 252 features"), X, y, 2)
    with pytest.raises(AssertionError):
        fit_checked(FakeModel([None, None, None]), X, y, 2)
    other = FakeModel([None, None], warn="some unrelated UserWarning")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert fit_checked(other, X, y, 2) is not None      # unrelated warnings are not errors
    assert estimator_features(object()) is None                                       # a stub has nothing to inspect


def test_feature_check_reports_degenerate_drop_separately_from_subsampling():
    X, y = np.zeros((3, 252)), np.zeros(3)
    names = [f"c{i}" for i in range(252)]
    support = [i for i in range(252) if i != 7]                       # the wrapper dropped column 7 (all NaN)
    info = fit_checked(FakeModel([None, None], support=support, width=251), X, y, 2, names=names)
    assert info["n_columns_offered"] == 252 and info["n_features_in_model"] == 251
    assert info["n_degenerate_dropped"] == 1 and info["degenerate_dropped"] == ["c7"]
    assert info["per_estimator_features"] == [251, 251]               # a drop is not subsampling
    with pytest.raises(RuntimeError, match="STOP"):                   # drop + subsampling: still a STOP
        fit_checked(FakeModel([np.arange(100), None], support=support, width=251), X, y, 2, names=names)
    with pytest.raises(AssertionError, match="fitted width"):         # 252 offered, 251 fitted, nothing recorded as dropped
        fit_checked(FakeModel([None, None], width=251), X, y, 2)
