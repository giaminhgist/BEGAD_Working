"""Tiny registry used to test the nested-CV runner without the baseline package."""
from sklearn.linear_model import LogisticRegression
from common.registry import EstimatorModel, Registry

REGISTRY = Registry(__name__)


@REGISTRY.register("stub_lr")
class StubLr(EstimatorModel):
    pipeline = "scaled"
    score = "predict_proba"

    def build(self, params, fixed, ctx):
        return LogisticRegression(**fixed, **params)
