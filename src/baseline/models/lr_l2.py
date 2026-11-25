"""L2-regularised logistic regression (median impute, standardise)."""
from sklearn.linear_model import LogisticRegression
from . import EstimatorModel, register


@register("lr_l2")
class LrL2(EstimatorModel):
    pipeline = "scaled"
    score = "predict_proba"

    def build(self, params, fixed, ctx):
        return LogisticRegression(**fixed, **params)
