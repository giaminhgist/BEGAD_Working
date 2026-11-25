"""Random forest (NaN handled natively)."""
from sklearn.ensemble import RandomForestClassifier
from . import EstimatorModel, register


@register("rf")
class Rf(EstimatorModel):
    pipeline = "native"
    score = "predict_proba"

    def build(self, params, fixed, ctx):
        return RandomForestClassifier(**fixed, **params, random_state=ctx.seed)
