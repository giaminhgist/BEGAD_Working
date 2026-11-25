"""Feed-forward neural network (scikit-learn MLP). class_weight "balanced" = sample weight n / (2 n_c)."""
from sklearn.neural_network import MLPClassifier
from common.pipelines import BalancedSampleWeight
from . import EstimatorModel, register


@register("fnn")
class Fnn(EstimatorModel):
    pipeline = "scaled"
    score = "predict_proba"

    def build(self, params, fixed, ctx):
        p = dict(params)
        class_weight = p.pop("class_weight", None)
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        model = MLPClassifier(**fixed, **p, random_state=ctx.seed)
        return BalancedSampleWeight(model) if class_weight == "balanced" else model
