"""Linear SVM (squared hinge); the score is the signed distance to the hyperplane."""
from sklearn.svm import LinearSVC
from . import EstimatorModel, register


@register("linear_svm")
class LinearSvm(EstimatorModel):
    pipeline = "scaled"
    score = "decision_function"

    def build(self, params, fixed, ctx):
        return LinearSVC(**fixed, **params, random_state=ctx.seed)
