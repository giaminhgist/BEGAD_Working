"""Frozen TabPFN v3.5 classifier (no fine-tuning, no tuning grid; NaN passed natively)."""
from common.tabpfn_frozen import TabPFNFrozen
from . import EstimatorModel, register


@register("tabpfn")
class TabPfn(EstimatorModel):
    pipeline = "tabpfn"
    score = "predict_proba"

    def build(self, params, fixed, ctx):
        return TabPFNFrozen(task="classification", random_state=ctx.seed, n_threads=ctx.n_threads, **fixed)
