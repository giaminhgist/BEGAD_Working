"""Late fusion: mean of the per-task TabPFN MCI probabilities (no learned weights, no refit).

Built from the saved cells `tabpfn/task{t}` of the same output directory, per (seed, outer fold, visit):
- test score = mean of the per-task outer-test probabilities of the visit (each per-task model was trained on the
  outer-training rows only, so the visit is never in a training set);
- inner-OOF score = mean of the per-task inner-OOF probabilities (the inner folds are shared by the tasks);
- threshold = Youden index of the pooled inner-OOF scores of the (seed, outer fold).
"""
from common.latefusion import KEYS_O, KEYS_P, finish_cell, load_task_cells, mean_prob, members
from . import DerivedModel, register


@register("tabpfn_meanprob")
class TabPfnMeanProb(DerivedModel):
    feature_set = "fusion"

    def requires(self, model_cfg):
        return [(model_cfg["source_model"], f"task{t}") for t in model_cfg["tasks"]]

    def run(self, ctx):
        cfg = ctx.model_cfg
        pred_t, oof_t = load_task_cells(ctx.out_root, cfg["source_model"], cfg["tasks"], ctx.seeds)
        bp, Pp = members(pred_t, KEYS_P)
        bo, Po = members(oof_t, KEYS_O)
        pred, oof = finish_cell(bp.assign(y_score=mean_prob(Pp)), bo.assign(y_score=mean_prob(Po)))
        return {"predictions": pred, "inner_oof": oof,
                "selected": {"rule": "mean of the per-task probabilities; threshold = Youden index on the inner OOF",
                             "source_model": cfg["source_model"], "tasks": cfg["tasks"]}}
