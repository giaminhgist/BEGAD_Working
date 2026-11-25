"""Clinical baseline: predict MCI if MMSE <= cutoff (28), else CN. No training.

Rule source: Atiya et al. 2026, Alzheimers Dement (Amst) 18(1):e70291, doi:10.1002/dad2.70291 (ADNI; MMSE cutoff 28
for CN vs MCI).

How the cell is filled:
- score = the rule prediction (1 if MMSE <= cutoff, else 0; higher = more likely MCI); the decision threshold is
  0.5, so y_hat = score = 1 iff MMSE <= cutoff. AUC (ties count 1/2), Acc, Sen and Spec all come from this
  fixed prediction;
- predictions.csv: the outer-test rows of every seed and outer fold, as for the other models;
- no inner_oof.csv and no Val AUC: the rule has no parameters and no model selection, so there is nothing to validate
  (`has_validation = False`; metrics.json reports val_auc as null with a note).
MMSE comes from the COGDATA column `MMSE` of the same visit (join PotNumero = RealTag).
"""
import pandas as pd
from common.data import load_cohort, merge_folds
from common.make_folds import load_folds
from . import DerivedModel, register

ID_COLS = ["RealTag", "ParticipantID", "IsFollowup"]


@register("mmse_cutoff")
class MmseCutoff(DerivedModel):
    feature_set = "clinical"
    has_validation = False

    def run(self, ctx):
        cfg = ctx.model_cfg
        cutoff, policy = float(cfg["cutoff"]), cfg.get("missing", "error")
        cohort = load_cohort()
        absent = cohort.loc[cohort["mmse"].isna(), "RealTag"].tolist()
        if absent and policy == "error":
            raise ValueError(f"MMSE is missing for {len(absent)} visit(s): {absent}. Nothing is imputed; set "
                             "`missing: drop` in the config to leave these visits out of this model's cell")
        if policy not in ("error", "drop"):
            raise ValueError(f"missing must be 'error' or 'drop', got {policy!r}")
        has_mmse = cohort["mmse"].notna().to_numpy()
        pred_rows = []
        for seed in ctx.seeds:
            folds = merge_folds(cohort, load_folds(seed))
            outer = folds["outer_fold"].to_numpy()
            for o in range(4):
                part = cohort.loc[(outer == o) & has_mmse]
                d = part[ID_COLS].copy()
                d.insert(0, "outer_fold", o)
                d.insert(0, "seed", seed)
                d["y_true"] = part["label"].astype(float).to_numpy()
                d["y_score"] = (part["mmse"].to_numpy(dtype=float) <= cutoff).astype(float)   # the rule prediction
                d["threshold"] = 0.5
                d["y_hat"] = (d["y_score"] >= 0.5).astype(int)
                pred_rows.append(d)
        pred = pd.concat(pred_rows, ignore_index=True)
        return {"predictions": pred, "inner_oof": None,
                "val_note": "fixed rule; no model selection",
                "footnote": f"Fixed clinical rule (MMSE \u2264 {cutoff:g} \u2192 MCI); no model selection, so no validation AUC.",
                "selected": {"rule": f"MCI if MMSE <= {cutoff:g}", "cutoff": cutoff, "score": "rule prediction (1 if MMSE <= cutoff, else 0)",
                             "visits_without_mmse": absent if policy == "drop" else []}}
