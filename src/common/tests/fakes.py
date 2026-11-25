"""Fake baseline cells built on the real cohort and folds (random scores), for tests that need per-task outputs."""
import numpy as np
import pandas as pd
from common.data import load_cohort, merge_folds
from common.latefusion import TASKS
from common.make_folds import load_folds


def fake_task_cells(seed=0, noise=0.1):
    """Per-task cells built on the real folds with random scores (shape of the baseline tabpfn cells)."""
    c = load_cohort()
    folds = merge_folds(c, load_folds(seed))
    rng = np.random.default_rng(5)
    pred, oof = {}, {}
    for t in TASKS:
        pr, oo = [], []
        for o in range(4):
            tr = folds["outer_fold"] != o
            te = ~tr
            d = c.loc[te, ["RealTag", "ParticipantID", "IsFollowup"]].copy()
            d.insert(0, "outer_fold", o); d.insert(0, "seed", seed)
            d["y_true"] = c.loc[te, "label"].astype(float)
            d["y_score"] = np.clip(0.3 + 0.3 * d["y_true"] + rng.normal(0, 0.2, len(d)), 0.01, 0.99)
            d["threshold"] = 0.5; d["y_hat"] = 0
            pr.append(d)
            e = c.loc[tr, ["RealTag", "ParticipantID", "IsFollowup"]].copy()
            e.insert(0, "inner_fold", folds.loc[tr, f"inner_fold_o{o}"].to_numpy())
            e.insert(0, "outer_fold", o); e.insert(0, "seed", seed)
            e["y_true"] = c.loc[tr, "label"].astype(float)
            e["y_score"] = np.clip(0.3 + 0.3 * e["y_true"] + rng.normal(0, 0.2, len(e)), 0.01, 0.99)
            oo.append(e)
        pred[t], oof[t] = pd.concat(pr, ignore_index=True), pd.concat(oo, ignore_index=True)
    return pred, oof


def write_source(root, pred, oof, model="tabpfn"):
    for t in TASKS:
        d = root / model / f"task{t}"
        d.mkdir(parents=True)
        pred[t].to_csv(d / "predictions.csv", index=False)
        oof[t].to_csv(d / "inner_oof.csv", index=False)


def write_baseline_dir(root, seeds=(0,), model="tabpfn"):
    """A minimal finished baseline output directory: per-task cells plus run_info.json."""
    import json
    from pathlib import Path
    root = Path(root)
    pred_all, oof_all = {t: [] for t in TASKS}, {t: [] for t in TASKS}
    for s in seeds:
        pred, oof = fake_task_cells(seed=s)
        for t in TASKS:
            pred_all[t].append(pred[t]); oof_all[t].append(oof[t])
    write_source(root, {t: pd.concat(v, ignore_index=True) for t, v in pred_all.items()},
                 {t: pd.concat(v, ignore_index=True) for t, v in oof_all.items()}, model)
    (root / "run_info.json").write_text(json.dumps({"status": "completed", "smoke": True, "seeds": list(seeds)}))
    return root
