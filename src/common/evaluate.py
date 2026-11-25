"""Per-cell evaluation and output files.

A cell = (model, feature set) = `output/<run>/<model>/<feature_set>/` with predictions.csv (outer test rows, all seeds),
inner_oof.csv, selected_params.json and metrics.json.

Metric definitions
- Test AUC / Acc / Sen / Spec: the outer-test rows of one seed are pooled (each visit is a test row once per seed),
  the metric is computed on the pooled rows, then averaged over the seeds. Acc/Sen/Spec use the stored decision
  `y_hat` (Youden threshold of the inner OOF scores of that outer fold).
- Val AUC: AUC of the inner OOF scores of each outer fold (pooled over its inner folds), averaged over the 4 outer
  folds, then over the seeds.
- 95% CI: percentile interval of the same statistic over the shared stratified subject-level bootstrap draws
  (each replicate: resample subjects, replicate their visits, recompute the statistic, average over the seeds).
"""
from __future__ import annotations
import json
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from .bootstrap import bootstrap_thetas, percentile_ci
from .metrics import auc_score, cls_metrics, threshold_metrics

TEST_CI = ["auc", "acc", "sens", "spec"]
SUMMARY_KEYS = {"val_auc": ("val", "auc"), "test_auc": ("test", "auc"), "test_acc": ("test", "acc"),
                "test_sen": ("test", "sens"), "test_spec": ("test", "spec")}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def ci_metric_fn(d: dict) -> dict:
    m = {"auc": auc_score(d["y"], d["score"])}
    if "yhat" in d:
        m.update(threshold_metrics(d["y"], d["yhat"]))
    return m


def _summ(per_seed: dict[int, dict], names: list[str]) -> tuple[dict, dict]:
    mean, spread = {}, {}
    for n in names:
        v = np.array([per_seed[s][n] for s in sorted(per_seed)], dtype=float)
        mean[n] = float(np.mean(v))
        spread[n] = {"sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
                     "min": float(np.min(v)), "max": float(np.max(v))}
    return mean, spread


def compute_metrics(predictions: pd.DataFrame, inner_oof: pd.DataFrame | None, draws: dict,
                    val_note: str | None = None) -> dict:
    """Test and validation metrics of one cell: per seed, per fold, seed means and bootstrap CIs.

    inner_oof=None (a fixed rule without model selection): no validation metrics; val_auc is null and `val_auc_note`
    carries `val_note`."""
    subj_index = {s: i for i, s in enumerate(draws["subjects"])}
    seeds = sorted(predictions["seed"].unique())
    test_names = ["auc", "acc", "sens", "spec", "bal_acc", "tp", "fp", "tn", "fn"]

    def comp(df, with_hat):
        c = {"sidx": df["ParticipantID"].map(subj_index).to_numpy(), "y": df["y_true"].to_numpy(),
             "score": df["y_score"].to_numpy()}
        if with_hat:
            c["yhat"] = df["y_hat"].to_numpy()
        return c

    test_seed, test_fold, val_seed, val_fold, test_groups, val_groups = {}, {}, {}, {}, [], []
    for s in seeds:
        p = predictions[predictions.seed == s]
        test_seed[s] = cls_metrics(p["y_true"], p["y_score"], p["y_hat"])
        test_fold[s] = {int(o): cls_metrics(q["y_true"], q["y_score"], q["y_hat"]) for o, q in p.groupby("outer_fold")}
        test_groups.append([comp(p, True)])
        if inner_oof is None:
            continue
        v = inner_oof[inner_oof.seed == s]
        val_fold[s] = {int(o): cls_metrics(q["y_true"], q["y_score"]) for o, q in v.groupby("outer_fold")}
        val_seed[s] = {"auc": float(np.mean([f["auc"] for f in val_fold[s].values()]))}
        val_groups.append([comp(q, False) for _, q in v.groupby("outer_fold")])

    th_test = bootstrap_thetas(test_groups, draws["counts"], ci_metric_fn, TEST_CI)
    has_val = inner_oof is not None
    th_val = bootstrap_thetas(val_groups, draws["counts"], ci_metric_fn, ["auc"]) if has_val else None
    first = predictions[predictions.seed == seeds[0]]
    out = {"seeds": [int(s) for s in seeds], "n_visits": int(len(first)),
           "n_subjects": int(first["ParticipantID"].nunique()), "n_pos": int((first["y_true"] == 1).sum()),
           "n_neg": int((first["y_true"] != 1).sum())}
    parts = [("test", test_seed, test_fold, test_names, th_test, TEST_CI)]
    if has_val:
        parts.append(("val", val_seed, val_fold, ["auc"], th_val, ["auc"]))
    for tag, per_seed, per_fold, nm, th, ci_names in parts:
        mean, spread = _summ(per_seed, nm)
        out[tag] = {"per_seed": per_seed, "per_fold": per_fold, "mean": mean, "spread": spread,
                    "n_seeds": len(per_seed), "ci": {n: percentile_ci(th[n]) for n in ci_names}}
    out["bootstrap"] = {"B": int(draws["B"]), "seed": int(draws["seed"]), "draws_sha256": draws.get("sha256"),
                        "n_nan_replicates": {"test": {n: int(np.isnan(th_test[n]).sum()) for n in TEST_CI},
                                             "val": {"auc": int(np.isnan(th_val["auc"]).sum())} if has_val else None}}
    out["summary"] = {}
    for key, (tag, m) in SUMMARY_KEYS.items():
        if tag == "val" and not has_val:
            out["summary"][key] = {"mean": None, "lo": None, "hi": None}
            continue
        ci = out[tag]["ci"][m]
        out["summary"][key] = {"mean": out[tag]["mean"][m], "lo": ci["lo"], "hi": ci["hi"]}
    if not has_val:
        out["val"] = None
        out["val_auc_note"] = val_note or "no validation AUC"
    return out


# ---------------------------------------------------------------------------------------------------
# per-seed partial results (resume) and cell files
def partial_path(cell_dir: Path, seed: int) -> Path:
    return Path(cell_dir) / "_partial" / f"seed_{seed}.pkl"


def write_partial(result: dict, cell_dir: Path, seed: int, fingerprint: str) -> None:
    """Result + fingerprint of the producing configuration, moved into place atomically."""
    p = partial_path(cell_dir, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(dict(result, fingerprint=fingerprint), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def read_partial(cell_dir: Path, seed: int) -> dict | None:
    p = partial_path(cell_dir, seed)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def write_cell(cell_dir: Path, family: str, model: str, feature_set: str, predictions: pd.DataFrame,
               inner_oof: pd.DataFrame | None, selected: dict, draws: dict, extra: dict | None = None,
               cv_results: pd.DataFrame | None = None, val_note: str | None = None,
               footnote: str | None = None) -> dict:
    """Write the files of one cell and return its metrics."""
    cell_dir = Path(cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(cell_dir / "predictions.csv", index=False)
    if inner_oof is not None:
        inner_oof.to_csv(cell_dir / "inner_oof.csv", index=False)
    if cv_results is not None:
        cv_results.to_csv(cell_dir / "cv_results.csv", index=False)
    (cell_dir / "selected_params.json").write_text(json.dumps(jsonable(selected), indent=1), encoding="utf-8")
    metrics = compute_metrics(predictions, inner_oof, draws, val_note)
    metrics.update({"family": family, "model": model, "feature_set": feature_set, **(extra or {})})
    if footnote:
        metrics["footnote"] = footnote
    (cell_dir / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=1), encoding="utf-8")
    return metrics


def finalize_cell(cell_dir: Path, family: str, model: str, feature_set: str, seeds: list[int], expected: dict,
                  draws: dict) -> dict:
    """Merge the per-seed partials of one fitted cell into the cell files (and remove the partials)."""
    parts = {}
    for s in seeds:
        r = read_partial(cell_dir, s)
        if r is None:
            raise RuntimeError(f"{cell_dir.name}: seed {s} is missing")
        if r.get("fingerprint") != expected[s]:
            raise RuntimeError(f"{cell_dir.name} seed {s}: partial result was produced by a different configuration; "
                               "use a new output directory")
        parts[s] = r
    cat = lambda k: pd.concat([parts[s][k] for s in seeds], ignore_index=True)  # noqa: E731
    metrics = write_cell(cell_dir, family, model, feature_set, cat("predictions"), cat("inner_oof"),
                         {s: parts[s]["selected"] for s in seeds}, draws,
                         {"job_seconds": {int(s): parts[s]["seconds"] for s in seeds}}, cat("cv_results"))
    for s in seeds:
        partial_path(cell_dir, s).unlink()
    partial_path(cell_dir, seeds[0]).parent.rmdir()
    return metrics


def load_cell_metrics(path: Path) -> dict:
    return json.loads((Path(path) / "metrics.json").read_text(encoding="utf-8"))
