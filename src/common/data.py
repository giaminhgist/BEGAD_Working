"""Data layer: cohort and feature loading, applicable-feature filter, fold-file merge with fold-structure assertions.

Cohort report: `PYTHONPATH=src .venv/bin/python -m common.data`.
"""
from __future__ import annotations
import json
from functools import lru_cache
import numpy as np
import pandas as pd
from .config import load_data_config, repo_path

FS_ORDER = ["task1", "task3", "task4", "task5", "task7", "task9", "fusion"]
FOLD_COLS = ["outer_fold", "inner_fold_o0", "inner_fold_o1", "inner_fold_o2", "inner_fold_o3"]


@lru_cache(maxsize=None)
def _read_task_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_task(task: int, cfg: dict) -> pd.DataFrame:
    return _read_task_cached(str(repo_path(cfg["paths"]["features_dir"]) / str(task) / "features.csv")).copy()


def load_scores(cfg: dict | None = None) -> pd.DataFrame:
    """Clinical scores indexed by RealTag, short target names, numeric (blank -> NaN)."""
    cfg = cfg or load_data_config()
    c = pd.read_csv(repo_path(cfg["paths"]["cogdata"]), encoding="utf-8")
    c[cfg["cogdata_key"]] = c[cfg["cogdata_key"]].astype(str).str.strip()
    out = pd.DataFrame({"RealTag": c[cfg["cogdata_key"]]})
    for short, col in cfg["targets"].items():
        out[short] = pd.to_numeric(c[col], errors="coerce")
    assert out["RealTag"].is_unique, "COGDATA RealTag not unique"
    return out.set_index("RealTag")


def load_cohort(cfg: dict | None = None) -> pd.DataFrame:
    """One row per visit (176): RealTag, ParticipantID, Group, label, IsFollowup, clinical scores.

    Sorted by RealTag so the row order is deterministic.
    """
    cfg = cfg or load_data_config()
    f = _read_task(cfg["tasks"][0], cfg)
    f = f[f["Group"].isin(cfg["cohort"]["groups"])][cfg["metadata_columns"]].copy()
    assert f["RealTag"].is_unique, "cohort RealTag not unique"
    f["label"] = (f["Group"] == cfg["cohort"]["positive_group"]).astype(int)
    f = f.join(load_scores(cfg), on="RealTag")
    return f.sort_values("RealTag").reset_index(drop=True)


def load_task_features(task: int, cfg: dict | None = None) -> pd.DataFrame:
    """Feature columns of one task (before the applicability filter), indexed by RealTag, cohort rows only."""
    cfg = cfg or load_data_config()
    cohort = load_cohort(cfg)
    d = _read_task(task, cfg).set_index("RealTag")
    excluded = set(cfg.get("excluded_features") or [])
    feats = [c for c in d.columns if c not in cfg["metadata_columns"] and c not in excluded]
    X = d.loc[cohort["RealTag"], feats]
    empty = X.index[X.isna().all(axis=1)]
    assert len(empty) == 0, f"task {task}: visits with all features NaN: {list(empty)[:5]}"
    return X


def load_fusion_features(cfg: dict | None = None) -> pd.DataFrame:
    """Early fusion: all tasks side by side, columns prefixed t{task}__, indexed by RealTag."""
    cfg = cfg or load_data_config()
    parts = [load_task_features(t, cfg).add_prefix(f"t{t}__") for t in cfg["tasks"]]
    return pd.concat(parts, axis=1)


def parse_feature_set(name: str) -> str | int:
    if name == "fusion":
        return "fusion"
    if name.startswith("task") and name[4:].isdigit():
        return int(name[4:])
    raise ValueError(f"unknown feature set {name!r}")


def applicable_columns(task: int, cfg: dict | None = None) -> list[str]:
    """Features defined for a task: columns that are not entirely missing in the task's feature file.

    A feature that does not apply to a task (e.g. reading features on a King-Devick card) is blank for every
    recording of that task, so applicability is read from the data itself.
    """
    cfg = cfg or load_data_config()
    d = _read_task(int(task), cfg)
    excluded = set(cfg.get("excluded_features") or [])
    return [c for c in d.columns if c not in cfg["metadata_columns"] and c not in excluded and d[c].notna().any()]


def select_applicable(X: pd.DataFrame, feature_set: str | int, cfg: dict | None = None) -> pd.DataFrame:
    """Keep only applicable columns. For fusion the columns are `t{task}__name` and are filtered per task."""
    fs = parse_feature_set(feature_set) if isinstance(feature_set, str) else feature_set
    if fs == "fusion":
        keep = {t: set(applicable_columns(t, cfg)) for t in (cfg or load_data_config())["tasks"]}
        cols = []
        for c in X.columns:
            prefix, name = c.split("__", 1)
            if name in keep[int(prefix[1:])]:
                cols.append(c)
        return X[cols]
    return X[[c for c in X.columns if c in set(applicable_columns(int(fs), cfg))]]


def load_analysis_data(feature_set: str, cfg: dict | None = None):
    """(X [applicable columns], y, groups, cohort) for a feature-set name; y = diagnosis label (1 = MCI)."""
    cfg = cfg or load_data_config()
    cohort = load_cohort(cfg)
    fs = parse_feature_set(feature_set)
    X = load_fusion_features(cfg) if fs == "fusion" else load_task_features(fs, cfg)
    X = select_applicable(X.loc[cohort["RealTag"]], fs, cfg)
    return X, cohort["label"].to_numpy(), cohort["ParticipantID"].to_numpy(), cohort


def cohort_report(cfg: dict | None = None) -> dict:
    cfg = cfg or load_data_config()
    c = load_cohort(cfg)
    n_groups = c.groupby("ParticipantID")["Group"].nunique()
    return {
        "n_visits": len(c),
        "n_subjects": int(c["ParticipantID"].nunique()),
        "group_counts_visits": c["Group"].value_counts().to_dict(),
        "n_followup_visits": int((c["IsFollowup"] == 1).sum()),
        "missing_scores": {k: int(c[k].isna().sum()) for k in cfg["targets"]},
        "subjects_with_changing_group": sorted(n_groups[n_groups > 1].index),
    }


# ---------------------------------------------------------------------------------------------------
# folds merged onto the cohort
def merge_folds(cohort: pd.DataFrame, folds: pd.DataFrame, n_outer: int = 4, n_inner: int = 4,
                min_per_class: int = 5) -> pd.DataFrame:
    """`cohort[["RealTag"]]` + fold columns aligned to the cohort rows (merge by RealTag).

    Raises AssertionError on any mismatch of subject or label, or on a broken fold structure.
    """
    assert folds["RealTag"].is_unique, "fold file RealTag not unique"
    assert cohort["RealTag"].is_unique, "cohort RealTag not unique"
    missing = set(cohort["RealTag"]) - set(folds["RealTag"])
    assert not missing, f"cohort visits missing in fold file: {sorted(missing)[:5]}"
    m = cohort[["RealTag", "ParticipantID", "label"]].merge(
        folds, on="RealTag", how="left", suffixes=("", "_fold"), validate="one_to_one")
    assert len(m) == len(cohort) and m["outer_fold"].notna().all()
    assert (m["ParticipantID"] == m["ParticipantID_fold"]).all(), "ParticipantID mismatch cohort vs folds"
    assert (m["label"] == m["label_fold"]).all(), "label mismatch cohort vs folds"
    out = m[["RealTag"] + FOLD_COLS].copy()
    out[FOLD_COLS] = out[FOLD_COLS].astype(int)
    out = out.set_index(cohort.index)
    check_fold_structure(out, cohort["ParticipantID"].to_numpy(), cohort["label"].to_numpy(), n_outer, n_inner,
                         min_per_class)
    return out


def check_fold_structure(f: pd.DataFrame, groups: np.ndarray, y: np.ndarray,
                         n_outer: int = 4, n_inner: int = 4, min_per_class: int = 1) -> None:
    """Subject-disjoint outer and inner folds, inner == -1 iff the row is in that outer test fold,
    at least `min_per_class` visits of each class in every outer and inner fold."""
    g = pd.Series(np.asarray(groups), index=f.index)
    yy = pd.Series(np.asarray(y), index=f.index)
    outer = f["outer_fold"]
    assert set(outer.unique()) == set(range(n_outer)), "every row must be in one of the outer folds"
    assert (outer.groupby(g).nunique() == 1).all(), "subject in >1 outer fold"
    for o in range(n_outer):
        col = f[f"inner_fold_o{o}"]
        assert ((col == -1) == (outer == o)).all(), f"inner_fold_o{o}: -1 iff outer_fold == {o} violated"
        tr = outer != o
        assert set(col[tr].unique()) == set(range(n_inner)), f"inner folds of outer {o} incomplete"
        assert (col[tr].groupby(g[tr]).nunique() == 1).all(), f"subject in >1 inner fold (outer {o})"
        for part in [outer == o] + [col == i for i in range(n_inner)]:
            vc = yy[part].value_counts()
            assert vc.reindex([0, 1]).fillna(0).min() >= min_per_class, "fold with too few samples of a class"


if __name__ == "__main__":
    print(json.dumps(cohort_report(), indent=2, ensure_ascii=False, default=str))
