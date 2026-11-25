"""Data for wide_fusion: cohort, per-task eye features (applicable columns), the five clinical scores (regression
targets only), the input audit, the regression target C and the two card blocks (KD and reading)."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from common.config import load_data_config, repo_path
from common.data import applicable_columns, load_analysis_data, load_scores
from .composite import SCORE_COLS, composite, composite_params, sign_vector

# never an input (exact names): processed-file / cohort metadata and clinical columns
FORBIDDEN_INPUTS = {"Group", "RealTag", "IsFollowup", "ParticipantID", "CohortRole", "label", "SOB", "CDR_SOB", "MMSE",
                    "Age", "Sex", "PupilLeftBaseline", "PupilRightBaseline", "Final", "HasNulls", "Task"}
FORBIDDEN_NAME = re.compile(r"^(Ikä|Sukupuoli)$|CDR|MMSE|Poik|Baseline", re.IGNORECASE)


@dataclass(frozen=True)
class TargetSpec:
    """Regression target C: z moments (ddof = 1) from the fit's training rows only, higher = more impaired; rows with
    fewer than `min_scores` scores get NaN (left out of the regressor's training rows, still predicted as query rows)."""
    name: str
    S: np.ndarray               # (n, 5) clinical scores of the cohort visits (training targets only, never inputs)
    sign: np.ndarray
    min_scores: int

    def fit(self, idx, y) -> tuple[np.ndarray, np.ndarray, dict]:
        """(target of rows idx, number of available scores, params) from the rows idx only."""
        idx = np.asarray(idx)
        p = composite_params(self.S[idx], y[idx])
        C, nA = composite(self.S[idx], p, self.sign, self.min_scores)
        return C, nA, p


@dataclass(frozen=True)
class WideData:
    cohort: pd.DataFrame          # RealTag, ParticipantID, IsFollowup, label (no clinical score column)
    tags: np.ndarray
    pid: np.ndarray
    y: np.ndarray
    X: dict                       # task -> (n, p_t) eye features, applicable columns
    cols: dict                    # task -> column names
    targets: dict = field(default_factory=dict)   # name -> TargetSpec


@dataclass
class CardBlock:
    """One row per visit built from the visit's own card rows inside every fit: the per-card z blocks side by side.
    The design depends on the fit's training visits, so it is built by `features.wide_features` at fit time."""
    name: str
    tasks: list
    Xs: list                      # one (n, p) raw array per card, cohort order
    cols: list
    row_visit: np.ndarray
    row_task: np.ndarray          # -1: the row is not a single card
    pooled: bool = False


def audit_inputs(cols, X: np.ndarray, checks: dict) -> None:
    """No forbidden column name; no column with |Pearson r| > 0.999 with any vector in `checks` (age, clinical scores,
    label) over the cohort."""
    bad = [c for c in cols if c in FORBIDDEN_INPUTS or FORBIDDEN_NAME.search(c)]
    if bad:
        raise AssertionError(f"forbidden input columns: {bad}")
    for j, c in enumerate(cols):
        for name, v in checks.items():
            v = np.asarray(v, float)
            ok = ~np.isnan(X[:, j]) & ~np.isnan(v)
            if ok.sum() > 2 and np.std(X[ok, j]) > 0 and np.std(v[ok]) > 0 and abs(np.corrcoef(X[ok, j], v[ok])[0, 1]) > 0.999:
                raise AssertionError(f"column {c} has |r| > 0.999 with {name}")


def common_block_columns(tasks) -> list[str]:
    """Columns shared by the tasks of a block: the applicable columns, which must be identical (same order) on every
    task of the block, so no union or alignment is needed."""
    cols = [applicable_columns(t) for t in tasks]
    for t, c in zip(tasks, cols):
        if c != cols[0]:
            raise AssertionError(f"task {t}: applicable columns differ from task {tasks[0]}")
    return cols[0]


def _age(tags) -> np.ndarray:
    """Age (Ikä) of the visits, read for the input audit only (never an input or a target)."""
    d = load_data_config()
    c = pd.read_csv(repo_path(d["paths"]["cogdata"]), encoding="utf-8", usecols=[d["cogdata_key"], "Ikä"])
    c[d["cogdata_key"]] = c[d["cogdata_key"]].astype(str).str.strip()
    return pd.to_numeric(c.set_index(d["cogdata_key"]).loc[list(tags), "Ikä"], errors="coerce").to_numpy(float)


def load_data(cfg: dict, audit: bool = True) -> WideData:
    X, cols, cohort = {}, {}, None
    for t in cfg["tasks"]:
        X_df, _, _, c = load_analysis_data(f"task{t}")
        if cohort is None:
            cohort = c
        assert c["RealTag"].tolist() == cohort["RealTag"].tolist()
        X[t], cols[t] = X_df.to_numpy(dtype=float), list(X_df.columns)
    tags = cohort["RealTag"].to_numpy()
    y = cohort["label"].to_numpy()
    scores = load_scores().loc[tags]
    if audit:
        checks = {"age": _age(tags), "label": y.astype(float), **{f"score_{k}": scores[k].to_numpy(float) for k in scores.columns}}
        for t in cfg["tasks"]:
            if cols[t] != applicable_columns(t):
                raise AssertionError(f"task {t}: input columns differ from the applicable columns")
            audit_inputs(cols[t], X[t], checks)
    keep = cohort[["RealTag", "ParticipantID", "IsFollowup", "label"]].reset_index(drop=True)
    targets = {}
    for name, tc in cfg["targets"].items():
        if tc["kind"] != "composite":
            raise ValueError(f"{name}: unknown target kind {tc['kind']!r}")
        targets[name] = TargetSpec(name, scores[SCORE_COLS].to_numpy(float), sign_vector(tc["sign"]), tc["min_scores"])
    return WideData(keep, tags, cohort["ParticipantID"].to_numpy(), y, X, cols, targets)


def build_blocks(data: WideData, cfg: dict) -> list[CardBlock]:
    """One CardBlock per block of the config (KD: tasks 1, 3, 4, 5; reading: tasks 7, 9)."""
    n = len(data.tags)
    out = []
    for name, tasks in cfg["blocks"].items():
        cols = common_block_columns(tasks)
        if len(cols) != cfg["expected_common_columns"][name]:
            raise AssertionError(f"block {name}: {len(cols)} common columns, expected {cfg['expected_common_columns'][name]}")
        for t in tasks:
            assert data.cols[t] == cols
        out.append(CardBlock(name, list(tasks), [data.X[t] for t in tasks], list(cols), np.arange(n), np.full(n, -1)))
    if sorted(t for b in out for t in b.tasks) != sorted(cfg["tasks"]):
        raise AssertionError("the card blocks must cover every task exactly once")
    return out
