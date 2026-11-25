"""Assembly from saved and new members (no fit in this module).

Saved members (finished runs, read only): the 6 per-task baseline TabPFN classifiers (`<baseline>/tabpfn/task{t}`) and the
6 per-task cognitive_fusion C regressors (`<cognitive_fusion run>/regression/task{t}`), test and inner OOF.
New members (this run): the wide classifier and the wide C regressor; their per-block scores are read from
`rows_predictions` / `rows_inner_oof`, so a block alone (KD, reading) needs no extra fit.

Derived score frames (test, inner OOF), name -> (pred, oof) with y_score:
  kd_meanprob, kd_meanreg                               means of the 4 saved KD per-card members
  {kd,rd}wide_{cls,reg}                                 block scores of the wide members
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from common.config import file_sha256, repo_path
from common.latefusion import KEYS_O, KEYS_P, load_task_cells, members
from . import combine as CB

BLOCK_TAG = {"kd": "kd", "reading": "rd"}


# ---------------------------------------------------------------------------------------------------
# alignment helpers
def key_set(df: pd.DataFrame, keys) -> frozenset:
    if df[keys].duplicated().any():
        raise AssertionError("duplicated (seed, outer fold, [inner fold], RealTag) rows")
    return frozenset(map(tuple, df[keys].to_numpy().tolist()))


def assert_same_keys(frames: dict, keys, base: pd.DataFrame | None = None) -> None:
    """Every named frame covers exactly the same key set (and the same rows as `base` if given); no duplicates."""
    sets = {n: key_set(f, keys) for n, f in frames.items()}
    ref = key_set(base, keys) if base is not None else next(iter(sets.values()))
    bad = [n for n, s in sets.items() if s != ref]
    if bad:
        raise AssertionError(f"member frames {bad} do not cover the same (seed, outer fold, RealTag) keys as the others")


def align(df: pd.DataFrame, base: pd.DataFrame, keys, col: str) -> np.ndarray:
    """`col` of `df` on the rows of `base`; same key set, no NaN, same ParticipantID where present."""
    assert_same_keys({"member": df}, keys, base)
    s = df.set_index(keys)
    idx = pd.MultiIndex.from_frame(base[keys])
    v = s[col].reindex(idx).to_numpy(float)
    if not np.isfinite(v).all():
        raise AssertionError("a member score is missing or not finite")
    if "ParticipantID" in s.columns and not np.array_equal(s["ParticipantID"].reindex(idx).astype(str).to_numpy(),
                                                          base["ParticipantID"].astype(str).to_numpy()):
        raise AssertionError("members disagree on ParticipantID")
    return v


def col_mean(M: np.ndarray, idx) -> np.ndarray:
    """Plain mean of columns `idx` (the arithmetic of cognitive_fusion.combine: sum / count)."""
    return np.ascontiguousarray(M[:, list(idx)]).sum(axis=1) / len(idx)


def with_score(base: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    return base.assign(y_score=np.asarray(score, float))


# ---------------------------------------------------------------------------------------------------
# saved members
def load_saved(cfg: dict, seeds) -> dict:
    """Saved per-task classifier probabilities and C regressions on the rows of the baseline cells.
    Columns of P / R follow cfg['tasks']."""
    tasks, seeds = list(cfg["tasks"]), list(seeds)
    bdir = repo_path(cfg["baseline_classifier"]["run"])
    pred_t, oof_t = load_task_cells(bdir, cfg["baseline_classifier"]["model"], tasks, seeds)
    bt, Pt = members(pred_t, KEYS_P)
    bo, Po = members(oof_t, KEYS_O)
    rdir = repo_path(cfg["saved_regressors"]["run"])
    Rt, Ro = np.empty_like(Pt), np.empty_like(Po)
    files = {}
    for j, t in enumerate(tasks):
        for name, base, keys, out in (("predictions", bt, KEYS_P, Rt), ("inner_oof", bo, KEYS_O, Ro)):
            f = rdir / "regression" / f"task{t}" / f"{name}.csv"
            d = pd.read_csv(f)
            d = d[d.seed.isin(seeds)].reset_index(drop=True)
            out[:, j] = align(d, base, keys, "y_pred")
            files[f"regression/task{t}/{name}.csv"] = file_sha256(f)
        for name in ("predictions", "inner_oof"):
            files[f"baseline/{cfg['baseline_classifier']['model']}/task{t}/{name}.csv"] = file_sha256(
                bdir / cfg["baseline_classifier"]["model"] / f"task{t}" / f"{name}.csv")
    return {"bt": bt, "bo": bo, "Pt": Pt, "Po": Po, "Rt": Rt, "Ro": Ro, "tasks": tasks, "sha256": files}


def identity_check_saved(saved: dict, cfg: dict, seeds, w: float, clip: float, tol: float = 1e-9) -> dict:
    """The saved 6 classifiers + 6 regressors, fused as in cognitive_fusion, reproduce the saved cogfusion_w050 cell
    (test and inner-OOF scores, and y_hat). Also the saved mean of the classifiers equals tabpfn_meanprob exactly."""
    tasks, ix = saved["tasks"], list(range(len(saved["tasks"])))
    bt, bo = saved["bt"], saved["bo"]
    p_te, p_oo = col_mean(saved["Pt"], ix), col_mean(saved["Po"], ix)
    CB.identity_check(with_score(bt, p_te), with_score(bo, p_oo), repo_path(cfg["references"]["tabpfn_meanprob"]), seeds)
    r_te, r_oo = col_mean(saved["Rt"], ix), col_mean(saved["Ro"], ix)
    (pred, oof), _ = CB.fuse((with_score(bt, p_te), with_score(bo, p_oo)), (with_score(bt, r_te), with_score(bo, r_oo)), w, clip)
    ref = repo_path(cfg["references"]["cogfusion_w050"])
    out = {}
    for frame, name, keys in ((pred, "predictions.csv", KEYS_P), (oof, "inner_oof.csv", KEYS_O)):
        r = pd.read_csv(ref / name)
        r = r[r.seed.isin(seeds)]
        if len(r) != len(frame):
            raise AssertionError(f"{name}: row count differs from cogfusion_w050")
        rs = r.set_index(keys)
        rr = rs["y_score"].reindex(pd.MultiIndex.from_frame(frame[keys])).to_numpy()
        d = float(np.max(np.abs(rr - frame["y_score"].to_numpy())))
        if not d <= tol:
            raise AssertionError(f"saved members do not reproduce cogfusion_w050 {name} (max |d| = {d})")
        if name == "predictions.csv":
            yh = rs["y_hat"].reindex(pd.MultiIndex.from_frame(frame[keys])).to_numpy()
            if not np.array_equal(yh, frame["y_hat"].to_numpy()):
                raise AssertionError("saved members do not reproduce the cogfusion_w050 decisions")
        out[name] = {"max_abs_diff": d, "rows": int(len(frame))}
    return out


# ---------------------------------------------------------------------------------------------------
# derived frames
def block_scores(rows: pd.DataFrame, block: str, base: pd.DataFrame, keys) -> np.ndarray:
    return align(rows[rows["block"] == block].drop(columns=["task", "block"]), base, keys, "y_row")


def build_derived(cfg: dict, saved: dict, member_rows: dict, kinds: dict) -> dict:
    """name -> (pred, oof) for every derived score; `member_rows[m] = (rows_predictions, rows_inner_oof)`;
    `kinds` = {"cls": classifier member, "reg": regressor member}."""
    tasks = saved["tasks"]
    bt, bo = saved["bt"], saved["bo"]
    pos = {t: j for j, t in enumerate(tasks)}
    kd = [pos[t] for t in cfg["blocks"]["kd"]]
    out = {}

    def put(name, te, oo):
        out[name] = (with_score(bt, te), with_score(bo, oo))

    put("kd_meanprob", col_mean(saved["Pt"], kd), col_mean(saved["Po"], kd))
    put("kd_meanreg", col_mean(saved["Rt"], kd), col_mean(saved["Ro"], kd))
    for kind, m in kinds.items():
        rp, ro = member_rows[m]
        for bname, tag in BLOCK_TAG.items():
            put(f"{tag}wide_{kind}", block_scores(rp, bname, bt, KEYS_P), block_scores(ro, bname, bo, KEYS_O))
    return out


# ---------------------------------------------------------------------------------------------------
# logs of the wide members
def wide_logs(cfg: dict, parts: dict, seeds) -> tuple[dict, dict]:
    """(full per-fit log by member, feature check by member and block) from the fit logs of the wide members."""
    full, check = {}, {}
    for m in cfg["members"]:
        rows = []
        for s in seeds:
            for o in range(4):
                for f in parts[(m, s, o)]["fit_log"]:
                    rows.append({"member": m, "seed": s, "outer_fold": o, "fit": f["fit"], "block": f["block"],
                                 "n_train_visits": len(f["train"]), "n_query_visits": len(f["query"]),
                                 "tabpfn_check": f.get("tabpfn_check"),
                                 **{k: v for k, v in f["wide_log"].items() if k not in ("param_tags", "names")}})
        full[m] = rows
        by_block = {}
        for r in rows:
            by_block.setdefault(r["block"], []).append(r)
        check[m] = {}
        for b, rs in by_block.items():
            tc = [r["tabpfn_check"] for r in rs]
            if not all(t is not None for t in tc):
                continue

            def subs(t):    # an estimator that sees fewer columns than the fitted width = feature subsampling
                return any(c != t["n_features_in_model"] for c in t["per_estimator_features"])
            drops = {f"seed{r['seed']}/o{r['outer_fold']}/{r['fit']}": t["degenerate_dropped"]
                     for r, t in zip(rs, tc) if t["n_degenerate_dropped"]}
            check[m][b] = {"n_fits": len(tc),
                           "columns_offered": {"min": min(t["n_columns_offered"] for t in tc), "max": max(t["n_columns_offered"] for t in tc)},
                           "degenerate_dropped": {"n_fits_with_a_drop": len(drops),
                                                  "n_dropped_per_fit": {"min": min(t["n_degenerate_dropped"] for t in tc),
                                                                        "max": max(t["n_degenerate_dropped"] for t in tc)},
                                                  "per_fit": drops,
                                                  "note": "all-constant / all-NaN columns dropped by the wrapper's DropDegenerate rule (allowed); not subsampling"},
                           "columns_fitted": {"min": min(t["n_features_in_model"] for t in tc), "max": max(t["n_features_in_model"] for t in tc)},
                           "offered_minus_dropped_equals_fitted": all(t["n_columns_offered"] - t["n_degenerate_dropped"] == t["n_features_in_model"]
                                                                      for t in tc),
                           "n_estimators": sorted({t["n_estimators"] for t in tc}),
                           "max_features_per_estimator": sorted({x for t in tc for x in t["max_features_per_estimator"]}),
                           "per_estimator_features_min": min(min(t["per_estimator_features"]) for t in tc),
                           "per_estimator_features_max": max(max(t["per_estimator_features"]) for t in tc),
                           "n_fits_with_a_subsampled_estimator": sum(subs(t) for t in tc),
                           "all_estimators_see_all_fitted_columns": not any(subs(t) for t in tc)}
    return full, check
