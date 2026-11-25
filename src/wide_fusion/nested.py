"""Nested-CV member job: one (member, fold seed, outer fold) = 4 inner fits + 1 outer refit, each fit = one TabPFN per
card block (KD, reading) on one row per visit with the per-card features side by side. Inner fits give the inner-OOF
scores, the refit gives the test scores.

Inside every fit, from that fit's training visits only:
- the regression target C (visits with fewer than `min_scores` scores are not training rows);
- the per-card z parameters of the wide design.
A visit's score is the mean of its block predictions (KD and reading). Every fit is logged (training visits, query
visits, parameter rows) for `audit_fit_log`.
"""
from __future__ import annotations
import gc
import time
import numpy as np
import pandas as pd
from . import features as FT
from .data import CardBlock, WideData, build_blocks
from .tabcheck import fit_checked


def aggregate_rows(row_visit: np.ndarray, values: np.ndarray, n_per_visit: int):
    """(sorted visit indices, mean of each visit's own rows). Every visit must have exactly n_per_visit finite rows."""
    row_visit, values = np.asarray(row_visit), np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("a row prediction is missing or not finite")
    visits, inv, counts = np.unique(row_visit, return_inverse=True, return_counts=True)
    if not (counts == n_per_visit).all():
        raise AssertionError(f"visits with a row count other than {n_per_visit}")
    sums = np.zeros(len(visits))
    np.add.at(sums, inv, values)
    return visits, sums / n_per_visit


def fit_card_block(block: CardBlock, train_idx, query_idx, target: np.ndarray, model, kind: str, data: WideData,
                   n_estimators: int | None = None, check: bool = False) -> dict:
    """Fit one block: ONE row per visit; the design (per-card z) comes from this fit's training visits only."""
    tv = np.isin(block.row_visit, train_idx) & np.isfinite(target[block.row_visit])
    qv = np.isin(block.row_visit, query_idx)
    tr_rows, q_rows = np.flatnonzero(tv), np.flatnonzero(qv)
    tr_vis, q_vis = np.unique(block.row_visit[tv]), np.unique(block.row_visit[qv])
    if set(data.pid[tr_vis]) & set(data.pid[q_vis]):
        raise AssertionError(f"{block.name}: training and query subjects overlap")
    d = FT.wide_features(block.Xs, block.cols, block.tasks, tr_rows, q_rows)
    if len(d["Xq"]) != len(q_vis) or len(d["Xtr"]) != len(tr_vis):
        raise AssertionError(f"{block.name}: not exactly one design row per visit")
    yt = target[block.row_visit[tv]]
    t0 = time.perf_counter()
    if check:
        chk = fit_checked(model, d["Xtr"], yt, n_estimators, names=d["names"])
    else:
        model.fit(d["Xtr"], yt)
        chk = None
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict_proba(d["Xq"])[:, 1] if kind == "classification" else model.predict(d["Xq"])
    pred_s = time.perf_counter() - t0
    log = {**d["log"], "names": d["names"], "param_tags": data.tags[tr_vis].tolist()}
    return {"rows": q_rows, "pred": np.asarray(pred, dtype=np.float64), "train_visits": tr_vis,
            "query_visits": q_vis, "n_train_rows": int(tv.sum()), "n_query_rows": int(qv.sum()),
            "n_features": int(d["Xtr"].shape[1]), "fit_seconds": fit_s, "predict_seconds": pred_s,
            "wide_log": log, "tabpfn_check": chk}


def run_member_job(member: str, spec: dict, seed: int, o: int, data: WideData, folds: pd.DataFrame, cfg: dict,
                   factory, n_threads: int = 1) -> dict:
    t_start = time.perf_counter()
    kind = spec["kind"]
    if kind not in ("classification", "regression"):
        raise ValueError(f"{member}: unknown kind {kind!r}")
    if spec.get("layout", "wide") != "wide":
        raise ValueError(f"{member}: only the wide layout is implemented")
    blocks = build_blocks(data, cfg)
    rows_per_visit = len(blocks)
    check = bool(cfg.get("tabpfn_feature_check"))
    make_model = factory(kind, spec["n_estimators"], n_threads)
    tgt = data.targets[spec["target"]] if kind == "regression" else None
    tags, n = data.tags, len(data.tags)
    outer = folds["outer_fold"].to_numpy()
    tr, te = np.flatnonzero(outer != o), np.flatnonzero(outer == o)
    inner = folds[f"inner_fold_o{o}"].to_numpy()

    def target_for(idx):
        t = np.full(n, np.nan)
        if kind == "classification":
            t[idx] = data.y[idx].astype(float)
            return t, None
        C, nA, p = tgt.fit(idx, data.y)
        t[idx] = C
        return t, {"mu": p["mu"].tolist(), "sd": p["sd"].tolist(), "rows": tags[idx].tolist(),
                   "excluded": tags[idx][nA < tgt.min_scores].tolist()}

    # outer-training-scale C of the inner-OOF visits: diagnostics only (inner Spearman of r̄ with C)
    y_target = np.full(n, np.nan)
    if kind == "regression":
        y_target = target_for(tr)[0]

    fit_log, tgt_log, rows_out, visit_out = [], {"inner": {}}, {"inner": [], "test": []}, {"inner": [], "test": []}

    def one_fit(label, train_idx, query_idx):
        target, tp = target_for(train_idx)
        if tp is not None:
            if label == "refit":
                tgt_log["refit"] = tp
            else:
                tgt_log["inner"][label] = tp
        row_v, row_p = [], []
        for b in blocks:
            r = fit_card_block(b, train_idx, query_idx, target, make_model(seed), kind, data, spec["n_estimators"], check)
            fit_log.append({"member": member, "seed": seed, "outer_fold": o, "fit": label, "block": b.name,
                            "train": tags[r["train_visits"]].tolist(), "query": tags[r["query_visits"]].tolist(),
                            "z_param_tags": tags[r["train_visits"]].tolist(), "wide_log": r["wide_log"],
                            "tabpfn_check": r["tabpfn_check"],
                            "target_param_tags": None if tp is None else list(tp["rows"]),
                            "n_train_rows": r["n_train_rows"], "n_query_rows": r["n_query_rows"],
                            "n_features": r["n_features"], "fit_seconds": r["fit_seconds"],
                            "predict_seconds": r["predict_seconds"]})
            rv = b.row_visit[r["rows"]]
            row_v.append(rv)
            row_p.append(r["pred"])
            rows_out["test" if label == "refit" else "inner"].append(pd.DataFrame(
                {"seed": seed, "outer_fold": o, "inner_fold": -1 if label == "refit" else int(label[5:]),
                 "RealTag": tags[rv], "task": b.row_task[r["rows"]], "block": b.name, "y_row": r["pred"]}))
            gc.collect()
        rv, rp = np.concatenate(row_v), np.concatenate(row_p)
        vis, score = aggregate_rows(rv, rp, rows_per_visit)
        if set(vis.tolist()) != set(np.asarray(query_idx).tolist()):
            raise AssertionError(f"{label}: aggregated visits differ from the query visits")
        return vis, score

    for i in range(4):
        itr, iva = tr[inner[tr] != i], tr[inner[tr] == i]
        vis, score = one_fit(f"inner{i}", itr, iva)
        visit_out["inner"].append(pd.DataFrame({"seed": seed, "outer_fold": o, "inner_fold": i, "_v": vis, "y_score": score}))
    vis, score = one_fit("refit", tr, te)
    visit_out["test"].append(pd.DataFrame({"seed": seed, "outer_fold": o, "_v": vis, "y_score": score}))

    def finish(df, with_target):
        v = df.pop("_v").to_numpy()
        c = data.cohort.iloc[v].reset_index(drop=True)
        out = pd.concat([df.reset_index(drop=True), c[["RealTag", "ParticipantID", "IsFollowup"]]], axis=1)
        out["y_true"] = data.y[v].astype(float)
        if with_target:
            out["y_target"] = y_target[v]
        return out

    oof = finish(pd.concat(visit_out["inner"], ignore_index=True), True)
    oof = oof[["seed", "outer_fold", "inner_fold", "RealTag", "ParticipantID", "IsFollowup", "y_true", "y_score", "y_target"]]
    pred = finish(pd.concat(visit_out["test"], ignore_index=True), False)
    pred = pred[["seed", "outer_fold", "RealTag", "ParticipantID", "IsFollowup", "y_true", "y_score"]]
    rows_te = pd.concat(rows_out["test"], ignore_index=True).drop(columns="inner_fold")
    return {"member": member, "seed": seed, "outer_fold": o, "predictions": pred, "inner_oof": oof,
            "rows_predictions": rows_te, "rows_inner_oof": pd.concat(rows_out["inner"], ignore_index=True),
            "fit_log": fit_log, "target_params": tgt_log if kind == "regression" else None,
            "seconds": time.perf_counter() - t_start}


def audit_fit_log(fit_log: list, data: WideData, folds: pd.DataFrame) -> None:
    """Every fit: training and query visits disjoint (visit and subject); no outer-test visit of the fold in any
    training set; inner fits use the right inner split; z parameters come from exactly the fitted training visits and
    the target moments from the fit's training visits, never from a query visit."""
    pid = dict(zip(data.tags, data.pid))
    outer = dict(zip(data.tags, folds["outer_fold"].to_numpy()))
    for f in fit_log:
        tr, q, o = set(f["train"]), set(f["query"]), f["outer_fold"]
        assert tr and q, f"{f['fit']}: empty training or query set"
        assert not (tr & q), f"{f['fit']}: train/query visit overlap"
        assert not ({pid[t] for t in tr} & {pid[t] for t in q}), f"{f['fit']}: train/query subject overlap"
        assert all(outer[t] != o for t in tr), f"{f['fit']}: an outer-test visit of fold {o} was fitted on"
        if f["fit"] == "refit":
            fit_set = {t for t, v in outer.items() if v != o}
            assert q == {t for t, v in outer.items() if v == o}, "refit: query visits are not the outer-test fold"
        else:
            i = int(f["fit"][5:])
            inner = dict(zip(data.tags, folds[f"inner_fold_o{o}"].to_numpy()))
            assert all(inner[t] != i for t in tr) and all(inner[t] == i for t in q), f"{f['fit']}: wrong inner split"
            fit_set = {t for t, v in inner.items() if v not in (-1, i)}
        assert tr <= fit_set
        assert set(f["z_param_tags"]) == tr and not (set(f["z_param_tags"]) & q), "z parameters not from training rows"
        tags_k = f["wide_log"]["param_tags"]
        assert set(tags_k) == tr and not (set(tags_k) & q), f"{f['fit']}: param_tags not from the training visits"
        if f["target_param_tags"] is not None:
            mp = set(f["target_param_tags"])
            assert not (mp & q) and mp <= fit_set and tr <= mp, "target moments not from the fit's training visits"
