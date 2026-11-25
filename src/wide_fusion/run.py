"""wide_fusion entry point: cell wide_cf_w050 = TabPFN classifier + TabPFN regression of the clinical composite C,
both on the per-card features of a visit side by side (one model per block, one row per visit), fused on the z scale.

    PYTHONPATH=src .venv/bin/python -m wide_fusion.run [--smoke]

Jobs = (member, fold seed, outer fold), run with `run_failfast` (--tabpfn-workers x --tabpfn-threads, default 4 x 4 = 16
cores); a feature-subsampling STOP halts the run. Each job writes an atomic partial `<out>/_partial/<member>/seed<s>_o<o>.pkl`
with a fingerprint (config, member, seed, fold, smoke flag, code hash of src/wide_fusion + src/common, library versions,
input file hashes); a rerun resumes from matching partials and stops on a mismatch. Assembly writes members/<member>/, the
fused cell, the descriptive cells, contrasts.json, mmse_rules.json,
diagnostics.json, results.md and run_info.json.

Default output: output/wide_fusion_YYYY_MM_DD_HHMM/ (smoke: output/smoke_YYYY_MM_DD_HHMM/wide_fusion/).
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from common.bootstrap import B_DEFAULT, load_draws
from common.cli import (SMOKE_B, SMOKE_SEEDS, add_common_args, check_no_duplicates, check_requirements, load_env,
                        make_logger, resolve_out)
from common.config import DATA_CONFIG, file_sha256, fold_dir, load_data_config, load_seeds, repo_path
from common.data import merge_folds
from common.evaluate import compute_metrics, jsonable, write_cell
from common.latefusion import check_no_leakage
from common.make_folds import load_folds
from common.runinfo import job_fingerprint, scrub_paths, write_run_info
from common.runner import seed_everything
from . import combine as CB
from . import derived as DV
from . import mmse as MM
from . import report as R
from .contrast import paired_report
from .data import load_data
from .nested import audit_fit_log, run_member_job
from .tabcheck import SubsamplingStop

FAMILY = "wide_fusion"
FAMILY_DIRS = ("wide_fusion",)     # code hash = src/common + src/wide_fusion (non-test .py)
CONFIG = Path(__file__).with_name("config.yaml")
MODULE = "wide_fusion.run"


def load_cfg(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def model_factory(kind: str, n_estimators: int, n_threads: int):
    """seed -> frozen TabPFN; random_state = the fold seed (as src/common/tabpfn_frozen and cognitive_fusion)."""
    from common.tabpfn_frozen import TabPFNFrozen
    return lambda seed: TabPFNFrozen(task=kind, n_estimators=n_estimators, random_state=seed, n_threads=n_threads)


# ---------------------------------------------------------------------------------------------------
# fingerprints
def input_hashes(seeds) -> dict:
    cfg = load_data_config()
    out = {f"fold/seed_{s}.csv": file_sha256(fold_dir() / f"seed_{s}.csv") for s in seeds}
    for t in cfg["tasks"]:
        out[f"features/{t}"] = file_sha256(repo_path(cfg["paths"]["features_dir"]) / str(t) / "features.csv")
    out["cogdata"] = file_sha256(repo_path(cfg["paths"]["cogdata"]))
    out["common/data.yaml"] = file_sha256(DATA_CONFIG)
    out["features.md"] = file_sha256(repo_path("dataset/docs/features.md"))
    return out


def fingerprint(cfg: dict, member: str, seed: int, o: int, smoke: bool, n_threads: int) -> str:
    model_cfg = {k: cfg[k] for k in ("tasks", "blocks", "expected_common_columns", "targets", "tabpfn_feature_check")}
    return job_fingerprint(model_cfg, member, cfg["members"][member], seed, o, smoke, n_threads, input_hashes([seed]),
                           family_dirs=FAMILY_DIRS)


def partial_file(out: Path, member: str, seed: int, o: int) -> Path:
    return Path(out) / "_partial" / member / f"seed{seed}_o{o}.pkl"


# ---------------------------------------------------------------------------------------------------
# one job (runs in a worker process, or in-process with one worker)
_DATA: dict = {}


def _data(cfg: dict):
    key = json.dumps({k: cfg.get(k) for k in ("tasks", "targets")}, sort_keys=True)
    if key not in _DATA:
        _DATA[key] = load_data(cfg)
    return _DATA[key]


def execute(key: str, payload: dict) -> tuple[str, float]:
    cfg, member, seed, o = payload["cfg"], payload["member"], payload["seed"], payload["o"]
    seed_everything(seed)
    data = _data(cfg)
    folds = merge_folds(data.cohort, load_folds(seed))
    res = run_member_job(member, cfg["members"][member], seed, o, data, folds, cfg, model_factory, payload["n_threads"])
    audit_fit_log(res["fit_log"], data, folds)
    res["fingerprint"] = payload["fingerprint"]
    p = Path(payload["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(res, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    return key, res["seconds"]


def job_order(cfg: dict) -> list[str]:
    """Most expensive members first (n_estimators), then by name."""
    m = cfg["members"]
    return sorted(m, key=lambda k: (-m[k]["n_estimators"], k))


def run_jobs(cfg, out, seeds, smoke, resume, workers, n_threads, log) -> list[str]:
    args = []
    members = job_order(cfg)
    for member in members:
        for s in seeds:
            for o in range(4):
                fp = fingerprint(cfg, member, s, o, smoke, n_threads)
                p = partial_file(out, member, s, o)
                if p.exists() and resume:
                    with open(p, "rb") as f:
                        if pickle.load(f).get("fingerprint") != fp:
                            raise SystemExit(f"{member} seed {s} fold {o}: partial result was produced by a different "
                                             "configuration, code or input; use a new --out")
                    continue
                args.append((f"{member}/seed{s}/o{o}", {"cfg": cfg, "member": member, "seed": s, "o": o,
                                                        "n_threads": n_threads, "path": str(p), "fingerprint": fp}))
    log(f"{len(args)} jobs to run ({len(members) * len(seeds) * 4 - len(args)} resumed)")
    return run_failfast(execute, args, workers, n_threads, log)


def run_failfast(fn, args_list, workers: int, n_threads: int, log) -> list[str]:
    """Like common.scheduler.run_pool, but a SubsamplingStop halts the run: no further job is started (pending futures are
    cancelled, in-process jobs are not run) and the failures so far are returned. Other failures do not stop the others."""
    import multiprocessing as mp
    import traceback
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from common.scheduler import _init_worker, _scoped_threads, thread_env
    fails: list[str] = []

    def fail(a, e):
        fails.append(f"{a[0]}\n{scrub_paths(''.join(traceback.format_exception(type(e), e, e.__traceback__)))}")
        log(f"FAILED {a[0]}" + (" -- STOP (subsampling): halting, remaining jobs not run" if isinstance(e, SubsamplingStop) else ""))

    if not args_list:
        return fails
    if workers <= 1:
        with _scoped_threads(n_threads, True):
            for a in args_list:
                try:
                    key, secs = fn(*a)
                    log(f"done {key} ({secs:.1f}s)")
                except Exception as e:  # noqa: BLE001
                    fail(a, e)
                    if isinstance(e, SubsamplingStop):
                        break
        return fails
    with thread_env(n_threads):
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                                 initargs=(n_threads, True)) as ex:
            futs = {ex.submit(fn, *a): a for a in args_list}
            for f in as_completed(futs):
                if f.cancelled():
                    continue
                try:
                    key, secs = f.result()
                    log(f"done {key} ({secs:.1f}s)")
                except Exception as e:  # noqa: BLE001
                    fail(futs[f], e)
                    if isinstance(e, SubsamplingStop):
                        for g in futs:
                            g.cancel()
    return fails


# ---------------------------------------------------------------------------------------------------
# references
def _summary(cell: Path):
    f = cell / "metrics.json"
    return json.loads(f.read_text(encoding="utf-8"))["summary"]["test_auc"] if f.exists() else None


def _ref_summary(cell: Path, pred: pd.DataFrame, seeds, draws: dict):
    """Saved summary when it was computed on exactly these seeds and draws (the full run); otherwise the same metric
    recomputed on the requested seeds with the shared draws, so the displayed AUCs explain the displayed deltas."""
    f = cell / "metrics.json"
    m = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    if (sorted(m.get("seeds", [])) == sorted(seeds) and m.get("bootstrap", {}).get("draws_sha256") == draws.get("sha256")
            and m.get("bootstrap", {}).get("B") == draws.get("B")):
        return _summary(cell)
    return compute_metrics(pred, None, draws)["summary"]["test_auc"]


def check_references(cfg: dict, seeds, draws: dict) -> dict:
    refs = {}
    for name, rel in cfg["references"].items():
        d = repo_path(rel)
        info_p = d.parent.parent / "run_info.json"
        if not (d / "predictions.csv").exists() or not info_p.exists():
            raise SystemExit(f"reference cell {rel} not found")
        info = json.loads(info_p.read_text(encoding="utf-8"))
        if info.get("status") != "completed" or info.get("smoke"):
            raise SystemExit(f"reference {rel}: the run must be a completed full run")
        p = pd.read_csv(d / "predictions.csv")
        p = p[p.seed.isin(seeds)].reset_index(drop=True)
        if sorted(p.seed.unique()) != sorted(seeds):
            raise SystemExit(f"reference {rel}: seeds {sorted(p.seed.unique())} do not cover {seeds}")
        refs[name] = {"path": rel, "pred": p, "code_sha256": info.get("code_sha256"), "test_auc": _ref_summary(d, p, seeds, draws)}
    return refs


def check_saved_run(rel: str, seeds, what: str) -> dict:
    d = repo_path(rel)
    info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
    if info.get("status") != "completed" or info.get("smoke"):
        raise SystemExit(f"the {what} output must be a completed full run")
    if not set(seeds) <= set(info.get("seeds", [])):
        raise SystemExit(f"seeds {seeds} are not all in the {what} run")
    return info


# ---------------------------------------------------------------------------------------------------
# assembly
def load_parts(cfg, out, seeds, smoke, n_threads) -> dict:
    parts = {}
    for m in cfg["members"]:
        for s in seeds:
            for o in range(4):
                with open(partial_file(out, m, s, o), "rb") as f:
                    r = pickle.load(f)
                if r.get("fingerprint") != fingerprint(cfg, m, s, o, smoke, n_threads):
                    raise RuntimeError(f"{m} seed {s} fold {o}: stale partial")
                parts[(m, s, o)] = r
    return parts


def member_frames(parts, member, seeds):
    keys = [(member, s, o) for s in seeds for o in range(4)]
    cat = lambda k: pd.concat([parts[x][k] for x in keys], ignore_index=True)  # noqa: E731
    pred, oof = cat("predictions"), cat("inner_oof")
    check_no_leakage(pred, oof)
    return pred, oof, cat("rows_predictions"), cat("rows_inner_oof")


def write_member(out: Path, parts, member, seeds, frames) -> None:
    d = out / "members" / member
    d.mkdir(parents=True, exist_ok=True)
    pred, oof, rp, ro = frames
    pred.to_csv(d / "predictions.csv", index=False)
    oof.to_csv(d / "inner_oof.csv", index=False)
    rp.to_csv(d / "rows_predictions.csv", index=False)
    ro.to_csv(d / "rows_inner_oof.csv", index=False)
    log = [{k: (len(v) if isinstance(v, list) else v) for k, v in f.items() if k not in ("wide_log", "tabpfn_check")}
           for s in seeds for o in range(4) for f in parts[(member, s, o)]["fit_log"]]
    pd.DataFrame(log).rename(columns={"train": "n_train_visits", "query": "n_query_visits", "z_param_tags": "n_z_param_visits",
                                      "target_param_tags": "n_target_param_visits"}).to_csv(d / "fit_log.csv", index=False)
    tp = {str(s): {str(o): parts[(member, s, o)]["target_params"] for o in range(4)} for s in seeds}
    if any(v is not None for x in tp.values() for v in x.values()):
        (d / "target_params.json").write_text(json.dumps(jsonable(tp), indent=1), encoding="utf-8")


def rho(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(x[ok], y[ok])[0])


def _stats(v) -> dict:
    v = np.asarray(v, float)
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()),
            "max": float(v.max())}


def diagnostics(cfg, parts, frames, seeds) -> dict:
    d = {"inner_spearman_rbar_target": {}, "context_rows": {}, "seconds_per_fit": {}, "job_seconds": {},
         "target_excluded_visits": {}}
    for m, spec in cfg["members"].items():
        logs = [f for s in seeds for o in range(4) for f in parts[(m, s, o)]["fit_log"]]
        d["job_seconds"][m] = _stats([parts[(m, s, o)]["seconds"] for s in seeds for o in range(4)])
        cr, sp = {}, {}
        for b in sorted({f["block"] for f in logs}):
            cr[b], sp[b] = {}, {}
            for typ in ("inner", "refit"):
                fs = [f for f in logs if f["block"] == b and (f["fit"] == "refit") == (typ == "refit")]
                cr[b][typ] = {**_stats([f["n_train_rows"] for f in fs]), "query_rows": _stats([f["n_query_rows"] for f in fs])}
                sp[b][typ] = {"fit": _stats([f["fit_seconds"] for f in fs]), "predict": _stats([f["predict_seconds"] for f in fs])}
        d["context_rows"][m], d["seconds_per_fit"][m] = cr, sp
        if spec["kind"] == "regression":
            oof = frames[m][1]
            per = {int(s): rho(oof.loc[oof.seed == s, "y_score"], oof.loc[oof.seed == s, "y_target"]) for s in seeds}
            v = [x for x in per.values() if np.isfinite(x)]
            tname = spec["target"]
            d["inner_spearman_rbar_target"][m] = {"target": tname, "per_seed": per, "mean": float(np.mean(v)) if v else None,
                                                  "note": f"inner-OOF r̄ vs {tname} on the outer-training scale, pooled over outer folds"}
            d["target_excluded_visits"][m] = sorted({t for s in seeds for o in range(4)
                                                     for t in parts[(m, s, o)]["target_params"]["refit"]["excluded"]})
    return d


def context_summary(diag: dict) -> dict:
    return {m: {b: {t: {"min": v[t]["min"], "median": v[t]["median"], "max": v[t]["max"]} for t in v} for b, v in x.items()}
            for m, x in diag["context_rows"].items()}


def compute_contrasts(cfg, cells, refs, draws, seeds) -> dict:
    out = {}
    for a, b in cfg["contrasts"]:
        pa = cells[a]
        if b.startswith("ref:"):
            ref = b[4:]
            pb = refs[ref]["pred"]
        else:
            ref, pb = b, cells[b]
        out[f"{a} − {ref}"] = {"cell": a, "reference": ref, **paired_report(pa, pb, draws, seeds)}
    return out


# ---------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    add_common_args(ap)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    load_env()
    t_start, t0 = dt.datetime.now(), time.time()
    check_no_duplicates(seeds=args.seeds)
    cfg = load_cfg(args.config)
    army_seed = int(os.environ.get("ARMY_SEED") or 0)
    seed_everything(army_seed)            # global RNGs only; every fit is seeded with its fold seed
    seeds = args.seeds if args.seeds is not None else (SMOKE_SEEDS if args.smoke else load_seeds())
    n_boot = args.n_boot or (SMOKE_B if args.smoke else B_DEFAULT)
    if not args.skip_pin_check:
        check_requirements(True)
    for s in seeds:
        load_folds(s)
    draws = load_draws(B=n_boot)
    refs = check_references(cfg, seeds, draws)
    binfo = check_saved_run(cfg["baseline_classifier"]["run"], seeds, "baseline")
    check_saved_run(cfg["saved_regressors"]["run"], seeds, "cognitive_fusion")
    out = resolve_out(args.out, FAMILY, args.smoke)
    out.mkdir(parents=True, exist_ok=True)
    log = make_logger()
    workers, n_threads = args.tabpfn_workers, args.tabpfn_threads
    fused, desc = list(cfg["fused_cells"]), list(cfg["descriptive_cells"])
    info = dict(family=FAMILY, module=MODULE, argv=sys.argv[1:] if argv is None else list(argv), seeds=seeds,
                smoke=args.smoke, start=t_start, family_dirs=FAMILY_DIRS)
    extra = {"code_sha256_scope": "non-test .py of src/wide_fusion + src/common", "config_sha256": file_sha256(args.config),
             "army_seed": army_seed, "input_sha256": input_hashes(seeds), "fused_cells": fused,
             "descriptive_cells": desc, "members": cfg["members"],
             "baseline_classifier": {"run": cfg["baseline_classifier"]["run"], "code_sha256": binfo.get("code_sha256")},
             "references": {k: {"path": v["path"], "code_sha256": v["code_sha256"]} for k, v in refs.items()},
             "bootstrap": {"B": n_boot, "seed": draws["seed"]}, "bootstrap_draws_sha256": draws["sha256"],
             "tabpfn_workers": workers, "torch_threads_per_worker": n_threads}
    write_run_info(out, status="running", extra=extra, **info)
    try:
        log(f"{FAMILY}: seeds {seeds}, smoke = {args.smoke}, out = {out.name}")
        for p in (out / "_partial").rglob("*.tmp") if (out / "_partial").exists() else []:
            p.unlink()
        fails = run_jobs(cfg, out, seeds, args.smoke, not args.no_resume, workers, n_threads, log)
        if fails:
            (out / "failures.txt").write_text("\n".join(fails), encoding="utf-8")
            raise SystemExit(f"{len(fails)} job(s) failed; see failures.txt")
        t_jobs = time.time() - t0
        parts = load_parts(cfg, out, seeds, args.smoke, n_threads)
        data = _data(cfg)
        folds = {s: merge_folds(data.cohort, load_folds(s)) for s in seeds}
        for (m, s, o), r in parts.items():
            audit_fit_log(r["fit_log"], data, folds[s])

        frames, member_rows, member_oof_rows = {}, {}, {}
        for m in cfg["members"]:
            fr = member_frames(parts, m, seeds)
            write_member(out, parts, m, seeds, fr)
            frames[m] = (fr[0], fr[1])
            member_rows[m] = fr[2]
            member_oof_rows[m] = fr[3]
        saved = DV.load_saved(cfg, seeds)
        extra["saved_member_identity"] = DV.identity_check_saved(saved, cfg, seeds, cfg["w"], cfg["clip"])
        extra["saved_member_sha256"] = saved["sha256"]
        log(f"saved members reproduce cogfusion_w050: {extra['saved_member_identity']}")
        kinds = {"cls": cfg["fused_cells"]["wide_cf_w050"]["classifier"], "reg": cfg["fused_cells"]["wide_cf_w050"]["regressor"]}
        frames.update(DV.build_derived(cfg, saved, {m: (member_rows[m], member_oof_rows[m]) for m in cfg["members"]}, kinds))
        wide_full, extra["tabpfn_feature_check"] = DV.wide_logs(cfg, parts, seeds)
        for m, rows in wide_full.items():
            (out / "members" / m / "wide_fit_log.json").write_text(json.dumps(jsonable(rows)), encoding="utf-8")

        cells, metrics = {}, {}
        for c, spec in cfg["fused_cells"].items():
            (pred, oof), sel = CB.fuse(frames[spec["classifier"]], frames[spec["regressor"]], cfg["w"], cfg["clip"])
            cells[c] = pred
            metrics[c] = write_cell(out / c / "fusion", FAMILY, c, "fusion", pred, oof,
                                    {"w": cfg["w"], "clip": cfg["clip"], "members": dict(spec),
                                     "z_scale": "per (seed, outer fold), inner-OOF rows only", "per_fold": sel}, draws)
        for c, m in cfg["descriptive_cells"].items():
            pred, oof = CB.single_cell(frames[m])
            cells[c] = pred
            metrics[c] = write_cell(out / c / "fusion", FAMILY, c, "fusion", pred, oof,
                                    {"member": m, "spec": cfg["members"].get(m, "derived from saved or new members (no fit)"),
                                     "score": "member score as is"}, draws)
        t_cells = time.time() - t0 - t_jobs
        contrasts = compute_contrasts(cfg, cells, refs, draws, seeds)
        (out / "contrasts.json").write_text(json.dumps(jsonable(contrasts), indent=1), encoding="utf-8")
        mmse = MM.mmse_rows(cfg["mmse_rules"]["baseline_run"], seeds, draws)
        (out / "mmse_rules.json").write_text(json.dumps(jsonable(mmse), indent=1), encoding="utf-8")
        diag = diagnostics(cfg, parts, frames, seeds)
        diag["reference_test_auc"] = {k: v["test_auc"] for k, v in refs.items()}
        diag["tabpfn_feature_check"] = extra["tabpfn_feature_check"]
        (out / "diagnostics.json").write_text(json.dumps(jsonable(diag), indent=1), encoding="utf-8")
        auc = {c: metrics[c]["summary"]["test_auc"] for c in metrics}
        R.write_results(out, contrasts, mmse, {k: v["path"] for k, v in refs.items()},
                             {k: v["test_auc"] for k, v in refs.items()}, fused, desc)
        for c in fused + desc:
            log(f"{c}: test AUC {auc[c]['mean']:.6f}" + (" (smoke)" if args.smoke else ""))
        extra["context_rows"] = context_summary(diag)
    except BaseException:
        write_run_info(out, status="failed", extra=extra, **info)
        raise
    extra.update(wall_seconds=round(time.time() - t0, 1), stage_seconds={"jobs": round(t_jobs, 1),
                 "cells_and_bootstrap": round(t_cells, 1)},
                 job_seconds_by_member={m: round(sum(parts[(m, s, o)]["seconds"] for s in seeds for o in range(4)), 1)
                                        for m in cfg["members"]})
    write_run_info(out, status="completed", extra=extra, **info)
    cleanup_partials(out, log)
    log(f"finished in {time.time() - t0:.0f}s")
    return 0


def cleanup_partials(out: Path, log) -> None:
    root = Path(out) / "_partial"
    if not root.exists():
        return
    try:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in (".pkl", ".tmp"):
                p.unlink()
        for d in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            d.rmdir()
        root.rmdir()
    except OSError as e:
        log(f"partial cleanup incomplete ({type(e).__name__}); remaining files under _partial/ can be deleted by hand")


if __name__ == "__main__":
    raise SystemExit(main())
