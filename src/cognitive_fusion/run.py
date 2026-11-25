"""cognitive_fusion entry point: per-task TabPFN regressors of a training-only clinical composite, fused (z scale) with
the baseline mean-probability classifier score read from a finished baseline output directory.

    PYTHONPATH=src .venv/bin/python -m cognitive_fusion.run --baseline-out output/baseline_YYYY_MM_DD_HHMM [--smoke]

Arms: cogfusion_w000, _w025, _w050 (primary), _w075, _w100 (w = weight of the regressor score). Output:
output/cognitive_fusion_YYYY_MM_DD_HHMM/<arm>/fusion/{predictions.csv, inner_oof.csv, selected_params.json,
metrics.json}, regression/task<t>/, contrasts.json, diagnostics.json, results.md, run_info.json.
The clinical scores are used only as regression targets of training rows.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import pickle
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from common.bootstrap import B_DEFAULT, load_draws
from common.cli import (SMOKE_B, SMOKE_SEEDS, add_common_args, check_no_duplicates, check_requirements, load_env,
                        make_logger, resolve_out)
from common.config import REPO_ROOT, load_data_config, load_seeds, repo_path
from common.data import FS_ORDER, applicable_columns, load_analysis_data, merge_folds
from common.evaluate import jsonable, write_cell
from common.latefusion import TASKS, load_task_cells
from common.make_folds import load_folds
from common.report import write_results_md
from common.runinfo import job_fingerprint, write_run_info
from common.runner import seed_everything
from common.scheduler import run_pool
from . import combine as C
from .composite import SCORE_COLS, sign_vector
from .contrast import paired_report
from .regress import audit_fit_log, audit_inputs, run_regression_job

FAMILY = "cognitive_fusion"
FAMILY_DIRS = ("cognitive_fusion",)
CONFIG = Path(__file__).with_name("config.yaml")
MODULE = "cognitive_fusion.run"


def load_cfg(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def baseline_dir(cfg: dict, cli_value: str | None) -> Path:
    v = cli_value or cfg.get("baseline_out")
    if not v:
        raise SystemExit("set baseline_out in config.yaml or pass --baseline-out output/baseline_YYYY_MM_DD_HHMM")
    d = repo_path(v)
    if not (d / "run_info.json").exists():
        raise SystemExit(f"{v}: not a baseline output directory (run_info.json missing)")
    return d


def check_baseline(d: Path, smoke: bool, seeds: list[int]) -> dict:
    info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
    if info.get("status") != "completed":
        raise SystemExit("the baseline output must be a completed run")
    if info.get("smoke") and not smoke:
        raise SystemExit("the baseline output is a smoke run; use a full baseline run or pass --smoke")
    if not set(seeds) <= set(info.get("seeds", [])):
        raise SystemExit(f"seeds {seeds} are not all in the baseline run ({info.get('seeds')})")
    return info


def _cogdata() -> pd.DataFrame:
    cfg = load_data_config()
    c = pd.read_csv(repo_path(cfg["paths"]["cogdata"]), encoding="utf-8")
    c[cfg["cogdata_key"]] = c[cfg["cogdata_key"]].astype(str).str.strip()
    return c.set_index(cfg["cogdata_key"])


def load_inputs(task: int):
    """(eye features X, columns, clinical scores S, labels y, cohort) of one task."""
    X_df, y, _, cohort = load_analysis_data(f"task{task}")
    return X_df.to_numpy(dtype=float), list(X_df.columns), cohort[SCORE_COLS].to_numpy(dtype=float), np.asarray(y), cohort


def make_tabpfn(cfg: dict):
    from common.tabpfn_frozen import TabPFNFrozen
    r = cfg["regressor"]
    return lambda seed: TabPFNFrozen(task="regression", n_estimators=r["n_estimators"], random_state=seed,
                                     n_threads=cfg["n_threads"])


def fingerprint(cfg: dict, task: int, seed: int) -> str:
    return job_fingerprint("regression", cfg["regressor"], cfg["sign"], cfg["min_scores"], task, seed,
                           cfg["n_threads"], family_dirs=FAMILY_DIRS)


def execute(key: str, payload: dict) -> tuple[str, float]:
    """One regression job (task, seed); the result goes to <out>/_partial/<key>.pkl (atomic)."""
    cfg, task, seed = payload["cfg"], payload["task"], payload["seed"]
    seed_everything(seed)
    X, cols, S, y, cohort = load_inputs(task)
    cdata = _cogdata()
    audit_inputs(cols, applicable_columns(task), X, cdata.loc[cohort["RealTag"], "Ikä"].to_numpy(float), S,
                 list(cdata.columns))
    folds = merge_folds(cohort, load_folds(seed))
    ids = cohort[["RealTag", "ParticipantID", "IsFollowup"]]
    res = run_regression_job(task, seed, X, cols, S, y, ids, folds, make_tabpfn(cfg), sign_vector(cfg["sign"]),
                             cfg["min_scores"])
    audit_fit_log(res["fit_log"], ids, folds)
    res["fingerprint"] = fingerprint(cfg, task, seed)
    p = Path(payload["out"]) / "_partial" / f"{key}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(res, f)
    os.replace(tmp, p)
    return key, res["seconds"]


def run_jobs(cfg: dict, out: Path, seeds: list[int], resume: bool, workers: int, n_threads: int, log) -> list[str]:
    cfg = {**cfg, "n_threads": n_threads}
    args = []
    for seed in seeds:
        for t in TASKS:
            key = f"task{t}_seed{seed}"
            p = out / "_partial" / f"{key}.pkl"
            if p.exists() and resume:
                with open(p, "rb") as f:
                    if pickle.load(f).get("fingerprint") != fingerprint(cfg, t, seed):
                        raise SystemExit(f"{p.name}: partial result produced by a different configuration; use a new --out")
                continue
            args.append((key, {"cfg": cfg, "task": t, "seed": seed, "out": str(out)}))
    return run_pool(execute, args, workers, n_threads, True, log)


def assemble_regression(out: Path, seeds: list[int]):
    """Per-task regression cells (predictions, inner OOF, selected params); returns ({t: pred}, {t: oof}, parts)."""
    parts = {(t, s): pickle.load(open(out / "_partial" / f"task{t}_seed{s}.pkl", "rb")) for s in seeds for t in TASKS}
    pred, oof, params = {}, {}, None
    for t in TASKS:
        cell = out / "regression" / f"task{t}"
        cell.mkdir(parents=True, exist_ok=True)
        pred[t] = pd.concat([parts[(t, s)]["predictions"] for s in seeds], ignore_index=True)
        oof[t] = pd.concat([parts[(t, s)]["inner_oof"] for s in seeds], ignore_index=True)
        pred[t].to_csv(cell / "predictions.csv", index=False)
        oof[t].to_csv(cell / "inner_oof.csv", index=False)
        (cell / "selected_params.json").write_text(
            json.dumps(jsonable({s: parts[(t, s)]["selected"] for s in seeds}), indent=1), encoding="utf-8")
        cp = {s: parts[(t, s)]["composite_params"] for s in seeds}
        if params is not None and cp != params:
            raise AssertionError("composite parameters differ between tasks")   # they depend on the rows only
        params = cp
    (out / "regression" / "composite_params.json").write_text(json.dumps(jsonable(params), indent=1), encoding="utf-8")
    return pred, oof, parts


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--baseline-out", default=None, help="finished baseline output directory (default: config.yaml)")
    add_common_args(ap)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    load_env()
    t_start, t0 = dt.datetime.now(), time.time()
    check_no_duplicates(seeds=args.seeds)
    cfg = load_cfg(args.config)
    bdir = baseline_dir(cfg, args.baseline_out)
    binfo = json.loads((bdir / "run_info.json").read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds is not None else (SMOKE_SEEDS if args.smoke else binfo["seeds"])
    check_baseline(bdir, args.smoke, seeds)
    n_boot = args.n_boot or (SMOKE_B if args.smoke else B_DEFAULT)
    if not args.skip_pin_check:
        check_requirements(True)
    for s in seeds:
        load_folds(s)
    draws = load_draws(B=n_boot)
    out = resolve_out(args.out, FAMILY, args.smoke, bdir)
    out.mkdir(parents=True, exist_ok=True)
    log = make_logger()
    ws = cfg["w_grid"]
    arms = {w: C.arm_name(w) for w in ws}
    if cfg["primary"] not in arms.values():
        raise SystemExit(f"primary arm {cfg['primary']!r} is not in w_grid")
    info = dict(family=FAMILY, module=MODULE, argv=sys.argv[1:] if argv is None else list(argv), seeds=seeds,
                smoke=args.smoke, start=t_start, family_dirs=FAMILY_DIRS)
    extra = {"baseline_out": str(bdir.relative_to(REPO_ROOT)) if bdir.is_relative_to(REPO_ROOT) else bdir.name,
             "baseline_code_sha256": binfo.get("code_sha256"), "arms": list(arms.values()), "primary": cfg["primary"], "reference": cfg["reference"],
             "bootstrap": {"B": n_boot, "seed": draws["seed"]}, "bootstrap_draws_sha256": draws["sha256"]}
    write_run_info(out, status="running", extra=extra, **info)
    try:
        log(f"cognitive_fusion: seeds {seeds}, baseline = {bdir.name}, out = {out.name}")
        fails = run_jobs(cfg, out, seeds, not args.no_resume, args.tabpfn_workers, args.tabpfn_threads, log)
        if fails:
            (out / "failures.txt").write_text("\n".join(fails), encoding="utf-8")
            raise SystemExit(f"{len(fails)} regression job(s) failed; see failures.txt")
        t_fit = time.time() - t0
        reg_pred, reg_oof, parts = assemble_regression(out, seeds)

        src = cfg["source"]
        pred_t, oof_t = load_task_cells(bdir, src["model"], src["tasks"], seeds)
        cells, aux = C.fusion_arms(pred_t, oof_t, reg_pred, reg_oof, ws, cfg["clip"])

        def baseline_cell(spec: dict):
            f = bdir / spec["model"] / spec["feature_set"] / "predictions.csv"
            if not f.exists():
                return None
            d = pd.read_csv(f)
            return d[d.seed.isin(seeds)].reset_index(drop=True)

        ref, early = baseline_cell(cfg["reference"]), baseline_cell(cfg["early_fusion"])
        if ref is not None:                  # the classifier member equals the baseline mean-probability cell exactly
            refo = pd.read_csv(bdir / cfg["reference"]["model"] / cfg["reference"]["feature_set"] / "inner_oof.csv")
            refo = refo[refo.seed.isin(seeds)]
            for (b, p), r, keys in ((aux["p_te"], ref, C.KEYS_P), (aux["p_oo"], refo, C.KEYS_O)):
                rr = r.set_index(keys)["y_score"].reindex(pd.MultiIndex.from_frame(b[keys])).to_numpy()
                d = float(np.max(np.abs(rr - p)))
                if d != 0.0:
                    raise AssertionError(f"classifier member differs from the baseline reference cell (max |d| = {d})")
        else:
            log("reference cell not found in the baseline output: identity check and paired contrasts skipped")

        contrasts = {}
        for w, (pred, oof) in cells.items():
            name = arms[w]
            m = write_cell(out / name / "fusion", FAMILY, name, "fusion", pred, oof,
                           {"w": w, "z_scale": "per (seed, outer fold), inner-OOF rows only", "clip": cfg["clip"]}, draws)
            line = f"{name}: test AUC {m['summary']['test_auc']['mean']:.4f}, val AUC {m['summary']['val_auc']['mean']:.4f}"
            if ref is not None:
                c = paired_report(pred, ref, draws, seeds)
                if name == cfg["primary"] and early is not None:
                    c["vs_early_fusion"] = paired_report(pred, early, draws, seeds)
                contrasts[name] = c
                line += f", delta vs {cfg['reference']['model']}: {c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"
            log(line + (" (smoke)" if args.smoke else ""))
        if contrasts:
            (out / "contrasts.json").write_text(json.dumps(jsonable(contrasts), indent=1), encoding="utf-8")
        diag = {"inner_spearman_rbar_C": C.inner_spearman(reg_oof[TASKS[0]], aux["p_oo"][0], aux["r_oo"]),
                "fit_seconds_mean_by_task": {t: float(np.mean([x for s in seeds for o in parts[(t, s)]["selected"].values()
                                                               for x in o["fit_seconds"] + [o["refit_seconds"]]]))
                                             for t in TASKS}}
        (out / "diagnostics.json").write_text(json.dumps(jsonable(diag), indent=1), encoding="utf-8")
        write_results_md(out, "cognitive_fusion: CN vs MCI", list(arms.values()), FS_ORDER,
                         contrasts=contrasts or None, reference=cfg["reference"])
        for p in (out / "_partial").glob("*.pkl"):
            p.unlink()
        (out / "_partial").rmdir()
    except BaseException:
        write_run_info(out, status="failed", extra=extra, **info)
        raise
    extra.update(wall_seconds=round(time.time() - t0, 1), regression_stage_seconds=round(t_fit, 1))
    write_run_info(out, status="completed", extra=extra, **info)
    log(f"finished in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
