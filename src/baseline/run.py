"""Baseline entry point (CN vs MCI): five ML models x seven feature sets, the TabPFN mean-probability late fusion
and the MMSE cutoff rule, all with nested cross-validation and the shared bootstrap.

    PYTHONPATH=src .venv/bin/python -m baseline.run [--smoke] [--models ...] [--feature-sets ...] [--seeds ...] [--out DIR]

Output: output/baseline_YYYY_MM_DD_HHMM/<model>/<feature_set>/{predictions.csv, inner_oof.csv, selected_params.json,
metrics.json}, plus results.md and run_info.json.
"""
from __future__ import annotations
import argparse
import datetime as dt
import sys
import time
from pathlib import Path
from common.bootstrap import B_DEFAULT, load_draws
from common.cli import (SMOKE_B, SMOKE_SEEDS, add_common_args, check_no_duplicates, check_requirements,
                        load_env, make_logger, resolve_out)
from common.config import load_family_config, load_seeds
from common.data import FS_ORDER, load_cohort
from common.evaluate import finalize_cell, load_cell_metrics, write_cell
from common.make_folds import load_folds
from common.registry import DerivedContext
from common.report import write_results_md
from common.runinfo import write_run_info
from common.runner import Job
from common.scheduler import cell_dir_of, fingerprint_of, run_all, run_pool
from .models import REGISTRY

FAMILY = "baseline"
FAMILY_DIRS = ("baseline",)
CONFIG = Path(__file__).with_name("config.yaml")
MODULE = "baseline.run"


def _finalize(cell_dir: str, model: str, fs: str, seeds: list[int], expected: dict, n_boot: int):
    finalize_cell(Path(cell_dir), FAMILY, model, fs, seeds, expected, load_draws(B=n_boot))
    return f"{model}/{fs}", 0.0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--models", nargs="*", default=None, help="subset of the models enabled in config.yaml")
    ap.add_argument("--feature-sets", nargs="*", default=None, help="subset of task1 task3 task4 task5 task7 task9 fusion")
    add_common_args(ap)
    return ap


def _needed(cfg_models: list[dict], fsets: list[str], out: Path) -> None:
    """Derived models read cells of the same output directory: fail early if those will not exist."""
    scheduled = {(m["name"], fs) for m in cfg_models if REGISTRY[m["name"]].kind == "estimator" for fs in fsets}
    for m in cfg_models:
        spec = REGISTRY[m["name"]]
        if spec.kind != "derived":
            continue
        missing = [(sm, fs) for sm, fs in spec.requires(m)
                   if (sm, fs) not in scheduled and not (out / sm / fs / "predictions.csv").exists()]
        if missing:
            raise SystemExit(f"{m['name']} needs the cells {[f'{a}/{b}' for a, b in missing]}: add them to --models / "
                             "--feature-sets or run them first in the same --out")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    load_env()
    t_start = dt.datetime.now()
    t0 = time.time()
    check_no_duplicates(models=args.models, feature_sets=args.feature_sets, seeds=args.seeds)
    cfg = load_family_config(args.config)
    for m in cfg["models"]:
        if m["name"] not in REGISTRY:
            raise SystemExit(f"config model {m['name']!r} has no file in baseline/models (registered: {REGISTRY.names()})")
    if args.models:
        unknown = set(args.models) - {m["name"] for m in cfg["models"]}
        if unknown:
            raise SystemExit(f"models not enabled in config.yaml: {sorted(unknown)}")
        cfg["models"] = [m for m in cfg["models"] if m["name"] in args.models]
    seeds = args.seeds if args.seeds is not None else (SMOKE_SEEDS if args.smoke else load_seeds())
    fsets = args.feature_sets or cfg["feature_sets"]
    bad = set(fsets) - set(FS_ORDER)
    if bad:
        raise SystemExit(f"unknown feature sets {sorted(bad)}")
    n_boot = args.n_boot or (SMOKE_B if args.smoke else B_DEFAULT)
    needs_tabpfn = any(getattr(REGISTRY[m["name"]], "pipeline", "") == "tabpfn" for m in cfg["models"])
    if not args.skip_pin_check:
        check_requirements(needs_tabpfn)
    for s in seeds:
        load_folds(s)                                   # fail early if a fold file is missing
    draws = load_draws(B=n_boot)
    assert set(draws["subjects"]) == set(load_cohort()["ParticipantID"]), "bootstrap draws do not match the cohort"

    out = resolve_out(args.out, FAMILY, args.smoke)
    _needed(cfg["models"], fsets, out)
    out.mkdir(parents=True, exist_ok=True)
    log = make_logger()
    resume = not args.no_resume
    info = dict(family=FAMILY, module=MODULE, argv=sys.argv[1:] if argv is None else list(argv), seeds=seeds,
                smoke=args.smoke, start=t_start, family_dirs=FAMILY_DIRS)
    models = [m["name"] for m in cfg["models"]]
    extra = {"models": models, "feature_sets": fsets, "bootstrap": {"B": n_boot, "seed": draws["seed"]},
             "bootstrap_draws_sha256": draws["sha256"]}
    write_run_info(out, status="running", extra=extra, **info)

    jobs, cells = [], []
    for m in cfg["models"]:
        spec = REGISTRY[m["name"]]
        if spec.kind != "estimator":
            continue
        for fs in fsets:
            if resume and (out / m["name"] / fs / "metrics.json").exists():
                continue
            cells.append((out / m["name"] / fs, m["name"], fs))
            jobs += [Job(family=FAMILY, registry="baseline.models", model=m["name"], model_cfg=m, feature_set=fs,
                         seed=s, smoke=args.smoke) for s in seeds]
    log(f"baseline: {len(cells)} fitted cell(s), {len(jobs)} jobs, seeds {seeds}, out = {out.name}")

    try:
        fails = run_all(jobs, out, FAMILY_DIRS, args.workers, args.tabpfn_workers, args.tabpfn_threads, resume, log)
        if not fails:
            expected = {}
            for j in jobs:
                expected.setdefault(str(cell_dir_of(out, j)), {})[j.seed] = fingerprint_of(j, FAMILY_DIRS)
            fin = [(str(c), mn, fs, seeds, expected[str(c)], n_boot) for c, mn, fs in cells]
            fails = run_pool(_finalize, fin, args.workers, 1, False, log)
        if fails:
            (out / "failures.txt").write_text("\n\n".join(fails), encoding="utf-8")
            log(f"{len(fails)} job(s) failed; see failures.txt")
            write_run_info(out, status="failed", extra=extra, **info)
            return 1
        for m in cfg["models"]:
            spec = REGISTRY[m["name"]]
            if spec.kind != "derived":
                continue
            res = spec.run(DerivedContext(m, out, seeds, args.smoke))
            write_cell(out / m["name"] / spec.feature_set, FAMILY, m["name"], spec.feature_set, res["predictions"],
                       res["inner_oof"] if spec.has_validation else None, res["selected"], draws,
                       val_note=res.get("val_note"), footnote=res.get("footnote"))
            log(f"done {m['name']}/{spec.feature_set}")
        write_results_md(out, "Baseline: CN vs MCI", models, FS_ORDER + ["clinical"])
    except BaseException:
        write_run_info(out, status="failed", extra=extra, **info)
        raise
    extra["wall_seconds"] = round(time.time() - t0, 1)
    write_run_info(out, status="completed", extra=extra, **info)
    log(f"finished in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
