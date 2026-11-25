"""Parallel job scheduler. Two pools, never concurrent:
 - light pool: N workers, BLAS/OpenMP threads = 1 (scikit-learn models);
 - heavy pool: W workers x T torch threads (TabPFN).
One worker runs in-process (debugging and tests).
"""
from __future__ import annotations
import contextlib
import multiprocessing as mp
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable
from .evaluate import read_partial, write_partial
from .runinfo import job_fingerprint, scrub_paths
from .runner import Job, run_job

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@contextlib.contextmanager
def thread_env(n: int):
    """Set thread env vars while the pool is created, so spawned workers import numpy with them."""
    old = {k: os.environ.get(k) for k in THREAD_VARS}
    for k in THREAD_VARS:
        os.environ[k] = str(n)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _scoped_threads(n: int, torch_threads: bool):
    """In-process equivalent of the worker initialiser; limits are restored on exit."""
    prev_torch = None
    if torch_threads:
        import torch
        prev_torch = torch.get_num_threads()
    try:
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=n)
    except ImportError:
        limiter = contextlib.nullcontext()
    try:
        if prev_torch is not None:
            torch.set_num_threads(n)
        with limiter:
            yield
    finally:
        if prev_torch is not None:
            torch.set_num_threads(prev_torch)


def _init_worker(n_threads: int, torch_threads: bool) -> None:
    for k in THREAD_VARS:
        os.environ[k] = str(n_threads)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n_threads)
    except ImportError:
        pass
    if torch_threads:
        import torch
        torch.set_num_threads(n_threads)


def run_pool(fn: Callable, args_list: list[tuple], workers: int, n_threads: int, torch_threads: bool,
             log: Callable[[str], None]) -> list[str]:
    """Run fn(*args) for every tuple; returns the failure messages (all jobs are attempted)."""
    failures: list[str] = []
    if not args_list:
        return failures
    if workers <= 1:
        with _scoped_threads(n_threads, torch_threads):
            for a in args_list:
                try:
                    key, secs = fn(*a)
                    log(f"done {key} ({secs:.1f}s)")
                except Exception:  # noqa: BLE001
                    failures.append(f"{getattr(a[0], 'key', a[0])}\n{scrub_paths(traceback.format_exc())}")
                    log(f"FAILED {getattr(a[0], 'key', a[0])}")
        return failures
    with thread_env(n_threads):
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker,
                                 initargs=(n_threads, torch_threads)) as ex:
            futs = {ex.submit(fn, *a): a for a in args_list}
            for f in as_completed(futs):
                a = futs[f]
                try:
                    key, secs = f.result()
                    log(f"done {key} ({secs:.1f}s)")
                except Exception:  # noqa: BLE001
                    failures.append(f"{getattr(a[0], 'key', a[0])}\n{scrub_paths(traceback.format_exc())}")
                    log(f"FAILED {getattr(a[0], 'key', a[0])}")
    return failures


def cell_dir_of(out_root: Path, job: Job) -> Path:
    return Path(out_root) / job.model / job.feature_set


def fingerprint_of(job: Job, family_dirs: tuple[str, ...]) -> str:
    return job_fingerprint(job.family, job.model, job.model_cfg, job.feature_set, job.seed, job.smoke, job.n_threads,
                           family_dirs=family_dirs)


def execute_job(job: Job, out_root: str, family_dirs: tuple[str, ...]) -> tuple[str, float]:
    """Run one job and store its result as the seed partial of its cell."""
    cd = cell_dir_of(Path(out_root), job)
    cd.mkdir(parents=True, exist_ok=True)
    res = run_job(job)
    write_partial(res, cd, job.seed, fingerprint_of(job, family_dirs))
    return job.key, res["seconds"]


def is_done(job: Job, out_root: Path, family_dirs: tuple[str, ...]) -> bool:
    """True if the seed partial exists; raises if another configuration produced it."""
    part = read_partial(cell_dir_of(out_root, job), job.seed)
    if part is None:
        return False
    if part.get("fingerprint") != fingerprint_of(job, family_dirs):
        raise RuntimeError(f"{cell_dir_of(out_root, job)} seed {job.seed}: an existing partial result was produced by a "
                           "different configuration (grid, smoke flag, code or versions); use a new output directory")
    return True


def run_all(jobs: list[Job], out_root: Path, family_dirs: tuple[str, ...], workers: int, heavy_workers: int = 4,
            heavy_threads: int = 4, resume: bool = True, log: Callable[[str], None] = print) -> list[str]:
    """Light jobs first (workers x 1 thread), then heavy jobs (heavy_workers x heavy_threads)."""
    for j in jobs:                       # the thread count is part of the job fingerprint: set it before is_done
        if j.spec().heavy:
            j.n_threads = heavy_threads
    jobs = [j for j in jobs if not (resume and is_done(j, out_root, family_dirs))]
    order = lambda j: (j.feature_set != "fusion", j.model)  # noqa: E731  fusion (largest) first
    light = sorted([j for j in jobs if not j.spec().heavy], key=order)
    heavy = sorted([j for j in jobs if j.spec().heavy], key=order)
    log(f"{len(light)} light jobs, {len(heavy)} heavy jobs")
    fails = run_pool(execute_job, [(j, str(out_root), family_dirs) for j in light], workers, 1, False, log)
    fails += run_pool(execute_job, [(j, str(out_root), family_dirs) for j in heavy], heavy_workers, heavy_threads, True, log)
    return fails
