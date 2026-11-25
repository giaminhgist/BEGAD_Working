"""Stratified subject-level bootstrap: draws are generated once, saved, and shared by every model."""
from __future__ import annotations
from typing import Callable
import numpy as np
from .config import file_sha256, fold_dir

BOOT_SEED = 20261120
B_DEFAULT = 2000


def draws_path():
    return fold_dir() / "bootstrap_draws.npz"


def make_draws(cohort, B: int = B_DEFAULT, seed: int = BOOT_SEED) -> dict:
    """Draw subjects with replacement within strata (ever-MCI / CN-only) of the cohort.

    cohort needs ParticipantID and label (all 176 visits). Returns subjects (sorted), stratum (1 = ever-MCI),
    counts (B x n_subjects draw multiplicities m_j).
    """
    ever = cohort.groupby("ParticipantID")["label"].max()
    subjects = np.array(sorted(ever.index))
    stratum = ever.loc[subjects].to_numpy().astype(int)
    rng = np.random.default_rng(seed)
    counts = np.zeros((B, len(subjects)), dtype=np.int32)
    for s in (1, 0):                       # ever-MCI first, then CN: fixed order
        members = np.flatnonzero(stratum == s)
        draws = rng.integers(0, len(members), size=(B, len(members)))
        sub = np.zeros((B, len(members)), dtype=np.int32)
        np.add.at(sub, (np.arange(B)[:, None], draws), 1)
        counts[:, members] = sub
    return {"subjects": subjects, "stratum": stratum, "counts": counts, "seed": seed, "B": B}


def save_draws(d: dict, path=None) -> None:
    np.savez_compressed(path or draws_path(), subjects=d["subjects"], stratum=d["stratum"], counts=d["counts"],
                        seed=d["seed"], B=d["B"])


def load_draws(path=None, B: int | None = None) -> dict:
    path = path or draws_path()
    z = np.load(path, allow_pickle=False)
    d = {"subjects": z["subjects"], "stratum": z["stratum"], "counts": z["counts"],
         "seed": int(z["seed"]), "B": int(z["B"]), "sha256": file_sha256(path)}
    if B is not None:
        if B > d["B"]:
            raise ValueError(f"requested B={B} > stored {d['B']}")
        d["counts"], d["B"] = d["counts"][:B], B      # smoke: first B replicates
    return d


def replicate_index(sidx: np.ndarray, counts_b: np.ndarray) -> np.ndarray:
    """Row indices after replicating each row m_j times (np.repeat)."""
    return np.repeat(np.arange(len(sidx)), counts_b[sidx])


def bootstrap_thetas(groups: list[list[dict]], counts: np.ndarray,
                     metric_fn: Callable[[dict], dict], names: list[str]) -> dict[str, np.ndarray]:
    """theta*_b = mean over seeds of (mean over components of the metric on replicated rows).

    groups[s][f] = {"sidx": subject index per row, **arrays}; a component is the pooled test rows of a seed
    (one component) or one outer fold's training rows (validation). A NaN anywhere makes theta*_b NaN.
    """
    B = counts.shape[0]
    th = {n: np.full(B, np.nan) for n in names}
    for b in range(B):
        acc = {n: [] for n in names}
        for comps in groups:
            per_comp = {n: [] for n in names}
            for comp in comps:
                idx = replicate_index(comp["sidx"], counts[b])
                m = metric_fn({k: v[idx] for k, v in comp.items() if k != "sidx"})
                for n in names:
                    per_comp[n].append(m[n])
            for n in names:
                acc[n].append(np.mean(per_comp[n]))
        for n in names:
            th[n][b] = np.mean(acc[n])
    return th


def percentile_ci(theta: np.ndarray) -> dict:
    ok = np.isfinite(theta)
    n_nan = int((~ok).sum())
    if not ok.any():
        return {"lo": float("nan"), "hi": float("nan"), "n_nan": n_nan}
    lo, hi = np.percentile(theta[ok], [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "n_nan": n_nan}


def main() -> None:
    """Generate dataset/fold/bootstrap_draws.npz once; refuses to overwrite unless --force."""
    import argparse
    from .data import load_cohort
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    p = draws_path()
    if p.exists() and not a.force:
        raise SystemExit(f"{p.name} exists; draws are generated once (use --force to overwrite)")
    save_draws(make_draws(load_cohort()))
    print("saved", p.name)


if __name__ == "__main__":
    main()
