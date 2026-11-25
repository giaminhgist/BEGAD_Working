"""Command-line pieces shared by the family entry points (`baseline.run`, `cognitive_fusion.run`)."""
from __future__ import annotations
import argparse
import datetime as dt
import importlib.metadata as md
import os
import re
from pathlib import Path
from .config import REPO_ROOT
from .runinfo import output_dir_name

SMOKE_SEEDS = [0]
SMOKE_B = 200


def load_env() -> None:
    """Read TABPFN_TOKEN from the repository .env (the value is never printed)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="default: all seeds in dataset/fold/seeds.yaml")
    ap.add_argument("--smoke", action="store_true", help="code check: seed 0, reduced grid, B = 200 (not a result)")
    ap.add_argument("--out", default=None, help="output directory (default output/<family>_YYYY_MM_DD_HHMM)")
    ap.add_argument("--workers", type=int, default=16, help="light pool size (scikit-learn models)")
    ap.add_argument("--tabpfn-workers", type=int, default=4)
    ap.add_argument("--tabpfn-threads", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=None, help="bootstrap replicates (default 2000; smoke 200)")
    ap.add_argument("--no-resume", action="store_true", help="recompute even if partial results exist")
    ap.add_argument("--skip-pin-check", action="store_true", help="do not compare installed versions with requirements.txt")


def check_no_duplicates(**flags) -> None:
    for flag, vals in flags.items():
        if vals and len(set(vals)) != len(vals):
            dup = sorted({v for v in vals if vals.count(v) > 1})
            raise SystemExit(f"duplicate values for --{flag.replace('_', '-')}: {dup}")


def resolve_out(out: str | None, family: str, smoke: bool, sibling_of: Path | None = None) -> Path:
    """--out if given; else output/<family>_STAMP, or for smoke output/smoke_STAMP/<family>. A smoke run that reads
    another family's smoke output (`sibling_of`, e.g. .../smoke_STAMP/baseline) writes next to it."""
    if out:
        p = Path(out)
        return p if p.is_absolute() else Path.cwd() / p
    if smoke and sibling_of is not None and sibling_of.parent.name.startswith("smoke_"):
        return sibling_of.parent / family
    return REPO_ROOT / "output" / output_dir_name(family, smoke, dt.datetime.now())


def check_requirements(needs_tabpfn: bool) -> None:
    """Installed versions must equal the pins of requirements.txt (the TabPFN pins only if TabPFN is used)."""
    req = REPO_ROOT / "requirements.txt"
    for line in req.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)$", line.strip())
        if not m:
            continue
        pkg, want = m.groups()
        if pkg in ("tabpfn", "torch") and not needs_tabpfn:
            continue
        have = md.version(pkg)
        if have != want:
            raise RuntimeError(f"version pin violated: {pkg}=={have}, requirements.txt pins {want}")
    if needs_tabpfn and os.environ.get("TABPFN_MODEL_CACHE_SIZE", "0") not in ("", "0"):
        raise RuntimeError("TABPFN_MODEL_CACHE_SIZE must be unset or 0 (a cached model could be mutated)")


def make_logger():
    return lambda m: print(f"[{dt.datetime.now():%H:%M:%S}] {m}", flush=True)
