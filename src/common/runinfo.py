"""Output directory naming, code hash, package versions and the run_info.json writer."""
from __future__ import annotations
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
from functools import lru_cache
from pathlib import Path
from .config import REPO_ROOT, file_sha256, fold_dir
from .evaluate import jsonable

PACKAGES = ("scikit-learn", "tabpfn", "torch", "numpy", "pandas", "scipy", "pyyaml")


def output_dir_name(family: str, smoke: bool = False, now: dt.datetime | None = None) -> str:
    """`{family}_YYYY_MM_DD_HHMM`; smoke runs go to `smoke_YYYY_MM_DD_HHMM/{family}` (ignored by version control)."""
    stamp = (now or dt.datetime.now()).strftime("%Y_%m_%d_%H%M")
    return f"smoke_{stamp}/{family}" if smoke else f"{family}_{stamp}"


@lru_cache(maxsize=None)
def code_hash(*family_dirs: str) -> str:
    """sha256 over the non-test .py files of src/common and of the given family folders (names relative to src/)."""
    h = hashlib.sha256()
    for d in ("common", *family_dirs):
        root = REPO_ROOT / "src" / d
        for p in sorted(root.rglob("*.py")):
            rel = p.relative_to(root)
            if "tests" in rel.parts or "__pycache__" in rel.parts:
                continue
            h.update(f"{d}/{rel}".encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    out["python"] = platform.python_version()
    return out


def relativize(arg: str) -> str:
    """Absolute paths under the repository become relative; other absolute paths keep only their last component."""
    if not arg.startswith("/"):
        return arg
    p = Path(arg)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return p.name


def scrub_paths(text: str) -> str:
    """Repository paths become relative; any other absolute path keeps only its last component."""
    text = text.replace(str(REPO_ROOT) + "/", "").replace(str(REPO_ROOT), ".")
    return re.sub(r"(?<![\w./:~-])/(?:[A-Za-z0-9_.-]+/)+([A-Za-z0-9_.-]+)", r"\1", text)


def command_line(module: str, argv: list[str]) -> str:
    return " ".join(["python -m", module, *[relativize(a) for a in argv]])


def job_fingerprint(*parts, family_dirs: tuple[str, ...]) -> str:
    """Identity of one job: its description, the code, the library versions and the fold file."""
    blob = json.dumps({"parts": parts, "code": code_hash(*family_dirs), "versions": versions()},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def fold_hashes(seeds: list[int]) -> dict:
    return {f"seed_{s}.csv": file_sha256(fold_dir() / f"seed_{s}.csv") for s in seeds}


def write_run_info(out: Path, family: str, module: str, argv: list[str], status: str, seeds: list[int],
                   smoke: bool, start: dt.datetime, extra: dict | None = None, family_dirs: tuple[str, ...] = ()) -> dict:
    info = {"family": family, "status": status, "smoke": smoke, "command": command_line(module, argv),
            "seeds": seeds, "start": start.isoformat(timespec="seconds"),
            "end": None if status == "running" else dt.datetime.now().isoformat(timespec="seconds"),
            "cpu_count": os.cpu_count(), "versions": versions(), "code_sha256": code_hash(*family_dirs),
            "fold_sha256": fold_hashes(seeds), **(extra or {})}
    (Path(out) / "run_info.json").write_text(json.dumps(jsonable(info), indent=1), encoding="utf-8")
    return info
