"""Repository paths, data configuration and grid helpers shared by every family."""
from __future__ import annotations
import hashlib
import itertools
from pathlib import Path
import yaml

# src/common/config.py -> parents[2] is the repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG = Path(__file__).resolve().parent / "data.yaml"


def load_data_config() -> dict:
    with open(DATA_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(rel: str | Path) -> Path:
    """Resolve a path relative to the repository root (absolute paths pass through)."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def fold_dir() -> Path:
    """Single resolver of the fold directory (fold files, seeds, bootstrap draws)."""
    return repo_path(load_data_config()["paths"]["fold_dir"])


def load_seeds() -> list[int]:
    with open(fold_dir() / "seeds.yaml", encoding="utf-8") as f:
        return [int(s) for s in yaml.safe_load(f)["seeds"]]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_family_config(path: str | Path) -> dict:
    """Read a family config.yaml; every model entry needs a unique `name`."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg.get("models"):
        raise ValueError(f"{p.name}: no models enabled")
    names = [m["name"] for m in cfg["models"]]
    if len(set(names)) != len(names):
        raise ValueError(f"{p.name}: duplicate model names")
    return cfg


def first_last(values: list) -> list:
    """Smoke-grid rule: first and last value of a list."""
    return [values[0]] if len(values) == 1 else [values[0], values[-1]]


def expand_grid(model_cfg: dict, smoke: bool) -> list[dict]:
    """Cartesian product in YAML key order (first key outermost; the first config wins ties).

    Smoke grid = first and last value of every list of the grid."""
    grid = dict(model_cfg.get("grid") or {})
    if smoke:
        grid = {k: first_last(v) for k, v in grid.items()}
    keys = list(grid)
    if not keys:
        return [{}]
    return [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]


def estimator_fixed(model_cfg: dict, smoke: bool) -> dict:
    fixed = dict(model_cfg.get("fixed") or {})
    if smoke:
        fixed.update(model_cfg.get("smoke_fixed") or {})
    return fixed
