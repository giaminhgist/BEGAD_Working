"""results.md generator: tables of the result statistics, built from the cells' metrics.json (and contrasts.json)."""
from __future__ import annotations
import json
from pathlib import Path

HEADER = ["Model", "Feature set", "Test AUC [95% CI]", "Test Acc [95% CI]", "Test Sen [95% CI]",
          "Test Spec [95% CI]"]
TITLES = {"baseline": "Baseline: CN vs MCI", "cognitive_fusion": "cognitive_fusion: CN vs MCI"}


def _fmt(m: dict, ci: bool = True) -> str:
    if m is None or m.get("mean") is None:
        return "\u2014"
    s = f"{m['mean']:.3f}"
    if ci and m.get("lo") is not None and m.get("hi") is not None:
        s += f" [{m['lo']:.3f}, {m['hi']:.3f}]"
    return s


def collect(run_dir: Path) -> list[dict]:
    """metrics.json of every `<model>/<feature_set>` cell directly under run_dir."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(run_dir).glob("*/*/metrics.json"))]


def build_table(cells: list[dict], model_order: list[str] | None = None, fs_order: list[str] | None = None) -> str:
    mo = {m: i for i, m in enumerate(model_order or [])}
    fo = {f: i for i, f in enumerate(fs_order or [])}
    cells = sorted(cells, key=lambda c: (mo.get(c["model"], len(mo)), c["model"],
                                         fo.get(c["feature_set"], len(fo)), c["feature_set"]))
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "|".join("---" for _ in HEADER) + "|"]
    for c in cells:
        s = c["summary"]
        lines.append("| " + " | ".join([c["model"], c["feature_set"], _fmt(s["test_auc"]),
                                        _fmt(s["test_acc"]), _fmt(s["test_sen"]), _fmt(s["test_spec"])]) + " |")
    return "\n".join(lines)


def contrast_table(contrasts: dict, reference: dict | None = None, cells: list[dict] | None = None) -> str:
    """Paired test-AUC contrasts: delta [CI], seeds with a positive difference, visits. Visits come from the contrast
    itself, else from the metrics.json of the cell with the same name as the contrast."""
    nv = {c["model"]: c.get("n_visits") for c in cells or []}
    lines = ["| Contrast | \u0394AUC [95% CI] | Seeds > 0 | Visits |", "|---|---|---|---|"]
    for name, c in contrasts.items():
        label = f"{name} \u2212 {reference['model']}" if reference and "\u2212" not in name else name
        pos = c.get("n_seeds_positive", c.get("n_pos"))
        visits = c.get("n_visits", nv.get(name))
        lines.append(f"| {label} | {c['delta']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] | {pos}/{c['n_seeds']} "
                     f"| {visits if visits is not None else chr(0x2014)} |")
    return "\n".join(lines)


def write_results_md(run_dir: Path, title: str, model_order: list[str] | None = None,
                     fs_order: list[str] | None = None, tables: list[str] | None = None,
                     contrasts: dict | None = None, reference: dict | None = None) -> Path:
    """results.md = title, one line on what the numbers are, the metrics table, then the given tables (each a
    string that starts with its own heading) and the paired-contrast table when contrasts are given."""
    cells = collect(run_dir)
    if not cells:
        raise RuntimeError(f"{run_dir}: no metrics.json found")
    n_seeds = len(cells[0].get("seeds", []))
    b = cells[0].get("bootstrap", {}).get("B")
    text = [f"# {title}", "", f"Mean over {n_seeds} seeds, 95% bootstrap CI (B = {b}).", "",
            build_table(cells, model_order, fs_order), ""]
    if contrasts:
        text += ["## Paired contrasts", "", contrast_table(contrasts, reference, cells), ""]
    for t in tables or []:
        text += [t, ""]
    p = Path(run_dir) / "results.md"
    p.write_text("\n".join(text), encoding="utf-8")
    return p


def regenerate(run_dir: Path) -> list[Path]:
    """Rebuild results.md from the existing metrics.json (and contrasts.json) without refitting. A folder without
    run_info.json (e.g. a smoke parent folder) is searched one level down. Nothing else is written."""
    import yaml
    from .config import REPO_ROOT
    from .data import FS_ORDER
    run_dir = Path(run_dir)
    if not (run_dir / "run_info.json").exists():
        done = []
        for sub in sorted(p for p in run_dir.iterdir() if p.is_dir() and (p / "run_info.json").exists()):
            done += regenerate(sub)
        return done
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    family = info["family"]
    order = info.get("models") or info.get("arms")
    if family == "wide_fusion":
        raise SystemExit("wide_fusion: use `python -m wide_fusion.report --out <dir>`")
    contrasts = ref = None
    cj = run_dir / "contrasts.json"
    if cj.exists():
        contrasts = json.loads(cj.read_text(encoding="utf-8"))
        ref = info.get("reference")
        if ref is None:
            cfg = yaml.safe_load((REPO_ROOT / "src" / family / "config.yaml").read_text(encoding="utf-8"))
            ref = cfg["reference"]
    return [write_results_md(run_dir, TITLES.get(family, f"{family}: CN vs MCI"), order, FS_ORDER + ["clinical"],
                             contrasts=contrasts, reference=ref)]


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Rebuild results.md from existing metrics.json files (no refitting).")
    ap.add_argument("--out", nargs="+", required=True, help="output directory (or a smoke parent folder)")
    for d in ap.parse_args(argv).out:
        for p in regenerate(Path(d)):
            print("wrote", p.parent.name + "/" + p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
