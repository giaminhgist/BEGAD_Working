"""results.md tables of wide_fusion: reference cells, MMSE rules. No Val AUC, no p-values."""
from __future__ import annotations
import json
from pathlib import Path
from common.report import write_results_md


DASH = "\u2014"


def _auc(a: dict) -> str:
    return f"{a['mean']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]"


def reference_table(paths: dict, test_auc: dict) -> str:
    lines = ["## Reference cells", "", "| Reference | Path | Test AUC [95% CI] |", "|---|---|---|"]
    for k, path in paths.items():
        lines.append(f"| {k} | {path} | {_auc(test_auc[k]) if test_auc.get(k) else DASH} |")
    return "\n".join(lines)


def mmse_table(rows: dict) -> str:
    """MMSE rule table, from mmse_rules.json."""
    f = lambda m: f"{m['mean']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"  # noqa: E731
    label = {"mmse_cutoff": "MCI if MMSE <= 28", "mmse_cutoff_lt28": "MCI if MMSE < 28"}
    lines = ["## MMSE rules", "",
             "| Rule | AUC [95% CI] | Sens [95% CI] | Spec [95% CI] | Visits |", "|---|---|---|---|---|"]
    for k, r in rows.items():
        lines.append(f"| {k} ({label.get(k, k)}) | {f(r['auc01'])} | {f(r['sens'])} | {f(r['spec'])} | {r['n_visits']} |")
    return "\n".join(lines)


def write_results(out, contrasts: dict, mmse: dict, ref_paths: dict, ref_auc: dict, fused: list, desc: list) -> Path:
    """results.md of a wide_fusion run from already computed statistics (no fitting)."""
    p = write_results_md(out, "wide_fusion: CN vs MCI", fused + desc, ["fusion"],
                         [reference_table(ref_paths, ref_auc), mmse_table(mmse)], contrasts=contrasts)
    p.write_text(descriptive_auc_only(p.read_text(encoding="utf-8"), desc), encoding="utf-8")
    return p


def regenerate(run_dir, info: dict) -> Path:
    """Rebuild results.md from the saved json files of a finished run."""
    run_dir = Path(run_dir)
    rd = lambda n: json.loads((run_dir / n).read_text(encoding="utf-8"))  # noqa: E731
    return write_results(run_dir, rd("contrasts.json"), rd("mmse_rules.json"),
                         {k: v["path"] for k, v in info["references"].items()},
                         rd("diagnostics.json")["reference_test_auc"], info["fused_cells"], info["descriptive_cells"])


def descriptive_auc_only(text: str, names) -> str:
    """Descriptive cells show Test AUC [CI] only; their Acc / Sen / Spec columns are blanked in the main table."""
    names, out = set(names), []
    for line in text.split("\n"):
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cols) == 6 and cols[0] in names:
            line = "| " + " | ".join(cols[:3] + ["—"] * 3) + " |"
        out.append(line)
    return "\n".join(out)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Rebuild results.md of wide_fusion runs from the saved json files (no refitting).")
    ap.add_argument("--out", nargs="+", required=True, help="wide_fusion output directories")
    for d in ap.parse_args(argv).out:
        d = Path(d)
        print("wrote", regenerate(d, json.loads((d / "run_info.json").read_text(encoding="utf-8"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
