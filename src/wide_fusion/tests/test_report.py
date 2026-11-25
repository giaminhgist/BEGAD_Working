from common.report import contrast_table
from wide_fusion import report as R


def test_descriptive_rows_are_auc_only():
    md = ("| Model | Feature set | A | B | C | D |\n|---|---|---|---|---|---|\n| g | fusion | 0.7 [0.6, 0.8] | 0.5 [1, 2] | 0.4 | 0.3 |\n"
          "| d | fusion | 0.7 [0.6, 0.8] | 0.5 [1, 2] | 0.4 | 0.3 |\n")
    out = R.descriptive_auc_only(md, ["d"]).split("\n")
    assert out[2] == "| g | fusion | 0.7 [0.6, 0.8] | 0.5 [1, 2] | 0.4 | 0.3 |"
    assert out[3] == "| d | fusion | 0.7 [0.6, 0.8] | — | — | — |"


def test_contrast_and_mmse_tables_are_plain():
    c = {"a − b": {"cell": "a", "reference": "b", "delta": 0.01, "lo": -0.02, "hi": 0.04, "n_seeds_positive": 7,
                   "n_seeds": 10, "n_visits": 176}}
    t = contrast_table(c)
    assert "+0.010 [-0.020, +0.040]" in t.replace("−", "-") and "7/10" in t
    m = {k: {"auc01": {"mean": .7, "lo": .6, "hi": .8}, "sens": {"mean": .7, "lo": .6, "hi": .8},
             "spec": {"mean": .7, "lo": .6, "hi": .8}, "n_visits": 175} for k in ("mmse_cutoff", "mmse_cutoff_lt28")}
    t += R.mmse_table(m)
    assert "| AUC [95% CI]" in t and "MMSE <= 28" in t and "MMSE < 28" in t
    for word in ("p =", "Val AUC", "xploratory", "EXPLORATORY", "inflation", "multiplicity"):
        assert word not in t


from common.tests.test_metrics_evaluate import _synthetic  # noqa: E402


def test_regenerate_reads_saved_json_only(tmp_path):
    import json
    from common.evaluate import write_cell
    from common import bootstrap
    d = bootstrap.load_draws(B=20)
    p, o = _synthetic(d)
    write_cell(tmp_path / "g" / "fusion", "wide_fusion", "g", "fusion", p, o, {}, d)
    a = {"mean": .7, "lo": .6, "hi": .8}
    (tmp_path / "run_info.json").write_text(json.dumps({"fused_cells": ["g"], "descriptive_cells": [],
                                                        "references": {"r": {"path": "output/x"}}}))
    (tmp_path / "contrasts.json").write_text(json.dumps({"g − r": {"delta": .01, "lo": -.02, "hi": .04,
                                                                  "n_seeds_positive": 1, "n_seeds": 1, "n_visits": 176}}))
    (tmp_path / "mmse_rules.json").write_text(json.dumps({"mmse_cutoff": {"auc01": a, "sens": a, "spec": a, "n_visits": 175}}))
    (tmp_path / "diagnostics.json").write_text(json.dumps({"reference_test_auc": {"r": a}}))
    assert R.main(["--out", str(tmp_path)]) == 0
    t = (tmp_path / "results.md").read_text()
    assert "## Reference cells" in t and "## Paired contrasts" in t and "## MMSE rules" in t
