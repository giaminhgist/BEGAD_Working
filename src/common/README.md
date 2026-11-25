# common

This folder holds the code shared by every model family. Nothing family-specific lives here, and `common` never
imports a family.

| Module | Purpose |
|---|---|
| `config.py`, `data.yaml` | repository paths; data configuration (cohort groups, tasks, excluded feature, clinical-score columns, CV sizes); grid expansion (`expand_grid`, the smoke grid = first and last value of each list) |
| `data.py` | cohort (176 CN/MCI visits, 141 subjects), feature loading per task and for early fusion, applicable-feature filter, fold merge with fold-structure checks |
| `make_folds.py` | writes `dataset/fold/seed_<seed>.csv`, `seeds.yaml` and the fold README (outer 4 × inner 4 `StratifiedGroupKFold`, grouped by subject) |
| `bootstrap.py` | stratified subject-level bootstrap draws (`dataset/fold/bootstrap_draws.npz`, B = 2000) and percentile CIs |
| `metrics.py` | rank-based AUC, Youden threshold, accuracy, sensitivity, specificity |
| `pipelines.py`, `tabpfn_frozen.py` | preprocessing pipelines (`scaled`, `native`, `tabpfn`) and the frozen TabPFN v3.5 wrapper (classification and regression, CPU) |
| `registry.py` | model registry used by `<family>/models/`: `EstimatorModel` (fitted by the nested CV) and `DerivedModel` (computed from data or from other cells) |
| `runner.py`, `scheduler.py` | nested cross-validation of one (model, feature set, seed) job; parallel job pools with resume |
| `evaluate.py` | per-cell metrics (`metrics.json`) and cell files |
| `latefusion.py` | helpers for scores built from other cells' saved outputs (merging, leakage check, Youden step) |
| `report.py`, `runinfo.py`, `cli.py` | `results.md`, output naming and `run_info.json`, the shared command-line options |

Commands (run from the repository root):

    PYTHONPATH=src .venv/bin/python -m common.data                       # cohort summary
    PYTHONPATH=src .venv/bin/python -m common.make_folds                 # fold files
    PYTHONPATH=src .venv/bin/python -m common.bootstrap                  # bootstrap draws (refuses to overwrite without --force)
    PYTHONPATH=src .venv/bin/python -m common.report --out output/<dir>  # rebuild results.md from metrics.json, no refitting

Tests: `PYTHONPATH=src .venv/bin/python -m pytest -q src/common/tests`.
