import numpy as np
import pandas as pd
import pytest
from cognitive_fusion.contrast import paired_report
from common import bootstrap


def frames(better=False):
    d = bootstrap.load_draws(B=60)
    rng = np.random.default_rng(0)
    rows = []
    for s in (0, 1):
        for i, (sub, st) in enumerate(zip(d["subjects"], d["stratum"])):
            rows.append({"seed": s, "RealTag": f"{sub}_a", "ParticipantID": sub, "y_true": float(st),
                         "y_score": rng.normal(st * 0.5, 1)})
    a = pd.DataFrame(rows)
    b = a.copy()
    if better:
        b["y_score"] = a["y_score"] + a["y_true"] * 2.0
    return a, b, d


def test_identical_cells_have_zero_delta():
    a, _, d = frames()
    r = paired_report(a, a.copy(), d, [0, 1])
    assert r["delta"] == 0 and r["n_pos"] == 0
    assert "p" not in r


def test_better_scores_have_positive_delta():
    a, b, d = frames(better=True)
    r = paired_report(b, a, d, [0, 1])
    assert r["delta"] > 0 and r["lo"] <= r["delta"] + 1e-9 and r["n_pos"] == 2


def test_misaligned_rows_are_rejected():
    a, b, d = frames()
    b = b.copy()
    b["y_true"] = 1 - b["y_true"]
    with pytest.raises(AssertionError):
        paired_report(a, b, d, [0, 1])
