import numpy as np
import pandas as pd
import pytest
from common import bootstrap
from common.config import fold_dir, load_seeds
from common.data import (FS_ORDER, applicable_columns, check_fold_structure, load_analysis_data, load_cohort,
                         merge_folds, parse_feature_set)
from common.make_folds import load_folds, make_folds

SEEDS = [0, 1, 7, 13, 42, 123, 777, 1234, 2024, 3407]


def test_cohort_counts():
    c = load_cohort()
    assert len(c) == 176 and c["ParticipantID"].nunique() == 141
    assert c["Group"].value_counts().to_dict() == {"CN": 122, "MCI": 54}
    assert c["RealTag"].is_unique and c["RealTag"].is_monotonic_increasing
    assert c.loc[c["Group"] == "MCI", "label"].eq(1).all() and c.loc[c["Group"] == "CN", "label"].eq(0).all()


def test_seeds_file():
    assert load_seeds() == SEEDS


@pytest.mark.parametrize("fs,n", [("task1", 63), ("task3", 63), ("task4", 63), ("task5", 63), ("task7", 60),
                                   ("task9", 60), ("fusion", 372)])
def test_applicable_feature_counts(fs, n):
    X, y, g, c = load_analysis_data(fs)
    assert X.shape == (176, n)
    assert "gaze_quadrant_prop_bottom_left" not in " ".join(X.columns)
    assert list(X.index) == list(c["RealTag"])
    assert not X.isna().all(axis=0).any()


def test_applicability_excludes_other_task_features():
    assert "reading_line_change_downward_proportion" in applicable_columns(7)
    assert "reading_line_change_downward_proportion" not in applicable_columns(1)
    assert "kd_downward_line_transition_count" in applicable_columns(1)
    assert "kd_downward_line_transition_count" not in applicable_columns(9)


def test_parse_feature_set():
    assert parse_feature_set("task9") == 9 and parse_feature_set("fusion") == "fusion"
    with pytest.raises(ValueError):
        parse_feature_set("task")
    assert FS_ORDER[-1] == "fusion"


@pytest.mark.parametrize("seed", SEEDS)
def test_stored_folds_valid_and_subject_disjoint(seed):
    c = load_cohort()
    f = merge_folds(c, load_folds(seed))
    outer = f["outer_fold"].to_numpy()
    pid = c["ParticipantID"].to_numpy()
    for o in range(4):
        assert not (set(pid[outer == o]) & set(pid[outer != o])), "subject in train and test"
        inner = f[f"inner_fold_o{o}"].to_numpy()
        tr = outer != o
        for i in range(4):
            assert not (set(pid[tr & (inner == i)]) & set(pid[tr & (inner != i)]))


def test_stored_folds_equal_regenerated():
    c = load_cohort()
    for seed in (0, 3407):
        re = make_folds(c, seed)
        assert re.equals(load_folds(seed))


def test_merge_folds_detects_label_mismatch():
    c = load_cohort()
    f = load_folds(0)
    f.loc[0, "label"] = 1 - f.loc[0, "label"]
    with pytest.raises(AssertionError):
        merge_folds(c, f)


def test_check_fold_structure_detects_subject_in_two_outer_folds():
    c = load_cohort()
    f = merge_folds(c, load_folds(0))
    multi = c.groupby("ParticipantID").size()
    sub = multi[multi > 1].index[0]
    rows = np.flatnonzero(c["ParticipantID"] == sub)
    f.loc[rows[0], "outer_fold"] = (f.loc[rows[1], "outer_fold"] + 1) % 4
    with pytest.raises(AssertionError):
        check_fold_structure(f, c["ParticipantID"].to_numpy(), c["label"].to_numpy())


def test_bootstrap_draws_stored_equal_regenerated_and_stratified():
    c = load_cohort()
    d = bootstrap.load_draws()
    assert d["B"] == 2000 and d["counts"].shape == (2000, 141)
    new = bootstrap.make_draws(c)
    assert np.array_equal(new["counts"], d["counts"]) and np.array_equal(new["subjects"], d["subjects"])
    # every replicate keeps the number of subjects in each stratum
    for s in (0, 1):
        assert (d["counts"][:, d["stratum"] == s].sum(axis=1) == (d["stratum"] == s).sum()).all()
    ever = c.groupby("ParticipantID")["label"].max()
    assert (ever.loc[d["subjects"]].to_numpy() == d["stratum"]).all()


def test_bootstrap_smoke_prefix_and_replicate_index():
    d = bootstrap.load_draws(B=200)
    assert d["counts"].shape[0] == 200 and np.array_equal(d["counts"], bootstrap.load_draws()["counts"][:200])
    idx = bootstrap.replicate_index(np.array([0, 1, 1, 2]), np.array([2, 0, 1]))
    assert idx.tolist() == [0, 0, 3]
    with pytest.raises(ValueError):
        bootstrap.load_draws(B=5000)


def test_fold_dir_has_documented_files():
    d = fold_dir()
    for name in ["README.md", "seeds.yaml", "bootstrap_draws.npz", *[f"seed_{s}.csv" for s in SEEDS]]:
        assert (d / name).exists(), name
