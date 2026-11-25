"""Publication hygiene of code, configs, docs and fold files (structural checks plus an internal pattern list)."""
import json
import re
from pathlib import Path
import pytest
from common.config import REPO_ROOT

PATTERN_FILE = next(REPO_ROOT.glob(".*/hygiene_patterns.txt"), REPO_ROOT / "hygiene_patterns.txt")
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt", ".json", ".csv", ".ini"}
ABS_PATH = re.compile(r"(?<![\w./:~-])/(?:[A-Za-z0-9_.-]+/){2,}")


def _files(*roots):
    for r in roots:
        for p in sorted((REPO_ROOT / r).rglob("*")):
            if p.is_file() and p.suffix in TEXT_SUFFIXES and "__pycache__" not in p.parts:
                yield p


def _public_files():
    yield from _files("src", "dataset/fold")
    for name in ["README.md", "requirements.txt", "pytest.ini"]:
        if (REPO_ROOT / name).exists():
            yield REPO_ROOT / name


def test_internal_pattern_list():
    if not PATTERN_FILE.exists():
        pytest.skip("internal pattern list not available")
    pats = [re.compile(l.strip(), re.IGNORECASE) for l in PATTERN_FILE.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]
    hits = []
    for f in _public_files():
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if any(p.search(line) for p in pats):
                hits.append(f"{f.relative_to(REPO_ROOT)}:{n}")
    assert not hits, "\n".join(hits)


def test_token_value_not_in_files():
    env = REPO_ROOT / ".env"
    if not env.exists():
        pytest.skip("no .env")
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("TABPFN_TOKEN="):
            token = line.split("=", 1)[1].strip().strip("'\"")
            assert len(token) > 8
            for f in _public_files():
                assert token not in f.read_text(encoding="utf-8", errors="ignore"), f


def test_gitignore_covers_private_paths():
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for entry in [".env", ".venv", "output/smoke_*", "__pycache__"]:
        assert entry in text


def test_no_absolute_paths_in_public_text():
    bad = []
    for f in _public_files():
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if ABS_PATH.search(line):
                bad.append(f"{f.relative_to(REPO_ROOT)}:{n}")
    assert not bad, "\n".join(bad)


def test_no_paths_outside_dataset_and_output_in_code():
    bad = []
    for f in _files("src"):
        if f.suffix != ".py" or "tests" in f.parts:
            continue
        for m in re.finditer(r"""["']((?:\.\./)+[^"']*)["']""", f.read_text(encoding="utf-8")):
            bad.append(f"{f.relative_to(REPO_ROOT)}: {m.group(1)}")
    assert not bad, "\n".join(bad)


def test_written_outputs_have_no_absolute_paths(tmp_path):
    """run_info.json and failure messages carry repository-relative paths only."""
    import datetime as dt
    from common.runinfo import scrub_paths, write_run_info
    write_run_info(tmp_path, "baseline", "baseline.run", ["--out", str(REPO_ROOT / "output" / "x"), "--config",
                   "/" + "elsewhere/dir/cfg.yaml"], "completed", [0], False, dt.datetime(2026, 1, 1))
    text = (tmp_path / "run_info.json").read_text()
    assert not ABS_PATH.search(text) and "output/x" in text
    sep = "/"
    msg = scrub_paths(f'File "{REPO_ROOT}{sep}src{sep}common{sep}runner.py", line 3; File "{sep}usr{sep}lib{sep}python3{sep}inspect.py"')
    assert str(REPO_ROOT) not in msg and not ABS_PATH.search(msg) and "src/common/runner.py" in msg
