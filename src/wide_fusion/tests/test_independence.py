import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def imports_of(folder: Path) -> set[str]:
    """Top-level names of every absolute import in the non-test files of `folder`."""
    mods = set()
    for p in folder.rglob("*.py"):
        if "tests" in p.relative_to(folder).parts:
            continue
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
    return mods


def test_package_imports_nothing_from_tmp_or_other_families():
    m = imports_of(PKG)
    assert not ({"memfusion", "tmp", "rescore", "normative_cognitive_fusion", "baseline", "cognitive_fusion"} & m), m


def test_no_non_test_code_refers_to_a_tmp_path():
    for p in PKG.rglob("*"):
        if p.suffix not in {".py", ".yaml", ".md"} or "tests" in p.relative_to(PKG).parts or "__pycache__" in p.parts:
            continue
        text = p.read_text(encoding="utf-8")
        assert "tmp/" not in text and "memfusion" not in text, p.name


def test_other_families_and_common_do_not_import_wide_fusion():
    src = PKG.parent
    for fam in ("common", "baseline", "cognitive_fusion"):
        assert "wide_fusion" not in imports_of(src / fam)
