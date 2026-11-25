import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]


def imports_of(folder: Path) -> set[str]:
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


def test_families_do_not_import_each_other():
    assert "cognitive_fusion" not in imports_of(SRC / "baseline") | imports_of(SRC / "common")
    assert "baseline" not in imports_of(SRC / "cognitive_fusion") | imports_of(SRC / "common")


def test_common_does_not_import_a_family():
    assert not ({"baseline", "cognitive_fusion"} & imports_of(SRC / "common"))
