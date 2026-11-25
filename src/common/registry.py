"""Model registry: a model is one file in `<family>/models/` that registers itself; the family config lists which
registered models run (a config entry's `name` selects the registered class).

Adding a model = add a file and a config entry; removing a model = delete both.
"""
from __future__ import annotations
import importlib
import pkgutil


class Registry:
    def __init__(self, package: str):
        self.package = package
        self._models: dict[str, object] = {}

    def register(self, name: str):
        """Class decorator: instantiate the class and store it under `name`."""
        def deco(cls):
            if name in self._models:
                raise ValueError(f"model {name!r} registered twice")
            inst = cls()
            inst.name = name
            self._models[name] = inst
            return cls
        return deco

    def autoload(self) -> None:
        """Import every module of the models package so that each registers itself."""
        pkg = importlib.import_module(self.package)
        for m in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"{self.package}.{m.name}")

    def __getitem__(self, name: str):
        if name not in self._models:
            raise KeyError(f"unknown model {name!r}; registered: {sorted(self._models)}")
        return self._models[name]

    def __contains__(self, name: str) -> bool:
        return name in self._models

    def names(self) -> list[str]:
        return sorted(self._models)


class EstimatorModel:
    """A model fitted inside the nested CV. Subclasses set `pipeline` (scaled / native / tabpfn) and `score`
    (predict_proba / decision_function) and implement `build`, which returns the bare final estimator."""
    kind = "estimator"
    pipeline = "scaled"
    score = "predict_proba"

    def build(self, params: dict, fixed: dict, ctx):
        raise NotImplementedError

    @property
    def heavy(self) -> bool:
        """Heavy models (TabPFN) run in the torch pool, after the light scikit-learn pool."""
        return self.pipeline == "tabpfn"


class DerivedModel:
    """A model computed from data or from other cells of the same output directory instead of being fitted by the
    nested-CV runner. `run(ctx)` returns {"predictions", "inner_oof", "selected"} in the common cell layout;
    `feature_set` is the output sub-folder."""
    kind = "derived"
    feature_set = "fusion"
    has_validation = True      # False for a fixed rule without model selection: no inner OOF, Val AUC not applicable

    def requires(self, model_cfg: dict) -> list[tuple[str, str]]:
        """(model, feature set) cells of the same output directory this model reads."""
        return []

    def run(self, ctx: "DerivedContext") -> dict:
        raise NotImplementedError


class DerivedContext:
    def __init__(self, model_cfg: dict, out_root, seeds: list[int], smoke: bool):
        self.model_cfg, self.out_root, self.seeds, self.smoke = model_cfg, out_root, seeds, smoke
