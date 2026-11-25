"""Baseline model registry. Every module in this package registers one model with `@register("<name>")`."""
from common.registry import Registry, DerivedModel, EstimatorModel  # noqa: F401  (re-exported for the model files)

REGISTRY = Registry(__name__)
register = REGISTRY.register
REGISTRY.autoload()
