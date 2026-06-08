"""Taylor-mode automatic differentiation (jets) in PyTorch.

This module is a pure re-export hub: every public name is defined in a
submodule and surfaced here so users can ``from jet import ...``. ``__all__``
pins the supported public API; everything else (interpreter internals,
validation, type aliases) stays in its submodule.
"""

from jet._bilaplacian import bilaplacian
from jet._jet import jet
from jet._laplacian import laplacian
from jet._simplify import common_subexpression_elimination
from jet.tracing import capture_graph
from jet.utils import visualize_graph

__all__ = [
    "jet",
    "laplacian",
    "bilaplacian",
    "capture_graph",
    "common_subexpression_elimination",
    "visualize_graph",
]
