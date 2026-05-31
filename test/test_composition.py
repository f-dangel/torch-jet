"""Composition tests for Taylor mode.

These tests exercise the interpreter / tracing machinery's ability to thread
jets through realistic compute graphs. They are NOT primitive coverage --
each primitive's standalone correctness is in ``test_primitives.py``. The
four fixtures cover the realistic-shape territory:

1. **Scalar `R^n → R`** -- the canonical Laplacian shape.
2. **Small MLP** -- the canonical PINN compute graph.
3. **Deep pytree → pytree** -- exercises ``tuple``/``list``/``dict``
   container handling on both input and output.
4. **Multi-input** -- exercises the variadic positional path.

Each fixture runs under both standard and collapsed mode at ``K ∈ {2, K_MAX}``.
"""

from typing import Any

from pytest import mark
from torch import Tensor, cos, float64, manual_seed, rand, sin, tanh
from torch.nn import Linear, Sequential, Tanh
from torch.testing import assert_close

import jet
from jet import rev_jet
from test.utils import (
    K_IDS,
    K_VALUES,
    make_collapsed_jet_args,
    make_standard_jet_args,
    shape,
    shapes,
)

# Module-level MLP so the captured graph is deterministic across runs.
manual_seed(0)
_MLP = Sequential(
    Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
).double()


def _deep_pytree_f(x: Tensor, params: list) -> tuple[Tensor, dict[str, Tensor]]:
    """Pytree-shaped composition: nested list input, mixed tuple/dict output."""
    w = params[0]
    b0, b1 = params[1]
    h = sin(x) * w
    return (h + b0, {"a": cos(h) * b1, "b": tanh(h + b0 + b1)})


def _deep_pytree_mock_args_fn() -> tuple:
    """Mock args for ``_deep_pytree_f``: ``(x, [w, [b0, b1]])`` of float64."""
    return (
        rand(3, dtype=float64),
        [rand(3, dtype=float64), [rand(3, dtype=float64), rand(3, dtype=float64)]],
    )


COMPOSITION_CASES = [
    {
        "id": "scalar_Rn_to_R",
        # ``.sum(0)`` (dim-IntList overload) — ``.sum()`` traces to
        # ``aten.sum.default`` which has no jet rule.
        "f": lambda x: (sin(x) * x).sum(0),
        "mock_args_fn": shape(5),
    },
    {
        "id": "mlp",
        "f": _MLP,
        "mock_args_fn": shape(5),
    },
    {
        "id": "deep_pytree",
        "f": _deep_pytree_f,
        "mock_args_fn": _deep_pytree_mock_args_fn,
    },
    {
        "id": "multi_input",
        "f": lambda x, y: sin(x) * cos(y),
        "mock_args_fn": shapes((4,), (4,)),
    },
]

COMPOSITION_IDS = [c["id"] for c in COMPOSITION_CASES]


@mark.parametrize("K", K_VALUES, ids=K_IDS)
@mark.parametrize("config", COMPOSITION_CASES, ids=COMPOSITION_IDS)
def test_composition_standard(config: dict[str, Any], K: int):
    """jet(composition) matches rev_jet(composition) on random inputs."""
    f = config["f"]
    mock_args = config["mock_args_fn"]()
    args = make_standard_jet_args(mock_args, K)

    jet_out = jet.jet(f, mock_args)(*args)
    rev_out = rev_jet(f)(*args)
    assert_close(jet_out, rev_out)


@mark.parametrize("K", K_VALUES, ids=K_IDS)
@mark.parametrize("config", COMPOSITION_CASES, ids=COMPOSITION_IDS)
def test_composition_collapsed(config: dict[str, Any], K: int):
    """Collapsed-mode composition matches the _uncollapsed_via_vmap oracle."""
    f = config["f"]
    mock_args = config["mock_args_fn"]()
    args = make_collapsed_jet_args(mock_args, K)

    cjet_out = jet.jet(f, mock_args, collapsed=True)(*args)
    oracle_out = jet._uncollapsed_via_vmap(f, mock_args, randomization=None)(*args)
    assert_close(cjet_out, oracle_out)
