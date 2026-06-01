"""Composition tests for Taylor mode.

Exercise the interpreter / tracing machinery on realistic compute graphs.
Not primitive coverage -- see ``test_primitives.py`` for that.
"""

from typing import Any

from pytest import mark
from torch import Tensor, cos, float64, manual_seed, rand, sin, tanh
from torch.nn import Linear, Sequential, Tanh

from test.utils import K_AND_MODE, assert_jet_matches_oracle, shape, shapes

# Module-level MLP so the captured graph is deterministic across runs.
manual_seed(0)
_MLP = Sequential(
    Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
).double()


def _deep_pytree_f(x: Tensor, params: list) -> tuple[Tensor, dict[str, Tensor]]:
    """Pytree-shaped composition: nested list input, mixed tuple/dict output."""
    w, (b0, b1) = params
    h = sin(x) * w
    return (h + b0, {"a": cos(h) * b1, "b": tanh(h + b0 + b1)})


def _deep_pytree_args_fn() -> tuple:
    """Mock args for ``_deep_pytree_f``: ``(x, [w, [b0, b1]])`` of float64."""
    return (
        rand(3, dtype=float64),
        [rand(3, dtype=float64), [rand(3, dtype=float64), rand(3, dtype=float64)]],
    )


COMPOSITION_CASES = [
    {
        "id": "scalar_Rn_to_R",
        "f": lambda x: (sin(x) * x).sum(0),
        "args_fn": shape(5),
    },
    {"id": "mlp", "f": _MLP, "args_fn": shape(5)},
    {"id": "mlp_batched", "f": _MLP, "args_fn": shape(10, 5)},
    {
        "id": "deep_pytree",
        "f": _deep_pytree_f,
        "args_fn": _deep_pytree_args_fn,
    },
    {
        "id": "sin_residual",
        # variable aliasing: ``x`` appears both inside ``sin`` and outside.
        "f": lambda x: sin(x) + x,
        "args_fn": shape(3),
    },
    {
        "id": "multi_input",
        "f": lambda x, y: sin(x) * cos(y),
        "args_fn": shapes((4,), (4,)),
    },
    {
        "id": "multi_input_tuple_output",
        "f": lambda x, y: (x + y, x * y),
        "args_fn": shapes((4,), (4,)),
    },
    {
        "id": "multi_input_dict_output",
        "f": lambda x, y: {"sum": x + y, "prod": x * y},
        "args_fn": shapes((4,), (4,)),
    },
    {
        "id": "dict_only_input",
        "f": lambda d: sin(d["a"]) * d["b"],
        "args_fn": lambda: (
            {"a": rand(4, dtype=float64), "b": rand(4, dtype=float64)},
        ),
    },
    {
        "id": "dict_first_input",
        "f": lambda params, x: params["scale"] * sin(x) + params["bias"],
        "args_fn": lambda: (
            {"scale": rand(3, dtype=float64), "bias": rand(3, dtype=float64)},
            rand(3, dtype=float64),
        ),
    },
]


@mark.parametrize("K, collapsed", K_AND_MODE)
@mark.parametrize("config", COMPOSITION_CASES, ids=lambda c: c["id"])
def test_composition(config: dict[str, Any], K: int, collapsed: bool):
    """``jet(composition)`` matches its mode-specific oracle."""
    assert_jet_matches_oracle(config, K, collapsed)
