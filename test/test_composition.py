"""Composition tests for Taylor mode.

Exercise the interpreter / tracing machinery on realistic compute graphs.
Not primitive coverage -- see ``test_primitives.py`` for that.
"""

from typing import Any

from pytest import mark
from torch import Tensor, cos, rand, sin, tanh

from test.utils import (
    K_AND_MODE,
    _stateless,
    assert_jet_matches_oracle,
    mlp,
)


def _deep_pytree_f(x: Tensor, params: list) -> tuple[Tensor, dict[str, Tensor]]:
    """Pytree-shaped composition: nested list input, mixed tuple/dict output."""
    w, (b0, b1) = params
    h = sin(x) * w
    return (h + b0, {"a": cos(h) * b1, "b": tanh(h + b0 + b1)})


COMPOSITION_CASES = [
    {
        "id": "scalar_Rn_to_R",
        "f": _stateless(lambda x: (sin(x) * x).sum(0)),
        "args_fn": lambda: (rand(5),),
    },
    {"id": "mlp", "f": mlp, "args_fn": lambda: (rand(5),)},
    {"id": "mlp_batched", "f": mlp, "args_fn": lambda: (rand(10, 5),)},
    {
        "id": "deep_pytree",
        "f": _stateless(_deep_pytree_f),
        "args_fn": lambda: (rand(3), [rand(3), [rand(3), rand(3)]]),
    },
    {
        "id": "sin_residual",
        # variable aliasing: ``x`` appears both inside ``sin`` and outside.
        "f": _stateless(lambda x: sin(x) + x),
        "args_fn": lambda: (rand(3),),
    },
    {
        "id": "multi_input",
        "f": _stateless(lambda x, y: sin(x) * cos(y)),
        "args_fn": lambda: (rand(4), rand(4)),
    },
    {
        "id": "multi_input_tuple_output",
        "f": _stateless(lambda x, y: (x + y, x * y)),
        "args_fn": lambda: (rand(4), rand(4)),
    },
    {
        "id": "multi_input_dict_output",
        "f": _stateless(lambda x, y: {"sum": x + y, "prod": x * y}),
        "args_fn": lambda: (rand(4), rand(4)),
    },
    {
        "id": "dict_only_input",
        "f": _stateless(lambda d: sin(d["a"]) * d["b"]),
        "args_fn": lambda: ({"a": rand(4), "b": rand(4)},),
    },
    {
        "id": "dict_first_input",
        "f": _stateless(lambda params, x: params["scale"] * sin(x) + params["bias"]),
        "args_fn": lambda: ({"scale": rand(3), "bias": rand(3)}, rand(3)),
    },
]


@mark.parametrize("K, collapsed", K_AND_MODE)
@mark.parametrize("config", COMPOSITION_CASES, ids=lambda c: c["id"])
def test_composition(config: dict[str, Any], K: int, collapsed: bool, device: str):
    """``jet(composition)`` matches its mode-specific oracle."""
    assert_jet_matches_oracle(config, K, collapsed, device)
