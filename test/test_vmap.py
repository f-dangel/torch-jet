"""Test that make_fx(vmap(f)) produces correct results."""

from typing import Any

from pytest import mark
from torch import Tensor, cos, manual_seed, ones, rand, sigmoid, sin, tanh, vmap, zeros
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import Linear, Sequential, Tanh
from torch.nn.functional import linear
from torch.random import fork_rng

VMAP_CASES = [
    # output does not depend on placeholder
    {"f": lambda _: ones(5), "shape": (2,), "id": "constant"},
    # addition
    {"f": lambda x: x + 5, "shape": (2,), "id": "add-5"},
    # returns a tuple
    {"f": lambda x: (x + 5, x + 3), "shape": (2,), "id": "return-tuple"},
    # subtraction
    {"f": lambda x: x - 5, "shape": (2,), "id": "sub-5"},
    # multiplication
    {"f": lambda x: 5 * x, "shape": (2,), "id": "mul-5"},
    # element-wise functions
    {"f": cos, "shape": (2,), "id": "cos"},
    {"f": sin, "shape": (2,), "id": "sin"},
    {"f": tanh, "shape": (2,), "id": "tanh"},
    {"f": sigmoid, "shape": (2,), "id": "sigmoid"},
    # power function
    {"f": lambda x: x**2, "shape": (2,), "id": "pow-2"},
    {"f": lambda x: x**2.5, "shape": (2,), "id": "pow-2.5"},
    # linear function
    {
        "f": lambda x: linear(
            x, Tensor([[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0], [7.0, 8.0]])
        ),
        "shape": (2,),
        "id": "linear-4-2",
    },
    # neural network
    {
        "f": Sequential(
            Linear(4, 3, bias=False), Tanh(), Linear(3, 2, bias=True), Tanh()
        ),
        "shape": (4,),
        "id": "tanh-mlp-4-3-2",
    },
]


@mark.parametrize("config", VMAP_CASES, ids=[c["id"] for c in VMAP_CASES])
def test_make_fx_vmap(config: dict[str, Any], vmapsize: int = 3):
    """Ensure make_fx(vmap(f)) behaves like torch.vmap.

    Args:
        config: Configuration for the test case.
        vmapsize: The size of the batch dimension for vmap. Defaults to `3`.
    """
    manual_seed(0)
    f, shape = config["f"], config["shape"]

    # set up input to batched function
    x = rand(vmapsize, *shape)

    # set up batched functions
    vmap_f = vmap(f, randomness="different")

    # trace the vmapped function with make_fx
    example_x = zeros(vmapsize, *shape)
    traced_vmap_f = make_fx(vmap_f)(example_x)

    # compare their results
    with fork_rng():
        manual_seed(1)
        truth = vmap_f(x)
    with fork_rng():
        manual_seed(1)
        result = traced_vmap_f(x)

    if isinstance(truth, tuple):
        assert len(truth) == len(result)
        for t, r in zip(truth, result):
            assert t.shape == r.shape
            assert t.allclose(r)
    else:
        assert isinstance(truth, Tensor)
        assert truth.shape == result.shape
        assert truth.allclose(result)
