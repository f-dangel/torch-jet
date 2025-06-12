"""Test symbolic vmap."""

from typing import Any

from pytest import mark
from torch import Tensor, cos, manual_seed, ones, rand, sigmoid, sin, tanh, vmap
from torch.fx import wrap
from torch.nn import Linear, Sequential, Tanh
from torch.nn.functional import linear

from jet.utils import replicate, sum_vmapped
from jet.vmap import traceable_vmap

wrap(replicate)
wrap(sum_vmapped)

CASES = [
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
    # replicate
    {"f": lambda x: replicate(x, 5), "shape": (2,), "id": "replicate-5"},
    # sum_vmapped
    {"f": lambda x: sum_vmapped(x), "shape": (6, 2), "id": "sum_vmapped_pos0"},
    {"f": lambda x: sum_vmapped(x, pos=1), "shape": (6, 2), "id": "sum_vmapped_pos1"},
]


@mark.parametrize("config", CASES, ids=[c["id"] for c in CASES])
def test_traceable_vmap(config: dict[str, Any], vmapsize: int = 3):
    """Ensure trace-able vmap behaves like torch.vmap.

    Args:
        config: Configuration for the test case.
        vmapsize: The size of the batch dimension for vmap. Defaults to `3`.
    """
    manual_seed(0)
    f, shape = config["f"], config["shape"]

    # set up input to batched function
    x = rand(vmapsize, *shape)

    # set up batched functions
    vmap_f = vmap(f)
    tr_vmap_f = traceable_vmap(f, vmapsize)

    # compare their results
    truth = vmap_f(x)
    result = tr_vmap_f(x)

    if isinstance(truth, tuple):
        assert len(truth) == len(result)
        for t, r in zip(truth, result):
            assert t.shape == r.shape
            assert t.allclose(r)
    else:
        assert isinstance(truth, Tensor)
        assert truth.shape == result.shape
        assert truth.allclose(result)
