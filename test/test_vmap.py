"""Test symbolic vmap."""

from typing import Any

from pytest import mark
from torch import Tensor, manual_seed, ones, rand, vmap

from jet.vmap import traceable_vmap

CASES = [
    # output does not depend on placeholder
    {"f": lambda _: ones(5), "shape": (2,), "id": "constant"},
    # addition
    {"f": lambda x: x + 5, "shape": (2,), "id": "add-5"},
    # returns a tuple
    {"f": lambda x: (x + 5, x + 3), "shape": (2,), "id": "return-tuple"},
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
