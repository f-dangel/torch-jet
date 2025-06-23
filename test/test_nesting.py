"""Test nesting calls to `jet`."""

from test.test___init__ import setup_case
from typing import Any, Callable, Dict

from pytest import mark
from torch import Tensor, cos, manual_seed, ones, sigmoid, sin, tanh
from torch.fx import GraphModule
from torch.nn import Linear, Module, Sequential, Tanh
from torch.nn.functional import linear

import jet

NEST_CASES = [
    # output does not depend on placeholder
    {"f": lambda _: ones(5), "shape": (2,), "id": "constant"},
    # addition
    {"f": lambda x: x + 5, "shape": (2,), "id": "add-5"},
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
            x, Tensor([[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0], [7.0, 8.0]]).double()
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
    {"f": lambda x: jet.utils.replicate(x, 5), "shape": (2,), "id": "replicate-5"},
    # sum_vmapped
    {"f": lambda x: jet.utils.sum_vmapped(x), "shape": (3, 5), "id": "sum_vmapped-3"},
    # sum_vmapped(sin)
    {
        "f": lambda x: jet.utils.sum_vmapped(sin(x), pos=1),
        "shape": (6, 2),
        "id": "sum_vmapped_pos1",
    },
]


class JetModule(Module):
    """A module that computes the k-th jet of a function f."""

    def __init__(
        self, f: Callable[[Tensor], Tensor], vs: tuple[Tensor, ...], k: int
    ) -> None:
        """Initialize the JetModule.

        Args:
            f: The function to compute the jet of.
            vs: The Taylor coefficients for the jet.
            k: The order of the jet.
        """
        super().__init__()
        self.jet_f = jet.jet(f, k=k)
        self.k = k
        self.vs = vs

    def forward(self, x: Tensor) -> Tensor:
        """Compute the k-th jet of f at x with the given Taylor coefficients.

        Args:
            x: The input tensor at which to evaluate the jet.

        Returns:
            The k-th jet of f at x with the given Taylor coefficients.
        """
        return self.jet_f(x, *self.vs)[self.k]


@mark.parametrize("k1, k2", [(0, 0), (0, 1), (2, 2), (3, 2)])
@mark.parametrize("config", NEST_CASES, ids=[c["id"] for c in NEST_CASES])
def test_nested_jet(config: Dict[str, Any], k1: int, k2: int):
    """Test whether jets can be nested.

    Args:
        config: Configuration dictionary of the test case.
        k1: The order of the first jet.
        k2: The order of the second jet.
    """
    manual_seed(0)
    # insert missing entries to make the config work with `setup_case`
    config["is_batched"] = False
    f, x, vs, _ = setup_case(config, k=k1 + k2)
    vs1, vs2 = vs[:k1], vs[k1:]

    # Compute the ground truth with autodiff
    jet_rev_f = jet.rev_jet(f, order=k1, detach=False)
    jet_rev_f_x = lambda x: jet_rev_f(x, *vs1)[k1]  # noqa: E731

    nested_jet_rev_f = jet.rev_jet(jet_rev_f_x, order=k2)
    nested_jet_rev_f_x = lambda x: nested_jet_rev_f(x, *vs2)[k2]  # noqa: E731

    truth = nested_jet_rev_f_x(x)

    # Compute the nested jet with the `jet` function
    tracer = jet.JetTracer()

    # Compute the first jet and evaluate it at the first set of vectors
    jet_f = JetModule(f, vs1, k1)
    jet_f = GraphModule(jet_f, tracer.trace(jet_f))
    print(f"Jet: {jet_f.graph}")

    # Compute the second jet and evaluate it at the second set of vectors
    nested_jet_f = JetModule(jet_f, vs2, k2)
    nested_jet_f = GraphModule(nested_jet_f, tracer.trace(nested_jet_f))
    print(f"Nested Jet: {nested_jet_f.graph}")

    # Compare
    result = nested_jet_f(x)
    assert result.allclose(truth)
