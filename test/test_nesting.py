"""Test nesting calls to `jet`."""

from test.test___init__ import CASE_IDS, CASES, setup_case
from typing import Any, Callable, Dict

from pytest import mark
from torch import Tensor, manual_seed
from torch.fx import symbolic_trace
from torch.nn import Module

from jet import jet, rev_jet
from jet.simplify import simplify


@mark.parametrize("config", CASES, ids=CASE_IDS)
def test_nested_jet(config: Dict[str, Any]):
    """Test whether jets can be nested.

    Args:
        config: Configuration dictionary of the test case.
    """
    manual_seed(0)
    f, x, vs, _ = setup_case(config)

    # split up the total degree k and the Taylor coefficients into two smaller ks
    k = config["k"]
    k1, k2 = k // 2, k - k // 2
    vs1, vs2 = vs[:k1], vs[k1:]

    # compute the ground truth with autodiff
    jet_rev_f = rev_jet(f, order=k1, detach=False)
    jet_rev_f_x = lambda x: jet_rev_f(x, *vs1)[k1]

    nested_jet_rev_f = rev_jet(jet_rev_f_x, order=k2, detach=True)
    nested_jet_rev_f_x = lambda x: nested_jet_rev_f(x, *vs2)[k2]

    truth = nested_jet_rev_f_x(x)

    # compute the nested jet with the `jet` function

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
            self.jet_f = jet(f, k=k)
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

    # Compute the first jet and evaluate it at the first set of vectors
    jet_f = JetModule(f, vs1, k1)
    # simplify the module
    jet_f = symbolic_trace(jet_f)
    jet_f = simplify(jet_f)
    print(f"Jet: {jet_f.graph}")

    # Compute the second jet and evaluate it at the second set of vectors
    nested_jet_f = JetModule(jet_f, vs2, k2)
    # simplify the nested module
    nested_jet_f = symbolic_trace(nested_jet_f)
    nested_jet_f = simplify(nested_jet_f)
    print(f"Nested Jet: {nested_jet_f.graph}")

    result = nested_jet_f(x)

    # compare
    assert result.allclose(truth)
