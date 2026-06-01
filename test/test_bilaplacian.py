"""Tests the computation of Bi-Laplacians.

The Bi-Laplacian of a function f(x) ∈ R with x ∈ Rⁿ is defined as the Laplacian of the
Laplacian, or Δf(x) = ∑ᵢ ∑ⱼ ∂⁴f(x) / ∂xᵢ²∂xⱼ² ∈ R where the sum ranges to n.

For functions that produce vectors or tensors, the Bi-Laplacian is defined per output
component. It has the same shape as f(x).
"""

from typing import Any, Callable

from einops import einsum
from pytest import mark
from torch import Tensor, manual_seed, sigmoid
from torch.func import hessian
from torch.testing import assert_close

from jet.bilaplacian import SUPPORTED_DISTRIBUTIONS
from jet.bilaplacian import bilaplacian as jet_bilaplacian
from jet.laplacian import laplacian as jet_laplacian
from jet.utils import run_seeded
from test.test_laplacian import _check_mc_convergence
from test.utils import MLP, setup_case, shape

DISTRIBUTIONS = SUPPORTED_DISTRIBUTIONS
DISTRIBUTION_IDS = [f"distribution={d}" for d in DISTRIBUTIONS]

# make generation of test cases deterministic
manual_seed(0)

BILAPLACIAN_CASES = [
    # 5d tanh-activated two-layer MLP
    {"f": MLP, "args_fn": shape(5), "id": "two-layer-tanh-mlp"},
    # 3d sigmoid(sigmoid) function
    {"f": lambda x: sigmoid(sigmoid(x)), "args_fn": shape(3), "id": "sigmoid-sigmoid"},
]

BILAPLACIAN_IDS = [config["id"] for config in BILAPLACIAN_CASES]


def bilaplacian(f: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Compute the Bi-Laplacian by taking the trace of the fourth derivative tensor.

    Args:
        f: The function to compute the Bi-Laplacian of.
        x: The point at which to compute the Bi-Laplacian.

    Returns:
        The Bi-Laplacian of the function f at the point X, evaluated
        for each element f[i](x). Has same shape as f(x).
    """
    # compute the derivative tensor of fourth derivatives
    d4f = hessian(hessian(f))

    # trace it using einsum to support functions with non-scalar outputs
    dims1 = " ".join([f"i{i}" for i in range(x.ndim)])
    dims2 = " ".join([f"j{j}" for j in range(x.ndim)])
    # if x is a vector, this is just '... i i j j -> ...' where '...' corresponds
    # to the shape of f(x)
    equation = f"... {dims1} {dims1} {dims2} {dims2} -> ..."

    return einsum(d4f(x), equation)


@mark.parametrize("collapsed", [True, False], ids=["collapsed", "standard"])
@mark.parametrize("config", BILAPLACIAN_CASES, ids=BILAPLACIAN_IDS)
def test_bilaplacian(config: dict[str, Any], collapsed: bool):
    """Compare Bi-Laplacian implementations.

    Args:
        config: Configuration dictionary of the test case.
        collapsed: Whether to use collapsed Taylor mode.
    """
    f, (x,) = setup_case(config)

    # using torch.func
    bilap_func = bilaplacian(f, x)

    # using jets
    bilap_fn = jet_bilaplacian(f, x, collapsed=collapsed)
    bilap_jet = bilap_fn(x)
    assert_close(bilap_func, bilap_jet)


@mark.xfail(
    raises=NotImplementedError,
    reason="aten.zeros_like.default has no jet rule; blocks nesting laplacian twice",
    strict=True,
)
@mark.parametrize("collapsed", [True, False], ids=["collapsed", "standard"])
@mark.parametrize("config", BILAPLACIAN_CASES, ids=BILAPLACIAN_IDS)
def test_bilaplacian_matches_nested_laplacian(config: dict[str, Any], collapsed: bool):
    """``Δ(Δf)(x) == Δ²f(x)`` -- nesting laplacian twice yields the bilaplacian."""
    f, (x,) = setup_case(config)
    lap_of_lap = jet_laplacian(
        jet_laplacian(f, x, collapsed=collapsed), x, collapsed=collapsed
    )
    expected = jet_bilaplacian(f, x, collapsed=collapsed)(x)
    assert_close(lap_of_lap(x), expected)


@mark.parametrize("collapsed", [True, False], ids=["collapsed", "standard"])
@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
@mark.parametrize("config", BILAPLACIAN_CASES, ids=BILAPLACIAN_IDS)
def test_Bilaplacian_randomization(
    config: dict[str, Any],
    distribution: str,
    collapsed: bool,
    max_num_chunks: int = 500,
    chunk_size: int = 256,
    target_rel_error: float = 1e-2,
):
    """Test convergence of the Bi-Laplacian's Monte-Carlo estimator.

    Args:
        config: Configuration dictionary of the test case.
        distribution: The distribution from which to draw random vectors.
        collapsed: Whether to use collapsed Taylor mode.
        max_num_chunks: Maximum number of chunks to accumulate. Default: `200`.
        chunk_size: Number of samples per chunk. Default: `256`.
        target_rel_error: Target relative error for convergence. Default: `1e-2`.
    """
    f, (x,) = setup_case(config)

    # reference: Using PyTorch
    bilap = bilaplacian(f, x)

    randomization = (distribution, chunk_size)

    # check convergence of MC estimator
    bilap_fn = jet_bilaplacian(f, x, randomization=randomization, collapsed=collapsed)

    converged = _check_mc_convergence(
        bilap,
        lambda idx: run_seeded(bilap_fn, idx, x),
        chunk_size,
        max_num_chunks,
        target_rel_error,
    )
    assert converged, f"Monte-Carlo Bi-Laplacian ({distribution}) did not converge."
