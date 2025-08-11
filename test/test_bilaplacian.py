"""Tests the computation of Bi-Laplacians.

The Bi-Laplacian of a function f(x) ∈ R with x ∈ Rⁿ is defined as the Laplacian of the
Laplacian, or Δf(x) = ∑ᵢ ∑ⱼ ∂⁴f(x) / ∂xᵢ²∂xⱼ² ∈ R where the sum ranges to n.

For functions that produce vectors or tensors, the Bi-Laplacian is defined per output
component. It has the same shape as f(x).
"""

from test.test___init__ import report_nonclose, setup_case
from test.test_laplacian import _check_mc_convergence
from typing import Any, Callable

from einops import einsum
from pytest import mark
from torch import Tensor, manual_seed, sigmoid
from torch.func import hessian
from torch.nn import Linear, Sequential, Tanh

from jet.bilaplacian import Bilaplacian

DISTRIBUTIONS = Bilaplacian.SUPPORTED_DISTRIBUTIONS
DISTRIBUTION_IDS = [f"distribution={d}" for d in DISTRIBUTIONS]

# make generation of test cases deterministic
manual_seed(0)

BILAPLACIAN_CASES = [
    # 5d tanh-activated two-layer MLP
    {
        "f": Sequential(
            Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
        ),
        "shape": (5,),
        "id": "two-layer-tanh-mlp",
    },
    # 3d sigmoid(sigmoid) function
    {"f": lambda x: sigmoid(sigmoid(x)), "shape": (3,), "id": "sigmoid-sigmoid"},
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


@mark.parametrize("config", BILAPLACIAN_CASES, ids=BILAPLACIAN_IDS)
def test_bilaplacian(config: dict[str, Any]):
    """Compare Laplacian implementations.

    Args:
        config: Configuration dictionary of the test case.
    """
    f, x, _, _ = setup_case({**config, "is_batched": None})

    # using torch.func
    bilap_func = bilaplacian(f, x)

    # using jets
    bilap_mod = Bilaplacian(f, x)
    bilap_jet = bilap_mod(x)
    report_nonclose(bilap_func, bilap_jet, name="functorch and jet Bi-Laplacians")


@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
@mark.parametrize("config", BILAPLACIAN_CASES, ids=BILAPLACIAN_IDS)
def test_Bilaplacian_randomization(
    config: dict[str, Any],
    distribution: str,
    max_num_chunks: int = 200,
    chunk_size: int = 256,
    target_rel_error: float = 1e-2,
):
    """Test convergence of the Bi-Laplacian's Monte-Carlo estimator.

    Args:
        config: Configuration dictionary of the test case.
        distribution: The distribution from which to draw random vectors.
        max_num_chunks: Maximum number of chunks to accumulate. Default: `200`.
        chunk_size: Number of samples per chunk. Default: `256`.
        target_rel_error: Target relative error for convergence. Default: `1e-2`.
    """
    f, x, _, _ = setup_case({**config, "is_batched": None})

    # reference: Using PyTorch
    bilap = bilaplacian(f, x)

    randomization = (distribution, chunk_size)

    # check convergence of MC estimator
    def sample(idx: int) -> Tensor:
        manual_seed(idx)
        return Bilaplacian(f, x, randomization=randomization)(x)

    converged = _check_mc_convergence(
        bilap, sample, chunk_size, max_num_chunks, target_rel_error
    )
    assert converged, f"Monte-Carlo Bi-Laplacian ({distribution}) did not converge."
