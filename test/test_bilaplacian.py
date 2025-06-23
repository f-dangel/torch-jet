"""Tests the computation of Bi-Laplacians.

The Bi-Laplacian of a function f(x) ∈ R with x ∈ Rⁿ is defined as the Laplacian of the
Laplacian, or Δf(x) = ∑ᵢ ∑ⱼ ∂⁴f(x) / ∂xᵢ²∂xⱼ² ∈ R where the sum ranges to n.

For functions that produce vectors or tensors, the Bi-Laplacian is defined per output
component. It has the same shape as f(x).
"""

from test.test___init__ import report_nonclose, setup_case
from test.test_laplacian import _check_mc_convergence
from typing import Any, Callable, Dict

from einops import einsum
from pytest import mark
from torch import Tensor, manual_seed, sigmoid, zeros, zeros_like
from torch.autograd import grad
from torch.func import hessian, vmap
from torch.nn import Linear, Sequential, Tanh

from jet.bilaplacian import Bilaplacian, RandomizedBilaplacian

DISTRIBUTIONS = RandomizedBilaplacian.SUPPORTED_DISTRIBUTIONS
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
    # 5d tanh-activated two-layer MLP with batched input
    {
        "f": Sequential(
            Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
        ),
        "shape": (10, 5),
        "is_batched": True,
        "id": "batched-two-layer-tanh-mlp",
    },
    # 3d sigmoid(sigmoid) function
    {"f": lambda x: sigmoid(sigmoid(x)), "shape": (3,), "id": "sigmoid-sigmoid"},
]
# set the `is_batched` flag for all cases
for config in BILAPLACIAN_CASES:
    config["is_batched"] = config.get("is_batched", False)

BILAPLACIAN_IDS = [config["id"] for config in BILAPLACIAN_CASES]


def bilaplacian(f: Callable[[Tensor], Tensor], X: Tensor, is_batched: bool) -> Tensor:
    """Compute the Bi-Laplacian by taking the trace of the fourth derivative tensor.

    Args:
        f: The function to compute the Bi-Laplacian of.
        X: The point at which to compute the Bi-Laplacian.
        is_batched: Whether the input tensor X is batched.

    Returns:
        The Bi-Laplacian of the function f at the point X, evaluated
        for each element f[i](X). Has same shape as f(X).
    """
    # compute the derivative tensor of fourth derivatives
    # TODO Check if changing the inner Hessian to rev-rev improves performance
    # TODO Look into making this more efficient by directly contracting with the
    # correct unit vectors, similar to the vHv-approach for the randomized Laplacian.
    d4f = hessian(hessian(f))

    # trace it using einsum to support functions with non-scalar outputs
    num_summed_dims = X.ndim - 1 if is_batched else X.ndim
    dims1 = " ".join([f"i{i}" for i in range(num_summed_dims)])
    dims2 = " ".join([f"j{j}" for j in range(num_summed_dims)])
    # if x is a vector, this is just '... i i j j -> ...' where '...' corresponds
    # to the shape of f(x)
    equation = f"... {dims1} {dims1} {dims2} {dims2} -> ..."

    def _bilaplacian(x: Tensor) -> Tensor:
        """Compute the Bi-Laplacian on an unbatched input.

        Args:
            x: The input tensor.

        Returns:
            The Bi-Laplacian of the function f at the point x. Has same shape as f(x).
        """
        return einsum(d4f(x), equation)

    if is_batched:
        _bilaplacian = vmap(_bilaplacian)

    return _bilaplacian(X)


def bilaplacian_naive(
    f: Callable[[Tensor], Tensor], X: Tensor, detach: bool = True
) -> Tensor:
    """Naive implementation of the Bi-Laplacian via a for-loop.

    Args:
        f: The function to compute the Bi-Laplacian of.
        X: The point at which to compute the Bi-Laplacian. Must have `requires_grad`.
        detach: Whether to detach the result from the compute graph. Default: `True`.

    Returns:
        The Bi-Laplacian of the function f at the point X, evaluated
        for each element f[i](X). Has same shape as f(X).
    """
    D = X.numel()
    f_X = f(X)
    bilap = zeros_like(f_X).flatten()

    grad_kwargs = {
        "allow_unused": True,
        "materialize_grads": True,
        "create_graph": True,
    }

    def _maybe_grad(f: Tensor, X: Tensor) -> Tensor:
        """Compute the gradient if f requires grad, otherwise return zeros.

        Args:
            f: The function output for which to compute the gradient.
            X: The input tensor at which to compute the gradient.

        Returns:
            The gradient of f w.r.t. X if f requires grad, otherwise a tensor of zeros.
            Has the same shape as X.
        """
        return grad(f, X, **grad_kwargs)[0] if f.requires_grad else zeros_like(X)

    # loop over all components of f(X) and compute their bi-Laplacian
    for k, f_k in enumerate(f_X.flatten()):

        # take the second derivative w.r.t. xᵢ
        for i in range(D):
            e_i = zeros(D, dtype=X.dtype, device=X.device)
            e_i[i] = 1
            e_i = e_i.view_as(X)

            df = _maybe_grad(f_k, X)
            d2f = _maybe_grad((df * e_i).sum(), X)

            # the third derivative can be re-cycled for all xⱼ
            d3f = _maybe_grad((d2f * e_i).sum(), X)

            # differentiate twice w.r.t. xⱼ
            for j in range(D):
                e_j = zeros(D, dtype=X.dtype, device=X.device)
                e_j[j] = 1
                e_j = e_j.view_as(X)

                d4f = _maybe_grad((d3f * e_j).sum(), X)
                d4f_ii_jj = (d4f * e_j).sum()

                bilap[k] += d4f_ii_jj.detach() if detach else d4f_ii_jj

    return bilap.reshape_as(f_X)


@mark.parametrize("config", BILAPLACIAN_CASES, ids=BILAPLACIAN_IDS)
def test_bilaplacian(config: Dict[str, Any]):
    """Compare Laplacian implementations.

    Args:
        config: Configuration dictionary of the test case.
    """
    f, x, _, is_batched = setup_case(config, taylor_coefficients=False)

    # reference: Using PyTorch's autograd
    bilap_rev = bilaplacian_naive(f, x.clone().requires_grad_())

    # using torch.func
    bilap_func = bilaplacian(f, x, is_batched)
    report_nonclose(bilap_rev, bilap_func, name="Naive and functorch Bi-Laplacians")

    # using jets
    bilap_mod = Bilaplacian(f, x, is_batched)
    bilap_jet = bilap_mod(x)
    report_nonclose(bilap_rev, bilap_jet, name="Naive and jet Bi-Laplacians")


@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
@mark.parametrize("config", BILAPLACIAN_CASES, ids=BILAPLACIAN_IDS)
def test_RandomizedBilaplacian(
    config: Dict[str, Any],
    distribution: str,
    max_num_chunks: int = 500,
    chunk_size: int = 1_024,
    target_rel_error: float = 1e-2,
):
    """Test convergence of the Bi-Laplacian's Monte-Carlo estimator.

    Args:
        config: Configuration dictionary of the test case.
        distribution: The distribution from which to draw random vectors.
        max_num_chunks: Maximum number of chunks to accumulate. Default: `500`.
        chunk_size: Number of samples per chunk. Default: `1_024`.
        target_rel_error: Target relative error for convergence. Default: `1e-2`.
    """
    f, x, _, is_batched = setup_case(config, taylor_coefficients=False)

    # reference: Using PyTorch
    bilap = bilaplacian(f, x, is_batched)

    # check convergence of MC estimator
    def sample(idx: int) -> Tensor:
        manual_seed(idx)
        return RandomizedBilaplacian(f, x, is_batched, chunk_size, distribution)(x)

    converged = _check_mc_convergence(
        bilap, sample, chunk_size, max_num_chunks, target_rel_error
    )
    assert converged, f"Monte-Carlo Bi-Laplacian ({distribution}) did not converge."
