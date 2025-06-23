"""Test the weighted Laplacian."""

from functools import partial
from test.test___init__ import report_nonclose, setup_case
from test.test_laplacian import _check_mc_convergence
from typing import Any, Callable, Dict

from einops import einsum
from pytest import mark
from torch import Tensor, manual_seed, sigmoid
from torch.func import hessian, vmap
from torch.nn import Linear, Sequential, Tanh

from jet.weighted_laplacian import (
    C_func_diagonal_increments,
    RandomizedWeightedLaplacian,
    WeightedLaplacian,
)

DISTRIBUTIONS = RandomizedWeightedLaplacian.SUPPORTED_DISTRIBUTIONS
DISTRIBUTION_IDS = [f"distribution={d}" for d in DISTRIBUTIONS]

# make generation of test cases deterministic
manual_seed(0)

WEIGHTED_LAPLACIAN_CASES = [
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
for config in WEIGHTED_LAPLACIAN_CASES:
    config["is_batched"] = config.get("is_batched", False)

WEIGHTED_LAPLACIAN_IDS = [config["id"] for config in WEIGHTED_LAPLACIAN_CASES]


def weighted_laplacian(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    is_batched: bool,
    C_func_diagonal_increments: Callable[[Tensor], Tensor],
) -> Tensor:
    """Compute the weighted Laplacian.

    Args:
        f: The function to compute the weighted Laplacian of.
        x: The point at which to compute the weighted Laplacian.
        is_batched: Whether the function and its input are batched.
        C_func_diagonal_increments: Function that computes the coefficient matrix C(x).
            If is_batched is True, then C(x) must return a tensor of shape
            (batch_size, *x.shape[1:], *x.shape[1:]), otherwise
            it must return a tensor of shape (*x.shape, *x.shape).

    Raises:
        ValueError: If the coefficient tensor has an unexpected shape.

    Returns:
        The weighted Laplacian of the function f at the point x, evaluated
        for each element f[i](x). Has same shape as f(x).
    """
    # compute the coefficient tensor
    C = C_func_diagonal_increments(x)

    # make sure it has the correct shape
    unbatched = x.shape[1:] if is_batched else x.shape
    C_shape = ((x.shape[0],) if is_batched else ()) + (*unbatched, *unbatched)
    if C.shape != C_shape:
        raise ValueError(
            f"Coefficient tensor C has shape {tuple(C.shape)}, expected {C_shape}."
        )

    # compute the Hessian
    H_func = hessian(f)
    if is_batched:
        H_func = vmap(H_func)
    H = H_func(x)

    # do the contraction with einsum to support non-vector functions
    in_dims1 = " ".join([f"i{i}" for i in range(len(unbatched))])
    in_dims2 = " ".join([f"j{j}" for j in range(len(unbatched))])
    batch_str = "n " if is_batched else ""
    equation = (
        f"{batch_str}... {in_dims1} {in_dims2}, "
        + f"{batch_str}{in_dims1} {in_dims2} -> {batch_str}..."
    )
    return einsum(H, C, equation)


@mark.parametrize("config", WEIGHTED_LAPLACIAN_CASES, ids=WEIGHTED_LAPLACIAN_IDS)
def test_WeightedLaplacian(config: Dict[str, Any]):
    """Test computing dot products of the Hessian with a PSD matrix.

    Use a diagonal coefficient tensor whose diagonal elements are
    increments of 1 starting from 1.

    Args:
        config: Configuration dictionary of the test case.
    """
    f, x, _, is_batched = setup_case(config)
    C_func = partial(C_func_diagonal_increments, is_batched=is_batched)

    # compute ground truth
    H_dot_C_truth = weighted_laplacian(f, x, is_batched, C_func)

    # use jets
    mod = WeightedLaplacian(f, x, is_batched, "diagonal_increments")
    _, _, H_dot_C = mod(x)

    report_nonclose(H_dot_C_truth, H_dot_C)


@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
@mark.parametrize("config", WEIGHTED_LAPLACIAN_CASES, ids=WEIGHTED_LAPLACIAN_IDS)
def test_RandomizedWeightedLaplacian(
    config: Dict[str, Any],
    distribution: str,
    max_num_chunks: int = 500,
    chunk_size: int = 1_024,
    target_rel_error: float = 1e-2,
):
    """Test convergence of the weighted Laplacian's Monte-Carlo estimator.

    Args:
        config: Configuration dictionary of the test case.
        distribution: The distribution from which to draw random vectors.
        max_num_chunks: Maximum number of chunks to accumulate. Default: `500`.
        chunk_size: Number of samples per chunk. Default: `1_024`.
        target_rel_error: Target relative error for convergence. Default: `1e-2`.
    """
    f, x, _, is_batched = setup_case(config)

    # reference: Using PyTorch
    C_func = partial(C_func_diagonal_increments, is_batched=is_batched)
    H_dot_C_truth = weighted_laplacian(f, x, is_batched, C_func)

    # check convergence of MC estimator
    def sample(idx: int) -> Tensor:
        manual_seed(idx)
        _, _, H_dot_C = RandomizedWeightedLaplacian(
            f, x, is_batched, chunk_size, distribution, "diagonal_increments"
        )(x)
        return H_dot_C

    converged = _check_mc_convergence(
        H_dot_C_truth, sample, chunk_size, max_num_chunks, target_rel_error
    )
    assert (
        converged
    ), f"Monte-Carlo weighted Laplacian ({distribution=}) did not converge."
