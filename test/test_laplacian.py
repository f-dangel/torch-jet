"""Test the Laplacian."""

from functools import partial
from test.test___init__ import setup_case
from typing import Any, Callable

from einops import einsum
from pytest import mark
from torch import Tensor, eye, manual_seed, sigmoid
from torch.func import hessian
from torch.linalg import norm
from torch.nn import Linear, Sequential, Tanh

from jet.laplacian import Laplacian
from jet.weighted_laplacian import (
    C_func_diagonal_increments,
    apply_S_func_diagonal_increments,
)

DISTRIBUTIONS = Laplacian.SUPPORTED_DISTRIBUTIONS
DISTRIBUTION_IDS = [f"distribution={d}" for d in DISTRIBUTIONS]

WEIGHTS = [None, "diagonal_increments"]
WEIGHT_IDS = ["standard-laplacian", "weighted-laplacian"]

# make generation of test cases deterministic
manual_seed(0)

LAPLACIAN_CASES = [
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

LAPLACIAN_IDS = [config["id"] for config in LAPLACIAN_CASES]


def laplacian(f: Callable[[Tensor], Tensor], x: Tensor, C: Tensor) -> Tensor:
    """Compute the (weighted) Laplacian of a tensor-to-tensor function.

    Args:
        f: The function to compute the Laplacian of.
        x: The point at which to compute the Laplacian.
        C: Coefficient tensor C(x) for weighting the Laplacian. Has shape
            `(*x.shape, *x.shape)`.

    Returns:
        The Laplacian of the function f at the point x, evaluated
        for each element f[i](x). Has same shape as f(x).
    """
    # compute the Hessian
    H = hessian(f)(x)

    # check the coefficient tensor
    C_shape = (*x.shape, *x.shape)
    if C.shape != C_shape:
        raise ValueError(
            f"Coefficient tensor C has shape {tuple(C.shape)}, expected {C_shape}."
        )

    # do the contraction with einsum to support non-vector functions
    in_dims1 = " ".join([f"i{i}" for i in range(x.ndim)])
    in_dims2 = " ".join([f"j{j}" for j in range(x.ndim)])
    equation = f"... {in_dims1} {in_dims2}, {in_dims1} {in_dims2} -> ..."

    return einsum(H, C, equation)


def get_weighting(
    x: Tensor, weights: str | None, randomization: tuple[str, int] | None = None
) -> tuple[Callable[[Tensor, Tensor], Tensor], int] | None:
    # determine the Laplacian's weighting
    if weights == "diagonal_increments":
        fx_info = {
            "in_shape": x.shape,
            "device": x.device,
            "dtype": x.dtype,
            "rank_C": x.numel(),
            "V_rows": x.numel() if randomization is None else randomization[1],
        }
        apply_weighting = partial(apply_S_func_diagonal_increments, fx_info=fx_info)
        rank_weighting = x.numel()
        return apply_weighting, rank_weighting
    else:
        assert weights is None
        return None


def get_coefficients(x: Tensor, weights: str | None) -> Tensor:
    if weights == "diagonal_increments":
        # Use a synthetic coefficient tensor C(x) with diagonal increments
        return C_func_diagonal_increments(x)
    assert weights is None
    return eye(x.numel(), x.numel(), device=x.device, dtype=x.dtype).reshape(
        *x.shape, *x.shape
    )


@mark.parametrize("weights", WEIGHTS, ids=WEIGHT_IDS)
@mark.parametrize("config", LAPLACIAN_CASES, ids=LAPLACIAN_IDS)
def test_Laplacian(config: dict[str, Any], weights: str | None):
    """Compare Laplacian implementations.

    Args:
        config: Configuration dictionary of the test case.
        weights: The weighting to use for the Laplacian. If `None`, the Laplacian is
            unweighted. If `diagonal_increments`, a synthetic coefficient tensor is
            used that has diagonal elements that are increments of 1 starting from 1.
    """
    f, x, _, _ = setup_case({**config, "is_batched": None})

    # reference: Using PyTorch
    C = get_coefficients(x, weights)
    lap_rev = laplacian(f, x, C)

    # Using a manually-vmapped jet
    weighting = get_weighting(x, weights)
    _, _, lap_mod = Laplacian(f, x, weighting=weighting)(x)
    assert lap_rev.allclose(lap_mod), "Functorch and jet Laplacians do not match."


@mark.parametrize("weights", WEIGHTS, ids=WEIGHT_IDS)
@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
@mark.parametrize("config", LAPLACIAN_CASES, ids=LAPLACIAN_IDS)
def test_Laplacian_randomization(
    config: dict[str, Any],
    distribution: str,
    weights: str | None,
    max_num_chunks: int = 100,
    chunk_size: int = 256,
    target_rel_error: float = 1e-2,
):
    """Test convergence of the Laplacian's Monte-Carlo estimator.

    Args:
        config: Configuration dictionary of the test case.
        distribution: The distribution from which to draw random vectors.
        weights: The weighting to use for the Laplacian. If `None`, the Laplacian is
            unweighted. If `diagonal_increments`, a synthetic coefficient tensor is
            used that has diagonal elements that are increments of 1 starting from 1.
        max_num_chunks: Maximum number of chunks to accumulate. Default: `500`.
        chunk_size: Number of samples per chunk. Default: `1_024`.
        target_rel_error: Target relative error for convergence. Default: `1e-2`.
    """
    f, x, _, _ = setup_case({**config, "is_batched": None})
    randomization = (distribution, chunk_size)

    # reference: Using PyTorch
    C = get_coefficients(x, weights)
    lap = laplacian(f, x, C)

    # check convergence of MC estimator
    weighting = get_weighting(x, weights, randomization=randomization)

    def sample(idx: int) -> Tensor:
        manual_seed(idx)
        _, _, lap = Laplacian(f, x, randomization=randomization, weighting=weighting)(x)
        return lap

    converged = _check_mc_convergence(
        lap, sample, chunk_size, max_num_chunks, target_rel_error
    )
    assert converged, f"Monte-Carlo Laplacian ({distribution}) did not converge."


def _check_mc_convergence(
    truth: Tensor,
    sample: Callable[[int], Tensor],
    chunk_size: int,
    max_num_chunks: int,
    target_rel_error: float,
) -> bool:
    """Check convergence of a Monte-Carlo estimator.

    Args:
        truth: Ground-truth value to compare the estimator to.
        sample: Function that draws a sample from the estimator and takes an
            integer seed as input.
        chunk_size: Number of samples used by one draw from `sample`.
        max_num_chunks: Maximum number of chunks to accumulate.
        target_rel_error: Target relative error for convergence.

    Returns:
        Whether the Monte-Carlo estimator has converged.
    """
    # accumulate the MC-estimator over multiple chunks
    estimate = 0.0
    converged = False

    norm_truth = norm(truth)
    if norm_truth == 0.0:  # add a small damping value
        norm_truth = 1e-10

    for i in range(max_num_chunks):
        estimate_i = sample(i)
        # update the Monte-Carlo estimator with the current chunk
        estimate = (estimate * i + estimate_i.detach()) / (i + 1.0)

        rel_error = (norm(estimate - truth) / norm_truth).item()
        print(f"Relative error at {(i + 1) * chunk_size} samples: {rel_error:.3e}.")

        # check for convergence
        if rel_error < target_rel_error:
            converged = True
            break

    return converged
