from functools import partial
from test.test___init__ import (
    CASES_COMPACT,
    CASES_COMPACT_IDS,
    report_nonclose,
    setup_case,
)
from typing import Any, Callable, Dict

from einops import einsum
from pytest import mark
from torch import Tensor, arange, manual_seed, zeros, zeros_like
from torch.func import hessian, vmap
from torch.linalg import norm

from jet.laplacian import HessianDotPSD, Laplacian, RandomizedLaplacian

DISTRIBUTIONS = ["normal", "rademacher"]
DISTRIBUTION_IDS = [f"distribution={d}" for d in DISTRIBUTIONS]


def laplacian(f: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Naively compute the Laplacian of a tensor-to-tensor function.

    Args:
        f: The function to compute the Laplacian of.
        x: The point at which to compute the Laplacian.

    Returns:
        The Laplacian of the function f at the point x, evaluated
        for each element f[i](x). Has same shape as f(x).
    """
    out = f(x)

    def f_flat(x_flat, i):
        return f(x_flat.reshape_as(x)).flatten()[i]

    lap = zeros_like(out).flatten()
    for i in range(out.numel()):
        f_i = partial(f_flat, i=i)
        lap[i] = hessian(f_i)(x.flatten()).trace()

    return lap.reshape_as(out)


@mark.parametrize("config", CASES_COMPACT, ids=CASES_COMPACT_IDS)
def test_Laplacian(config: Dict[str, Any]):
    """Compare Laplacian implementations.

    Args:
        config: Configuration dictionary of the test case.
    """
    f, x, _, is_batched = setup_case(config, taylor_coefficients=False)

    # reference: Using PyTorch
    lap_rev = laplacian(f, x)

    # Using a manually-vmapped jet
    _, _, lap_mod = Laplacian(f, x, is_batched)(x)
    assert lap_rev.allclose(lap_mod), "Functorch and jet Laplacians do not match."


@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
@mark.parametrize("config", CASES_COMPACT, ids=CASES_COMPACT_IDS)
def test_RandomizedLaplacian(
    config: Dict[str, Any],
    distribution: str,
    max_num_chunks: int = 500,
    chunk_size: int = 4_096,
    target_rel_error: float = 2e-3,
):
    """Test convergence of the Laplacian's Monte-Carlo estimator.

    Args:
        config: Configuration dictionary of the test case.
        distribution: The distribution from which to draw random vectors.
        max_num_chunks: Maximum number of chunks to accumulate. Default: `500`.
        chunk_size: Number of samples per chunk. Default: `4_096`.
        target_rel_error: Target relative error for convergence. Default: `2e-3`.
    """
    f, x, _, is_batched = setup_case(config, taylor_coefficients=False)

    # reference: Using PyTorch
    lap = laplacian(f, x)

    # check convergence of MC estimator
    def sample(idx: int) -> Tensor:
        manual_seed(idx)
        _, _, lap = RandomizedLaplacian(f, x, is_batched, chunk_size, distribution)(x)
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

    for i in range(max_num_chunks):
        estimate_i = sample(i)
        # update the Monte-Carlo estimator with the current chunk
        estimate = (estimate * i + estimate_i.detach()) / (i + 1.0)

        rel_error = (norm(estimate - truth) / norm(truth)).item()
        print(f"Relative error at {(i + 1) * chunk_size} samples: {rel_error:.3e}.")

        # check for convergence
        if rel_error < target_rel_error:
            converged = True
            break

    return converged


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


@mark.parametrize("config", CASES_COMPACT, ids=CASES_COMPACT_IDS)
def test_HessianDotPSD(config: Dict[str, Any]):
    """Test computing dot products of the Hessian with a PSD matrix.

    Use a diagonal coefficient tensor whose diagonal elements are
    increments of 1 starting from 1.

    Args:
        config: Configuration dictionary of the test case.
    """
    f, x, _, is_batched = setup_case(config, taylor_coefficients=False)

    D = (x.shape[1:] if is_batched else x.shape).numel()
    unbatched = x.shape[1:] if is_batched else x.shape
    rank_C = D

    # define the coefficient functions
    def S_func_diagonal_increments(_: Tensor) -> Tensor:
        S = zeros(D, rank_C, dtype=x.dtype, device=x.device)
        idx = arange(rank_C, device=x.device)
        S[idx, idx] = (arange(rank_C, dtype=x.dtype, device=x.device) + 1).sqrt()
        S = S.reshape(*unbatched, rank_C)
        if is_batched:
            batch_size = x.shape[0]
            S = S.unsqueeze(0).expand(batch_size, *unbatched, rank_C)
        return S

    def C_func_diagonal_increments(_: Tensor) -> Tensor:
        C = zeros(D, dtype=x.dtype, device=x.device)
        C[:rank_C] = arange(rank_C, dtype=x.dtype, device=x.device) + 1
        C = C.diag()
        C = C.reshape(*unbatched, *unbatched)
        if is_batched:
            batch_size = x.shape[0]
            C = C.unsqueeze(0).expand(batch_size, *unbatched, *unbatched)
        return C

    # compute ground truth
    H_dot_C_truth = weighted_laplacian(f, x, is_batched, C_func_diagonal_increments)

    # use jets
    mod = HessianDotPSD(f, x, is_batched, S_func_diagonal_increments)
    _, _, H_dot_C = mod(x)

    report_nonclose(H_dot_C_truth, H_dot_C)
