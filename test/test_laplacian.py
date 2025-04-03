from functools import partial
from test.test___init__ import (
    CASES_COMPACT,
    CASES_COMPACT_IDS,
    report_nonclose,
    setup_case,
)
from typing import Any, Callable, Dict

from pytest import mark
from torch import Tensor, arange, einsum, manual_seed, rand, zeros, zeros_like
from torch.func import hessian, vmap
from torch.linalg import norm
from torch.nn import Linear, Sequential, Tanh

from jet.laplacian import HessianDotPSD, Laplacian, RandomizedLaplacian

DISTRIBUTIONS = ["normal", "rademacher"]
DISTRIBUTION_IDS = [f"distribution={d}" for d in DISTRIBUTIONS]


def laplacian(f: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Compute the Laplacian of a tensor-to-tensor function.

    Args:
        f: The function to compute the Laplacian of.
        x: The point at which to compute the Laplacian.

    Returns:
        The Laplacian of the function f at the point x, evaluated
        for each element f[i](x). Has same shape as f(x)
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
        print(f"Relative error at {(i+1) * chunk_size} samples: {rel_error:.3e}.")

        # check for convergence
        if rel_error < target_rel_error:
            converged = True
            break

    return converged


def test_HessianDotPSD():
    """Test computing dot products of the Hessian with a PSD matrix."""
    manual_seed(0)
    batch_size, is_batched = 6, True
    D = 5
    net = Sequential(Linear(D, 4), Tanh(), Linear(4, 1)).double()
    X = rand(batch_size, D).double()
    rank_C = 3

    def S_func(_):
        S = zeros(D, rank_C).double()
        idx = arange(rank_C)
        S[idx, idx] = arange(rank_C).double() + 1
        return S.unsqueeze(0).expand(batch_size, D, rank_C)

    # compute ground truth
    H = vmap(hessian(net))(X).squeeze(1)
    C = arange(D) + 1
    C[rank_C:] = 0
    C = C.double().square().diag()
    H_dot_C_truth = einsum("nij,ij->n", H, C).unsqueeze(-1)

    # use jets
    mod = HessianDotPSD(net, X, is_batched, S_func)
    _, _, H_dot_C = mod(X)

    report_nonclose(H_dot_C_truth, H_dot_C)
