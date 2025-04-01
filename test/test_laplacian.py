from functools import partial
from test.test___init__ import CASES_COMPACT, CASES_COMPACT_IDS, setup_case
from typing import Any, Callable, Dict

from pytest import mark
from torch import Tensor, manual_seed, zeros_like
from torch.autograd.functional import hessian
from torch.linalg import norm

from jet.laplacian import Laplacian, RandomizedLaplacian

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
        lap[i] = hessian(f_i, x.flatten()).trace()

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
    lap_rev = laplacian(f, x)

    # accumulate the MC-Laplacian over multiple chunks
    lap_mod = 0.0
    converged = False

    for i in range(max_num_chunks):
        manual_seed(i)
        _, _, lap_i = RandomizedLaplacian(f, x, is_batched, chunk_size, distribution)(x)
        # update the Monte-Carlo estimator with the current chunk
        lap_mod = (lap_mod * i + lap_i.detach()) / (i + 1.0)

        rel_error = (norm(lap_mod - lap_rev) / norm(lap_rev)).item()
        print(f"Relative error at {(i+1) * chunk_size} samples: {rel_error:.3e}.")

        # check for convergence
        if rel_error < target_rel_error:
            converged = True
            break

    assert converged, f"Monte-Carlo Laplacian ({distribution}) did not converge."
