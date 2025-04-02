"""Tests for exp01 (Laplacian benchmark)."""

from test.test_laplacian import _check_mc_convergence, laplacian
from typing import Callable, Tuple

from pytest import mark
from torch import Tensor, manual_seed, rand

from jet.exp.exp01_benchmark_laplacian.execute import (
    SUPPORTED_STRATEGIES,
    laplacian_function,
    randomized_laplacian_function,
    setup_architecture,
)
from jet.laplacian import RandomizedLaplacian

STRATEGY_IDS = [f"strategy={s}" for s in SUPPORTED_STRATEGIES]
DISTRIBUTION_IDS = [
    f"distribution={d}" for d in RandomizedLaplacian.SUPPORTED_DISTRIBUTIONS
]


def setup() -> Tuple[Callable[[Tensor], Tensor], Tensor, bool]:
    """Set up the test case.

    Returns:
        The function, input tensor, and whether the input tensor is batched.
    """
    manual_seed(0)

    dim = 5
    f = setup_architecture("tanh_mlp_768_768_512_512_1", dim).double()
    batch_size = 3
    is_batched = True
    X = (rand(batch_size, dim) if is_batched else rand(dim)).double()

    return f, X, is_batched


@mark.parametrize("strategy", SUPPORTED_STRATEGIES, ids=STRATEGY_IDS)
def test_laplacian_functions(strategy: str):
    """Test that the benchmarked Laplacian functions produce the correct result.

    Args:
        strategy: The strategy to test.
    """
    f, X, is_batched = setup()
    lap = laplacian(f, X)
    lap_func = laplacian_function(f, X, is_batched, strategy)()

    assert lap.allclose(lap_func)


@mark.parametrize(
    "distribution", RandomizedLaplacian.SUPPORTED_DISTRIBUTIONS, ids=DISTRIBUTION_IDS
)
def test_randomized_laplacian_functions_identical(
    distribution: str, num_samples: int = 42
):
    """Test that the benchmarked MC-Laplacian functions are identical when seeding.

    Args:
        distribution: The distribution from which to draw random vectors.
        num_samples: Number of samples to draw. Default: `42`.
    """
    f, X, is_batched = setup()

    laps = {}
    for strategy in SUPPORTED_STRATEGIES:
        manual_seed(1)
        laps[strategy] = randomized_laplacian_function(
            f, X, is_batched, strategy, distribution, num_samples
        )()

    first_key = list(laps.keys())[0]
    for key in laps:
        assert laps[first_key].allclose(laps[key])


@mark.parametrize("strategy", SUPPORTED_STRATEGIES, ids=STRATEGY_IDS)
@mark.parametrize(
    "distribution", RandomizedLaplacian.SUPPORTED_DISTRIBUTIONS, ids=DISTRIBUTION_IDS
)
def test_randomized_laplacian_functions_converge(
    strategy: str,
    distribution: str,
    max_num_chunks: int = 10,
    chunk_size: int = 42,
    target_rel_error: float = 5e-2,
):
    """Test that the benchmarked MC-Laplacian functions converge.

    Args:
        strategy: The strategy to test.
        distribution: The distribution from which to draw random vectors.
        max_num_chunks: Maximum number of chunks to accumulate. Default: `10`.
        chunk_size: Number of samples per chunk. Default: `42`.
        target_rel_error: Target relative error for convergence. Default: `5e-2`.
    """
    f, X, is_batched = setup()

    lap = laplacian(f, X)

    # check convergence of the Monte-Carlo estimator
    def sample(idx: int) -> Tensor:
        manual_seed(idx)
        return randomized_laplacian_function(
            f, X, is_batched, strategy, distribution, chunk_size
        )()

    converged = _check_mc_convergence(
        lap, sample, chunk_size, max_num_chunks, target_rel_error
    )
    assert converged, f"MC Laplacian ({strategy}, {distribution}) did not converge."
