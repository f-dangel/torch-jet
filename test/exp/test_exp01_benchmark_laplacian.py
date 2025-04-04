"""Tests for exp01 (Laplacian benchmark)."""

from test.test___init__ import (
    CASES_COMPACT,
    CASES_COMPACT_IDS,
    report_nonclose,
    setup_case,
)
from test.test_laplacian import _check_mc_convergence, laplacian
from typing import Any, Callable, Dict

from pytest import mark, skip
from torch import Tensor, manual_seed

from jet.exp.exp01_benchmark_laplacian.execute import (
    SUPPORTED_STRATEGIES,
    laplacian_function,
    randomized_laplacian_function,
)
from jet.laplacian import RandomizedLaplacian

STRATEGY_IDS = [f"strategy={s}" for s in SUPPORTED_STRATEGIES]
DISTRIBUTION_IDS = [
    f"distribution={d}" for d in RandomizedLaplacian.SUPPORTED_DISTRIBUTIONS
]


@mark.parametrize("strategy", SUPPORTED_STRATEGIES, ids=STRATEGY_IDS)
@mark.parametrize("config", CASES_COMPACT, ids=CASES_COMPACT_IDS)
def test_laplacian_functions(config: Dict[str, Any], strategy: str):
    """Test that the benchmarked Laplacian functions produce the correct result.

    Args:
        config: Configuration dictionary of the test case.
        strategy: The strategy to test.
    """
    f, x, _, is_batched = setup_case(config, taylor_coefficients=False)
    lap = laplacian(f, x)
    lap_func = laplacian_function(f, x, is_batched, strategy)()

    report_nonclose(lap, lap_func)


@mark.parametrize(
    "distribution", RandomizedLaplacian.SUPPORTED_DISTRIBUTIONS, ids=DISTRIBUTION_IDS
)
@mark.parametrize("config", CASES_COMPACT, ids=CASES_COMPACT_IDS)
def test_randomized_laplacian_functions_identical(
    config: Dict[str, Any], distribution: str, num_samples: int = 42
):
    """Test that the benchmarked MC-Laplacian functions are identical when seeding.

    Args:
        config: Configuration dictionary of the test case.
        distribution: The distribution from which to draw random vectors.
        num_samples: Number of samples to draw. Default: `42`.
    """
    f, x, _, is_batched = setup_case(config, taylor_coefficients=False)
    # NOTE The randomized functorch Laplacian only supports scalar output functions
    _skip_non_scalar_output(f, x, is_batched)

    laps = {}
    for strategy in SUPPORTED_STRATEGIES:
        manual_seed(1)
        laps[strategy] = randomized_laplacian_function(
            f, x, is_batched, strategy, distribution, num_samples
        )()

    first_key = list(laps.keys())[0]
    for key in laps:
        report_nonclose(laps[first_key], laps[key])


@mark.parametrize("strategy", SUPPORTED_STRATEGIES, ids=STRATEGY_IDS)
@mark.parametrize(
    "distribution", RandomizedLaplacian.SUPPORTED_DISTRIBUTIONS, ids=DISTRIBUTION_IDS
)
@mark.parametrize("config", CASES_COMPACT, ids=CASES_COMPACT_IDS)
def test_randomized_laplacian_functions_converge(
    config: Dict[str, Any],
    strategy: str,
    distribution: str,
    max_num_chunks: int = 10,
    chunk_size: int = 42,
    target_rel_error: float = 5e-2,
):
    """Test that the benchmarked MC-Laplacian functions converge.

    Args:
        config: Configuration dictionary of the test case.
        strategy: The strategy to test.
        distribution: The distribution from which to draw random vectors.
        max_num_chunks: Maximum number of chunks to accumulate. Default: `10`.
        chunk_size: Number of samples per chunk. Default: `42`.
        target_rel_error: Target relative error for convergence. Default: `5e-2`.
    """
    f, X, _, is_batched = setup_case(config, taylor_coefficients=False)
    # NOTE The randomized functorch Laplacian only supports scalar output functions
    _skip_non_scalar_output(f, X, is_batched)

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


def _skip_non_scalar_output(f: Callable[[Tensor], Tensor], x: Tensor, is_batched: bool):
    """Skip if the function does not produce a scalar output.

    This may be necessary because some of the implementations only support functions
    with scalar-valued outputs.

    Args:
        f: The function to test.
        x: The input tensor.
        is_batched: Whether the input tensor is batched.
    """
    out_shape = f(x).shape
    if (out_shape[1:] if is_batched else out_shape).numel() != 1:
        skip(f"Skipping non-scalar function with {tuple(out_shape)} output.")
