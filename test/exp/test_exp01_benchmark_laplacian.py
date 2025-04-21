"""Tests for exp01 (Laplacian benchmark)."""

from functools import partial
from test.test___init__ import (
    CASES_COMPACT,
    CASES_COMPACT_IDS,
    report_nonclose,
    setup_case,
)
from test.test_bilaplacian import bilaplacian
from test.test_laplacian import _check_mc_convergence, laplacian
from test.test_weighted_laplacian import weighted_laplacian
from typing import Any, Dict

from pytest import mark
from torch import Tensor, manual_seed

from jet.exp.exp01_benchmark_laplacian.execute import (
    SUPPORTED_STRATEGIES,
    bilaplacian_function,
    laplacian_function,
    randomized_laplacian_function,
    randomized_weighted_laplacian_function,
    weighted_laplacian_function,
)
from jet.laplacian import RandomizedLaplacian
from jet.weighted_laplacian import (
    C_func_diagonal_increments,
    RandomizedWeightedLaplacian,
)

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
    max_num_chunks: int = 128,
    chunk_size: int = 64,
    target_rel_error: float = 5e-2,
):
    """Test that the benchmarked MC-Laplacian functions converge.

    Args:
        config: Configuration dictionary of the test case.
        strategy: The strategy to test.
        distribution: The distribution from which to draw random vectors.
        max_num_chunks: Maximum number of chunks to accumulate. Default: `128`.
        chunk_size: Number of samples per chunk. Default: `64`.
        target_rel_error: Target relative error for convergence. Default: `5e-2`.
    """
    f, X, _, is_batched = setup_case(config, taylor_coefficients=False)

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


@mark.parametrize("strategy", SUPPORTED_STRATEGIES, ids=STRATEGY_IDS)
@mark.parametrize("config", CASES_COMPACT, ids=CASES_COMPACT_IDS)
def test_weighted_laplacian_functions(config: Dict[str, Any], strategy: str):
    """Test that the benchmarked weighted Laplacians produce the correct result.

    Args:
        config: Configuration dictionary of the test case.
        strategy: The strategy to test.
    """
    f, x, _, is_batched = setup_case(config, taylor_coefficients=False)
    C_func = partial(C_func_diagonal_increments, is_batched=is_batched)
    weighted_lap = weighted_laplacian(f, x, is_batched, C_func)
    weighted_lap_func = weighted_laplacian_function(f, x, is_batched, strategy)()

    report_nonclose(weighted_lap, weighted_lap_func)


@mark.parametrize(
    "distribution",
    RandomizedWeightedLaplacian.SUPPORTED_DISTRIBUTIONS,
    ids=DISTRIBUTION_IDS,
)
@mark.parametrize("config", CASES_COMPACT, ids=CASES_COMPACT_IDS)
def test_randomized_weighted_laplacian_functions_identical(
    config: Dict[str, Any], distribution: str, num_samples: int = 42
):
    """Test that the weighted MC-Laplacian functions are identical when seeding.

    Args:
        config: Configuration dictionary of the test case.
        distribution: The distribution from which to draw random vectors.
        num_samples: Number of samples to draw. Default: `42`.
    """
    f, x, _, is_batched = setup_case(config, taylor_coefficients=False)

    h_dot_cs = {}
    for strategy in SUPPORTED_STRATEGIES:
        manual_seed(1)
        h_dot_cs[strategy] = randomized_weighted_laplacian_function(
            f, x, is_batched, strategy, distribution, num_samples
        )()

    first_key = list(h_dot_cs.keys())[0]
    for key in h_dot_cs:
        report_nonclose(h_dot_cs[first_key], h_dot_cs[key])


@mark.parametrize("strategy", SUPPORTED_STRATEGIES, ids=STRATEGY_IDS)
@mark.parametrize(
    "distribution",
    RandomizedWeightedLaplacian.SUPPORTED_DISTRIBUTIONS,
    ids=DISTRIBUTION_IDS,
)
@mark.parametrize("config", CASES_COMPACT, ids=CASES_COMPACT_IDS)
def test_randomized_weighted_laplacian_functions_converge(
    config: Dict[str, Any],
    strategy: str,
    distribution: str,
    max_num_chunks: int = 128,
    chunk_size: int = 64,
    target_rel_error: float = 5e-2,
):
    """Test that the benchmarked weighted MC-Laplacian functions converge.

    Args:
        config: Configuration dictionary of the test case.
        strategy: The strategy to test.
        distribution: The distribution from which to draw random vectors.
        max_num_chunks: Maximum number of chunks to accumulate. Default: `128`.
        chunk_size: Number of samples per chunk. Default: `64`.
        target_rel_error: Target relative error for convergence. Default: `5e-2`.
    """
    f, X, _, is_batched = setup_case(config, taylor_coefficients=False)

    C_func = partial(C_func_diagonal_increments, is_batched=is_batched)
    h_dot_c = weighted_laplacian(f, X, is_batched, C_func)

    # check convergence of the Monte-Carlo estimator
    def sample(idx: int) -> Tensor:
        manual_seed(idx)
        return randomized_weighted_laplacian_function(
            f, X, is_batched, strategy, distribution, chunk_size
        )()

    converged = _check_mc_convergence(
        h_dot_c, sample, chunk_size, max_num_chunks, target_rel_error
    )
    assert (
        converged
    ), f"MC weighted Laplacian ({strategy}, {distribution}) did not converge."


@mark.parametrize("strategy", SUPPORTED_STRATEGIES, ids=STRATEGY_IDS)
@mark.parametrize("config", CASES_COMPACT, ids=CASES_COMPACT_IDS)
def test_bilaplacian_functions(config: Dict[str, Any], strategy: str):
    """Test that the benchmarked Bi-Laplacians produce the correct result.

    Args:
        config: Configuration dictionary of the test case.
        strategy: The strategy to test.
    """
    f, x, _, is_batched = setup_case(config, taylor_coefficients=False)
    bilap = bilaplacian(f, x, is_batched)
    bilap_func = bilaplacian_function(f, x, is_batched, strategy)()

    report_nonclose(bilap, bilap_func)
