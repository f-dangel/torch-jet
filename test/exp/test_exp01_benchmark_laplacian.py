"""Tests for exp01 (Laplacian benchmark)."""

from typing import Any

from pytest import mark
from torch import manual_seed, rand, sigmoid, vmap
from torch.nn import Linear, Sequential, Tanh
from torch.testing import assert_close

from jet.bilaplacian import (
    SUPPORTED_DISTRIBUTIONS as BILAPLACIAN_SUPPORTED_DISTRIBUTIONS,
)
from jet.exp.exp01_benchmark_laplacian.execute import (
    SUPPORTED_STRATEGIES,
    bilaplacian_function,
    laplacian_function,
)
from jet.laplacian import SUPPORTED_DISTRIBUTIONS as LAPLACIAN_SUPPORTED_DISTRIBUTIONS
from jet.utils import run_seeded
from jet.weighted_laplacian import get_weighting
from test.test_bilaplacian import bilaplacian
from test.test_laplacian import (
    WEIGHT_IDS,
    WEIGHTS,
    _check_mc_convergence,
    get_coefficients,
    laplacian,
)
from test.utils import setup_case

STRATEGY_IDS = [f"strategy={s}" for s in SUPPORTED_STRATEGIES]
LAPLACIAN_DISTRIBUTION_IDS = [
    f"distribution={d}" for d in LAPLACIAN_SUPPORTED_DISTRIBUTIONS
]
BILAPLACIAN_DISTRIBUTION_IDS = [
    f"distribution={d}" for d in BILAPLACIAN_SUPPORTED_DISTRIBUTIONS
]

# make generation of test cases deterministic
manual_seed(0)

#: Batch size used by all exp01 tests. The benchmark always runs batched, so
#: the tests run batched too -- the cases below bake the batch dimension into
#: their ``mock_args_fn``.
BATCH_SIZE = 2

EXP01_CASES = [
    # 5d tanh-activated two-layer MLP
    {
        "f": Sequential(
            Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
        ).double(),
        "mock_args_fn": lambda: (rand(BATCH_SIZE, 5).double(),),
        "id": "two-layer-tanh-mlp",
    },
    # 3d sigmoid(sigmoid) function
    {
        "f": lambda x: sigmoid(sigmoid(x)),
        "mock_args_fn": lambda: (rand(BATCH_SIZE, 3).double(),),
        "id": "sigmoid-sigmoid",
    },
]
EXP01_IDS = [config["id"] for config in EXP01_CASES]


@mark.parametrize("weights", WEIGHTS, ids=WEIGHT_IDS)
@mark.parametrize("strategy", SUPPORTED_STRATEGIES, ids=STRATEGY_IDS)
@mark.parametrize("config", EXP01_CASES, ids=EXP01_IDS)
def test_laplacian_functions(
    config: dict[str, Any],
    strategy: str,
    weights: str | None | tuple[str, float],
):
    """Test that the benchmarked Laplacian functions produce the correct result.

    Args:
        config: Configuration dictionary of the test case.
        strategy: The strategy to test.
        weights: The weighting to use for the Laplacian. If `None`, the Laplacian is
            unweighted. If `diagonal_increments`, a synthetic coefficient tensor is
            used that has diagonal elements that are increments of 1 starting from 1.
    """
    f, x = setup_case(config)

    C = vmap(lambda x: get_coefficients(x, weights))(x)
    lap_func = vmap(lambda x, C: laplacian(f, x, C))
    lap = lap_func(x, C)

    weighting = get_weighting(x[0], weights)
    lap_func = laplacian_function(
        f, x, strategy, randomization=None, weighting=weighting
    )()

    assert_close(lap, lap_func)


@mark.parametrize("weights", WEIGHTS, ids=WEIGHT_IDS)
@mark.parametrize(
    "distribution", LAPLACIAN_SUPPORTED_DISTRIBUTIONS, ids=LAPLACIAN_DISTRIBUTION_IDS
)
@mark.parametrize("config", EXP01_CASES, ids=EXP01_IDS)
def test_randomized_laplacian_functions_identical(
    config: dict[str, Any],
    distribution: str,
    weights: str | None | tuple[str, float],
    num_samples: int = 42,
):
    """Test that the benchmarked MC-Laplacian functions are identical when seeding.

    Args:
        config: Configuration dictionary of the test case.
        distribution: The distribution from which to draw random vectors.
        num_samples: Number of samples to draw. Default: `42`.
        weights: The weighting to use for the Laplacian. If `None`, the Laplacian is
            unweighted. If `diagonal_increments`, a synthetic coefficient tensor is
            used that has diagonal elements that are increments of 1 starting from 1.
    """
    f, x = setup_case(config)

    randomization = (distribution, num_samples)
    weighting = get_weighting(x[0], weights, randomization=randomization)

    laps = {}
    for strategy in SUPPORTED_STRATEGIES:
        lap_fn = laplacian_function(
            f, x, strategy, randomization=randomization, weighting=weighting
        )
        laps[strategy] = run_seeded(lap_fn, 42)

    # same seed must yield identical results across strategies
    first_key = list(laps.keys())[0]
    reference = laps[first_key]
    for value in laps.values():
        assert_close(reference, value)

    # different seed must yield a different result (skip for rademacher:
    # v_i^2 = 1 makes the estimator exact for diagonal Hessians / rank-deficient
    # weights, so different seeds can produce identical results)
    if distribution != "rademacher":
        lap_fn = laplacian_function(
            f, x, first_key, randomization=randomization, weighting=weighting
        )
        lap_seed_a = run_seeded(lap_fn, 42)
        lap_seed_b = run_seeded(lap_fn, 999)
        assert not lap_seed_a.allclose(lap_seed_b)


@mark.parametrize("weights", WEIGHTS, ids=WEIGHT_IDS)
@mark.parametrize("strategy", SUPPORTED_STRATEGIES, ids=STRATEGY_IDS)
@mark.parametrize(
    "distribution",
    LAPLACIAN_SUPPORTED_DISTRIBUTIONS,
    ids=LAPLACIAN_DISTRIBUTION_IDS,
)
@mark.parametrize("config", EXP01_CASES, ids=EXP01_IDS)
def test_randomized_laplacian_functions_converge(
    config: dict[str, Any],
    strategy: str,
    distribution: str,
    weights: str | None | tuple[str, float],
    max_num_chunks: int = 128,
    chunk_size: int = 128,
    target_rel_error: float = 5e-2,
):
    """Test that the benchmarked MC-Laplacian functions converge.

    Args:
        config: Configuration dictionary of the test case.
        strategy: The strategy to test.
        distribution: The distribution from which to draw random vectors.
        weights: The weighting to use for the Laplacian. If `None`, the Laplacian is
            unweighted. If `diagonal_increments`, a synthetic coefficient tensor is
            used that has diagonal elements that are increments of 1 starting from 1.
        max_num_chunks: Maximum number of chunks to accumulate. Default: `128`.
        chunk_size: Number of samples per chunk. Default: `64`.
        target_rel_error: Target relative error for convergence. Default: `5e-2`.
    """
    f, X = setup_case(config)

    C = vmap(lambda x: get_coefficients(x, weights))(X)
    lap_func = vmap(lambda x, C: laplacian(f, x, C))
    lap = lap_func(X, C)

    randomization = (distribution, chunk_size)
    weighting = get_weighting(X[0], weights, randomization=randomization)

    # check convergence of the Monte-Carlo estimator
    fn = laplacian_function(
        f, X, strategy, randomization=randomization, weighting=weighting
    )

    converged = _check_mc_convergence(
        lap,
        lambda idx: run_seeded(fn, idx),
        chunk_size,
        max_num_chunks,
        target_rel_error,
    )
    assert converged, f"MC Laplacian ({strategy}, {distribution}) did not converge."


@mark.parametrize("strategy", SUPPORTED_STRATEGIES, ids=STRATEGY_IDS)
@mark.parametrize("config", EXP01_CASES, ids=EXP01_IDS)
def test_bilaplacian_functions(config: dict[str, Any], strategy: str):
    """Test that the benchmarked Bi-Laplacians produce the correct result.

    Args:
        config: Configuration dictionary of the test case.
        strategy: The strategy to test.
    """
    f, x = setup_case(config)
    bilap_func = vmap(lambda x: bilaplacian(f, x))
    bilap = bilap_func(x)

    bilap_func = bilaplacian_function(f, x, strategy)()

    assert_close(bilap, bilap_func)


@mark.parametrize(
    "distribution",
    BILAPLACIAN_SUPPORTED_DISTRIBUTIONS,
    ids=BILAPLACIAN_DISTRIBUTION_IDS,
)
@mark.parametrize("config", EXP01_CASES, ids=EXP01_IDS)
def test_randomized_bilaplacian_functions_identical(
    config: dict[str, Any], distribution: str, num_samples: int = 42
):
    """Test that the weighted MC-Bi-Laplacian functions are identical when seeding.

    Args:
        config: Configuration dictionary of the test case.
        distribution: The distribution from which to draw random vectors.
        num_samples: Number of samples to draw. Default: `42`.
    """
    f, x = setup_case(config)
    randomization = (distribution, num_samples)

    bilaps = {}
    for strategy in SUPPORTED_STRATEGIES:
        bilap_fn = bilaplacian_function(f, x, strategy, randomization=randomization)
        bilaps[strategy] = run_seeded(bilap_fn, 42)

    # same seed must yield identical results across strategies
    first_key = list(bilaps.keys())[0]
    reference = bilaps[first_key]
    for value in bilaps.values():
        assert_close(reference, value)

    # different seed must yield a different result (skip for rademacher:
    # v_i^2 = 1 makes the estimator exact for diagonal Hessians / rank-deficient
    # weights, so different seeds can produce identical results)
    if distribution != "rademacher":
        bilap_fn = bilaplacian_function(f, x, first_key, randomization=randomization)
        bilap_seed_a = run_seeded(bilap_fn, 42)
        bilap_seed_b = run_seeded(bilap_fn, 999)
        assert not bilap_seed_a.allclose(bilap_seed_b)


@mark.parametrize("strategy", SUPPORTED_STRATEGIES, ids=STRATEGY_IDS)
@mark.parametrize(
    "distribution",
    BILAPLACIAN_SUPPORTED_DISTRIBUTIONS,
    ids=BILAPLACIAN_DISTRIBUTION_IDS,
)
@mark.parametrize("config", EXP01_CASES, ids=EXP01_IDS)
def test_randomized_bilaplacian_functions_converge(
    config: dict[str, Any],
    strategy: str,
    distribution: str,
    max_num_chunks: int = 128,
    chunk_size: int = 128,
    target_rel_error: float = 5e-2,
):
    """Test that the benchmarked MC-Bi-Laplacian functions converge.

    Args:
        config: Configuration dictionary of the test case.
        strategy: The strategy to test.
        distribution: The distribution from which to draw random vectors.
        max_num_chunks: Maximum number of chunks to accumulate. Default: `128`.
        chunk_size: Number of samples per chunk. Default: `128`.
        target_rel_error: Target relative error for convergence. Default: `5e-2`.
    """
    f, X = setup_case(config)
    randomization = (distribution, chunk_size)

    bilap_func = vmap(lambda x: bilaplacian(f, x))
    bilap = bilap_func(X)

    # check convergence of the Monte-Carlo estimator
    fn = bilaplacian_function(f, X, strategy, randomization=randomization)

    converged = _check_mc_convergence(
        bilap,
        lambda idx: run_seeded(fn, idx),
        chunk_size,
        max_num_chunks,
        target_rel_error,
    )
    assert converged, f"MC-Bi-Laplacian ({strategy}, {distribution}) did not converge."
