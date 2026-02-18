"""Tests for exp01 (Laplacian benchmark)."""

from typing import Any

from pytest import mark
from torch import Tensor, manual_seed, sigmoid, vmap
from torch.random import fork_rng
from torch.nn import Linear, Sequential, Tanh

from jet.bilaplacian import Bilaplacian
from jet.exp.exp01_benchmark_laplacian.execute import (
    SUPPORTED_STRATEGIES,
    bilaplacian_function,
    laplacian_function,
)
from jet.laplacian import Laplacian
from jet.weighted_laplacian import get_weighting
from test.test___init__ import report_nonclose, setup_case
from test.test_bilaplacian import bilaplacian
from test.test_laplacian import (
    WEIGHT_IDS,
    WEIGHTS,
    _check_mc_convergence,
    get_coefficients,
    laplacian,
)

STRATEGY_IDS = [f"strategy={s}" for s in SUPPORTED_STRATEGIES]
LAPLACIAN_DISTRIBUTION_IDS = [
    f"distribution={d}" for d in Laplacian.SUPPORTED_DISTRIBUTIONS
]
BILAPLACIAN_DISTRIBUTION_IDS = [
    f"distribution={d}" for d in Bilaplacian.SUPPORTED_DISTRIBUTIONS
]

# make generation of test cases deterministic
manual_seed(0)

EXP01_CASES = [
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
EXP01_IDS = [config["id"] for config in EXP01_CASES]

BATCH_SIZES = [0, 2]
BATCH_SIZE_IDS = ["datum", "batch"]


@mark.parametrize("batch_size", BATCH_SIZES, ids=BATCH_SIZE_IDS)
@mark.parametrize("weights", WEIGHTS, ids=WEIGHT_IDS)
@mark.parametrize("strategy", SUPPORTED_STRATEGIES, ids=STRATEGY_IDS)
@mark.parametrize("config", EXP01_CASES, ids=EXP01_IDS)
def test_laplacian_functions(
    config: dict[str, Any],
    strategy: str,
    weights: str | None | tuple[str, float],
    batch_size: int,
):
    """Test that the benchmarked Laplacian functions produce the correct result.

    Args:
        config: Configuration dictionary of the test case.
        strategy: The strategy to test.
        weights: The weighting to use for the Laplacian. If `None`, the Laplacian is
            unweighted. If `diagonal_increments`, a synthetic coefficient tensor is
            used that has diagonal elements that are increments of 1 starting from 1.
        batch_size: The batch size to use for the test. `0` means no batching.
    """
    f, x, _ = setup_case(config, vmapsize=batch_size)
    is_batched = batch_size > 0

    C = (
        vmap(lambda x: get_coefficients(x, weights))(x)
        if is_batched
        else get_coefficients(x, weights)
    )
    lap_func = lambda x, C: laplacian(f, x, C)  # noqa: E731
    lap_func = vmap(lap_func) if is_batched else lap_func
    lap = lap_func(x, C)

    weighting = get_weighting(x[0] if is_batched else x, weights)
    lap_func = laplacian_function(
        f, x, is_batched, strategy, randomization=None, weighting=weighting
    )()

    report_nonclose(lap, lap_func)


@mark.parametrize("batch_size", BATCH_SIZES, ids=BATCH_SIZE_IDS)
@mark.parametrize("weights", WEIGHTS, ids=WEIGHT_IDS)
@mark.parametrize(
    "distribution", Laplacian.SUPPORTED_DISTRIBUTIONS, ids=LAPLACIAN_DISTRIBUTION_IDS
)
@mark.parametrize("config", EXP01_CASES, ids=EXP01_IDS)
def test_randomized_laplacian_functions_identical(
    config: dict[str, Any],
    distribution: str,
    weights: str | None | tuple[str, float],
    batch_size: int,
    num_samples: int = 42,
):
    """Test that the benchmarked MC-Laplacian functions are identical when seeding.

    Args:
        config: Configuration dictionary of the test case.
        distribution: The distribution from which to draw random vectors.
        num_samples: Number of samples to draw. Default: `42`.
        batch_size: The batch size to use for the test. `0` means no batching.
        weights: The weighting to use for the Laplacian. If `None`, the Laplacian is
            unweighted. If `diagonal_increments`, a synthetic coefficient tensor is
            used that has diagonal elements that are increments of 1 starting from 1.
    """
    f, x, _ = setup_case(config, vmapsize=batch_size)
    is_batched = batch_size > 0

    randomization = (distribution, num_samples)
    weighting = get_weighting(
        x[0] if is_batched else x, weights, randomization=randomization
    )

    laps = {}
    for strategy in SUPPORTED_STRATEGIES:
        lap_fn = laplacian_function(
            f, x, is_batched, strategy, randomization=randomization, weighting=weighting
        )
        with fork_rng():
            manual_seed(42)
            laps[strategy] = lap_fn()

    first_key = list(laps.keys())[0]
    for key in laps:
        report_nonclose(laps[first_key], laps[key])


@mark.parametrize("batch_size", BATCH_SIZES, ids=BATCH_SIZE_IDS)
@mark.parametrize("weights", WEIGHTS, ids=WEIGHT_IDS)
@mark.parametrize("strategy", SUPPORTED_STRATEGIES, ids=STRATEGY_IDS)
@mark.parametrize(
    "distribution",
    Laplacian.SUPPORTED_DISTRIBUTIONS,
    ids=LAPLACIAN_DISTRIBUTION_IDS,
)
@mark.parametrize("config", EXP01_CASES, ids=EXP01_IDS)
def test_randomized_laplacian_functions_converge(
    config: dict[str, Any],
    strategy: str,
    distribution: str,
    weights: str | None | tuple[str, float],
    batch_size: int,
    max_num_chunks: int = 128,
    chunk_size: int = 128,
    target_rel_error: float = 5e-2,
):
    """Test that the benchmarked MC-Laplacian functions converge.

    Args:
        config: Configuration dictionary of the test case.
        strategy: The strategy to test.
        distribution: The distribution from which to draw random vectors.
        batch_size: The batch size to use for the test. `0` means no batching.
        weights: The weighting to use for the Laplacian. If `None`, the Laplacian is
            unweighted. If `diagonal_increments`, a synthetic coefficient tensor is
            used that has diagonal elements that are increments of 1 starting from 1.
        max_num_chunks: Maximum number of chunks to accumulate. Default: `128`.
        chunk_size: Number of samples per chunk. Default: `64`.
        target_rel_error: Target relative error for convergence. Default: `5e-2`.
    """
    f, X, _ = setup_case(config, vmapsize=batch_size)
    is_batched = batch_size > 0

    C = (
        vmap(lambda x: get_coefficients(x, weights))(X)
        if is_batched
        else get_coefficients(X, weights)
    )
    lap_func = lambda x, C: laplacian(f, x, C)  # noqa: E731
    lap_func = vmap(lap_func) if is_batched else lap_func
    lap = lap_func(X, C)

    randomization = (distribution, chunk_size)
    weighting = get_weighting(
        X[0] if is_batched else X, weights, randomization=randomization
    )

    # check convergence of the Monte-Carlo estimator
    def sample(idx: int) -> Tensor:
        fn = laplacian_function(
            f, X, is_batched, strategy, randomization=randomization, weighting=weighting
        )
        with fork_rng():
            manual_seed(idx)
            return fn()

    converged = _check_mc_convergence(
        lap, sample, chunk_size, max_num_chunks, target_rel_error
    )
    assert converged, f"MC Laplacian ({strategy}, {distribution}) did not converge."


@mark.parametrize("batch_size", BATCH_SIZES, ids=BATCH_SIZE_IDS)
@mark.parametrize("strategy", SUPPORTED_STRATEGIES, ids=STRATEGY_IDS)
@mark.parametrize("config", EXP01_CASES, ids=EXP01_IDS)
def test_bilaplacian_functions(config: dict[str, Any], strategy: str, batch_size: int):
    """Test that the benchmarked Bi-Laplacians produce the correct result.

    Args:
        config: Configuration dictionary of the test case.
        strategy: The strategy to test.
        batch_size: The batch size to use for the test. `0` means no batching.
    """
    f, x, _ = setup_case(config, vmapsize=batch_size)
    is_batched = batch_size > 0
    bilap_func = lambda x: bilaplacian(f, x)  # noqa: E731
    bilap_func = vmap(bilap_func) if is_batched else bilap_func
    bilap = bilap_func(x)

    bilap_func = bilaplacian_function(f, x, is_batched, strategy)()

    report_nonclose(bilap, bilap_func)


@mark.parametrize("batch_size", BATCH_SIZES, ids=BATCH_SIZE_IDS)
@mark.parametrize(
    "distribution",
    Bilaplacian.SUPPORTED_DISTRIBUTIONS,
    ids=BILAPLACIAN_DISTRIBUTION_IDS,
)
@mark.parametrize("config", EXP01_CASES, ids=EXP01_IDS)
def test_randomized_bilaplacian_functions_identical(
    config: dict[str, Any], distribution: str, batch_size: int, num_samples: int = 42
):
    """Test that the weighted MC-Bi-Laplacian functions are identical when seeding.

    Args:
        config: Configuration dictionary of the test case.
        distribution: The distribution from which to draw random vectors.
        batch_size: The batch size to use for the test. `0` means no batching.
        num_samples: Number of samples to draw. Default: `42`.
    """
    f, x, _ = setup_case(config, vmapsize=batch_size)
    is_batched = batch_size > 0
    randomization = (distribution, num_samples)

    bilaps = {}
    for strategy in SUPPORTED_STRATEGIES:
        bilap_fn = bilaplacian_function(
            f, x, is_batched, strategy, randomization=randomization
        )
        with fork_rng():
            manual_seed(42)
            bilaps[strategy] = bilap_fn()

    first_key = list(bilaps.keys())[0]
    for key in bilaps:
        report_nonclose(bilaps[first_key], bilaps[key])


@mark.parametrize("batch_size", BATCH_SIZES, ids=BATCH_SIZE_IDS)
@mark.parametrize("strategy", SUPPORTED_STRATEGIES, ids=STRATEGY_IDS)
@mark.parametrize(
    "distribution",
    Bilaplacian.SUPPORTED_DISTRIBUTIONS,
    ids=BILAPLACIAN_DISTRIBUTION_IDS,
)
@mark.parametrize("config", EXP01_CASES, ids=EXP01_IDS)
def test_randomized_bilaplacian_functions_converge(
    config: dict[str, Any],
    strategy: str,
    distribution: str,
    batch_size: int,
    max_num_chunks: int = 128,
    chunk_size: int = 128,
    target_rel_error: float = 5e-2,
):
    """Test that the benchmarked MC-Bi-Laplacian functions converge.

    Args:
        config: Configuration dictionary of the test case.
        strategy: The strategy to test.
        distribution: The distribution from which to draw random vectors.
        batch_size: The batch size to use for the test. `0` means no batching.
        max_num_chunks: Maximum number of chunks to accumulate. Default: `128`.
        chunk_size: Number of samples per chunk. Default: `128`.
        target_rel_error: Target relative error for convergence. Default: `5e-2`.
    """
    f, X, _ = setup_case(config, vmapsize=batch_size)
    is_batched = batch_size > 0
    randomization = (distribution, chunk_size)

    bilap_func = lambda x: bilaplacian(f, x)  # noqa: E731
    bilap_func = vmap(bilap_func) if is_batched else bilap_func
    bilap = bilap_func(X)

    # check convergence of the Monte-Carlo estimator
    def sample(idx: int) -> Tensor:
        fn = bilaplacian_function(
            f, X, is_batched, strategy, randomization=randomization
        )
        with fork_rng():
            manual_seed(idx)
            return fn()

    converged = _check_mc_convergence(
        bilap, sample, chunk_size, max_num_chunks, target_rel_error
    )
    assert converged, f"MC-Bi-Laplacian ({strategy}, {distribution}) did not converge."
