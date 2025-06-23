"""Test the Laplacian."""

from functools import partial
from test.test___init__ import setup_case
from typing import Any, Callable, Dict

from pytest import mark
from torch import Tensor, manual_seed, sigmoid, zeros_like
from torch.func import hessian
from torch.linalg import norm
from torch.nn import Linear, Sequential, Tanh

from jet.laplacian import Laplacian, RandomizedLaplacian

DISTRIBUTIONS = RandomizedLaplacian.SUPPORTED_DISTRIBUTIONS
DISTRIBUTION_IDS = [f"distribution={d}" for d in DISTRIBUTIONS]

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
for config in LAPLACIAN_CASES:
    config["is_batched"] = config.get("is_batched", False)

LAPLACIAN_IDS = [config["id"] for config in LAPLACIAN_CASES]


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


@mark.parametrize("config", LAPLACIAN_CASES, ids=LAPLACIAN_IDS)
def test_Laplacian(config: Dict[str, Any]):
    """Compare Laplacian implementations.

    Args:
        config: Configuration dictionary of the test case.
    """
    f, x, _, is_batched = setup_case(config)

    # reference: Using PyTorch
    lap_rev = laplacian(f, x)

    # Using a manually-vmapped jet
    _, _, lap_mod = Laplacian(f, x, is_batched)(x)
    assert lap_rev.allclose(lap_mod), "Functorch and jet Laplacians do not match."


@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
@mark.parametrize("config", LAPLACIAN_CASES, ids=LAPLACIAN_IDS)
def test_RandomizedLaplacian(
    config: Dict[str, Any],
    distribution: str,
    max_num_chunks: int = 500,
    chunk_size: int = 1_024,
    target_rel_error: float = 1e-2,
):
    """Test convergence of the Laplacian's Monte-Carlo estimator.

    Args:
        config: Configuration dictionary of the test case.
        distribution: The distribution from which to draw random vectors.
        max_num_chunks: Maximum number of chunks to accumulate. Default: `500`.
        chunk_size: Number of samples per chunk. Default: `1_024`.
        target_rel_error: Target relative error for convergence. Default: `1e-2`.
    """
    f, x, _, is_batched = setup_case(config)

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
