from functools import partial
from test.test___init__ import CASES_COMPACT, CASES_COMPACT_IDS, setup_case
from typing import Any, Callable, Dict, Optional

from pytest import mark
from torch import Tensor, manual_seed, zeros_like
from torch.autograd.functional import hessian
from torch.linalg import norm

from jet.laplacian import Laplacian

RANDOMIZE = [None, "normal", "rademacher"]
RANDOMIZE_IDS = ["exact", "randomized_normal", "randomized_rademacher"]


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
@mark.parametrize("randomize", RANDOMIZE, ids=RANDOMIZE_IDS)
def test_laplacian(config: Dict[str, Any], randomize: Optional[str]):
    """Compare Laplacian implementations.

    Args:
        config: Configuration dictionary of the test case.
        randomize: Whether to randomize the Laplacian, and how.
    """
    f, x, _, is_batched = setup_case(config, taylor_coefficients=False)

    # reference: Using PyTorch
    lap_rev = laplacian(f, x)

    if randomize is None:
        # Using a manually-vmapped jet
        _, _, lap_mod = Laplacian(f, x, is_batched=is_batched, randomize=randomize)(x)
        assert lap_rev.allclose(lap_mod), "Functorch and jet Laplacians do not match."

    else:
        # total number of MC samples equals the product of these two numbers
        max_num_chunks = 500
        chunk_size = 4_096
        # relative error of || L - L_MC || / || L || detected as converged
        target_rel_error = 2e-3

        # accumulate the MC-Laplacian over multiple chunks
        lap_mod = 0.0
        converged = False

        for i in range(max_num_chunks):
            manual_seed(i)
            _, _, lap_i = Laplacian(
                f, x, is_batched=is_batched, randomize=(randomize, chunk_size)
            )(x)
            # update the Monte-Carlo estimator with the current chunk
            lap_mod = (lap_mod * i + lap_i.detach()) / (i + 1.0)

            rel_error = (norm(lap_mod - lap_rev) / norm(lap_rev)).item()
            print(f"Relative error at {(i+1) * chunk_size} samples: {rel_error:.3e}.")

            # check for convergence
            if rel_error < target_rel_error:
                converged = True
                break

        assert converged, f"Monte-Carlo Laplacian ({randomize}) did not converge."
