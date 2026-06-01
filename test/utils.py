"""Utility functions for testing."""

from typing import Any, Callable

from pytest import param
from torch import Tensor, float64, manual_seed, rand, rand_like, zeros, zeros_like
from torch.utils._pytree import tree_map

#: Valid ``(K, collapsed)`` pairs for the standard-vs-collapsed mode sweep.
#: ``K=0`` (primal-only) and ``K=1`` (Jacobian-vector product) are
#: standard-mode boundaries that exercise the no-recursion branches of the
#: jet rules; ``K=2`` is the collapsed-mode floor; ``K=5`` is a representative
#: high order. Intermediate orders exercise the same code paths and don't
#: earn their own coverage at the primitive / composition layer.
K_AND_MODE = [
    param(0, False, id="K=0-standard"),
    param(1, False, id="K=1-standard"),
    param(2, False, id="K=2-standard"),
    param(5, False, id="K=5-standard"),
    param(2, True, id="K=2-collapsed"),
    param(5, True, id="K=5-collapsed"),
]


def shape(*dims: int) -> Callable[[], tuple[Tensor]]:
    """Mock-args factory: one ``rand`` tensor of the given shape (float64)."""
    return lambda: (rand(*dims, dtype=float64),)


def shapes(*shape_pairs) -> Callable[[], tuple[Tensor, ...]]:
    """Mock-args factory: one ``rand`` tensor per shape (all float64)."""
    return lambda: tuple(rand(*s, dtype=float64) for s in shape_pairs)


def setup_case(
    config: dict[str, Any],
) -> tuple[Callable[..., Tensor], tuple[Tensor, ...]]:
    """Instantiate the function and its mock arguments.

    The mock arguments are taken verbatim from ``config["mock_args_fn"]()``
    -- if the case wants a batched input it should encode the batch dimension
    into its ``mock_args_fn`` directly.

    Args:
        config: Configuration dictionary of the test case. Must have ``"f"``
            and ``"mock_args_fn"`` keys.

    Returns:
        Tuple containing the function and the mock-args tuple it consumes
        (one entry per positional argument of ``f``).
    """
    manual_seed(0)
    return config["f"], config["mock_args_fn"]()


def make_jet_args(
    mock_args: tuple[Any, ...], K: int, *, collapsed: bool, R: int = 2
) -> tuple[Any, ...]:
    """Build random jet args matching ``mock_args``'s pytree structure.

    Each tensor leaf ``t`` in ``mock_args`` becomes a jet tuple
    ``(primal, c_1, ..., c_K)`` of tensors. Shape contract by mode:

    - **standard** (``collapsed=False``): every ``c_k`` has shape ``t.shape``.
      ``R`` is ignored.
    - **collapsed** (``collapsed=True``): ``primal`` has shape ``t.shape``,
      ``c_1..c_{K-1}`` have shape ``(R, *t.shape)`` (with ``c_2..c_{K-1}``
      zero for simplicity), and ``c_K`` is zero of shape ``t.shape``. Only
      the order-1 direction and the order-K collapsed slot are non-trivial
      -- enough to exercise both shape contracts.
    """
    manual_seed(42)

    def make_leaf(t: Tensor) -> tuple[Tensor, ...]:
        if not collapsed:
            return tuple(rand_like(t) for _ in range(K + 1))
        primal = rand_like(t)
        c_1 = rand(R, *t.shape, dtype=t.dtype)
        middle = [zeros(R, *t.shape, dtype=t.dtype) for _ in range(K - 2)]
        c_K = zeros_like(primal)
        return (primal, c_1, *middle, c_K)

    return tree_map(make_leaf, mock_args)
