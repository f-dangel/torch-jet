"""Utility functions for testing."""

from typing import Any, Callable

from torch import Tensor, float64, manual_seed, rand, rand_like, zeros, zeros_like
from torch.utils._pytree import tree_map

#: Sweep K at the collapsed-mode floor and a representative high order.
#: Intermediate K values exercise the same code paths and don't earn their
#: own coverage at the primitive / composition layer.
K_VALUES = [2, 5]
K_IDS = [f"K={k}" for k in K_VALUES]


def shape(*dims: int) -> Callable[[], tuple[Tensor]]:
    """Mock-args factory: one ``rand`` tensor of the given shape (float64)."""
    return lambda: (rand(*dims, dtype=float64),)


def shapes(*shape_pairs) -> Callable[[], tuple[Tensor, ...]]:
    """Mock-args factory: one ``rand`` tensor per shape (all float64)."""
    return lambda: tuple(rand(*s, dtype=float64) for s in shape_pairs)


def setup_case(
    config: dict[str, Any], vmapsize: int = 0
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Instantiate the function and its input.

    Args:
        config: Configuration dictionary of the test case. Must have ``"f"``
            and ``"mock_args_fn"`` keys.
        vmapsize: Whether to generate an input for a vmap-ed operation.
            ``0`` means no vmap is applied. Default: ``0``.

    Returns:
        Tuple ``(f, x)`` with ``x`` in ``float64`` to avoid numerical issues.
    """
    manual_seed(0)
    f = config["f"]
    # Extract shape from mock_args_fn (single-input cases only).
    mock_args = config["mock_args_fn"]()
    arg_shape = mock_args[0].shape
    vmap_shape = arg_shape if vmapsize == 0 else (vmapsize, *arg_shape)
    # ``.double()`` (not ``dtype=float64``) — the two paths consume different
    # RNG bits, and the downstream MC tests are seeded against this draw.
    x = rand(*vmap_shape).double()
    return f, x


def make_standard_jet_args(
    mock_args: tuple[Any, ...], K: int, *, seed: int = 42
) -> tuple[Any, ...]:
    """Build random standard-mode jet args matching ``mock_args``'s structure.

    Each tensor leaf ``t`` in ``mock_args`` becomes a jet tuple
    ``(rand(t.shape), c_1, ..., c_K)`` with all coefficients of shape
    ``t.shape``.
    """
    manual_seed(seed)

    def make_leaf(t: Tensor) -> tuple[Tensor, ...]:
        return tuple(rand_like(t) for _ in range(K + 1))

    return tree_map(make_leaf, mock_args)


def make_collapsed_jet_args(
    mock_args: tuple[Any, ...], K: int, R: int = 2, *, seed: int = 42
) -> tuple[Any, ...]:
    """Build collapsed-mode jet args matching ``mock_args``'s structure.

    Each tensor leaf ``t`` in ``mock_args`` becomes a jet tuple
    ``(primal, c_1, c_2, ..., c_{K-1}, c_K)`` where ``primal`` and ``c_1`` are
    random, ``c_1..c_{K-1}`` carry the leading direction dim ``R``, and
    ``c_K`` is zero (the collapsed slot). Coefficients of orders ``2..K-1``
    are zero for simplicity -- only the order-1 direction and the order-K
    collapsed slot need to be non-trivial to exercise both shape contracts.
    """
    manual_seed(seed)

    def make_leaf(t: Tensor) -> tuple[Tensor, ...]:
        primal = rand_like(t)
        c_1 = rand(R, *t.shape, dtype=t.dtype)
        middle = [zeros(R, *t.shape, dtype=t.dtype) for _ in range(K - 2)]
        c_K = zeros_like(primal)
        return (primal, c_1, *middle, c_K)

    return tree_map(make_leaf, mock_args)


def make_jet_args(
    mock_args: tuple[Any, ...], K: int, *, collapsed: bool, R: int = 2
) -> tuple[Any, ...]:
    """Dispatch to :func:`make_standard_jet_args` or :func:`make_collapsed_jet_args`."""
    if collapsed:
        return make_collapsed_jet_args(mock_args, K, R)
    return make_standard_jet_args(mock_args, K)
