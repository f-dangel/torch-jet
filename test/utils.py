"""Utility functions for testing."""

from typing import Any, Callable

from torch import Tensor, float64, manual_seed, rand, rand_like, zeros, zeros_like
from torch.utils._pytree import tree_map

#: Standard-mode K sweep. ``K=0`` (primal-only) and ``K=1`` (Jacobian-vector
#: product) are boundary cases that exercise the no-recursion branches of the
#: jet rules; ``K=2`` is the collapsed-mode floor; ``K=5`` is a representative
#: high order. Intermediate orders exercise the same code paths and don't
#: earn their own coverage at the primitive / composition layer.
K_VALUES = [0, 1, 2, 5]
K_IDS = [f"K={k}" for k in K_VALUES]

#: Collapsed-mode K sweep. Collapsed mode requires ``K >= 2``, so the
#: standard-mode boundary cases ``K=0`` and ``K=1`` are not applicable.
K_VALUES_COLLAPSED = [2, 5]
K_IDS_COLLAPSED = [f"K={k}" for k in K_VALUES_COLLAPSED]


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
