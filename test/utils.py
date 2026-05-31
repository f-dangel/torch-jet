"""Utility functions for testing."""

from typing import Any, Callable

from torch import Tensor, manual_seed, rand, zeros, zeros_like
from torch.utils._pytree import tree_map


def setup_case(
    config: dict[str, Any], vmapsize: int = 0, derivative_order: int | None = None
) -> tuple[Callable[[Tensor], Tensor], Tensor, tuple[Tensor, ...]]:
    """Instantiate the function, its input, and Taylor coefficients.

    Args:
        config: Configuration dictionary of the test case. Must have ``"f"``
            and ``"mock_args_fn"`` keys.
        vmapsize: Whether to generate inputs and Taylor coefficients for a
            vmap-ed operation. ``0`` means no vmap is applied. Default: ``0``.
        derivative_order: The number of Taylor coefficients to generate. No
            coefficients are generated if ``None``. Default: ``None``.

    Returns:
        Tuple containing the function, the input tensor, and the Taylor
        coefficients. All are in double precision to avoid numerical issues.
    """
    manual_seed(0)
    f = config["f"]
    # Extract shape from mock_args_fn (single-input cases only).
    mock_args = config["mock_args_fn"]()
    shape = mock_args[0].shape
    vmap_shape = shape if vmapsize == 0 else (vmapsize, *shape)
    x = rand(*vmap_shape).double()
    vs = (
        ()
        if derivative_order is None
        else tuple(rand(*vmap_shape).double() for _ in range(derivative_order))
    )
    return f, x, vs


def zip_jet(primals: tuple, *coeffs_by_order: tuple) -> tuple:
    """Zip primals with per-order coefficients into per-argument jet pytrees.

    Args:
        primals: Tuple of pytrees of tensors, one per positional arg of ``f``.
        *coeffs_by_order: For each derivative order ``k = 1..K``, one tuple
            of pytrees mirroring ``primals``'s structure that holds the
            ``k``-th-order coefficients.

    Returns:
        Tuple of pytrees mirroring ``primals``, with each tensor leaf
        replaced by a jet tuple ``(primal, c_1, ..., c_K)``.
    """
    return tuple(
        tree_map(lambda *ts: tuple(ts), p, *cs)
        for p, *cs in zip(primals, *coeffs_by_order)
    )


def make_standard_jet_args(
    mock_args: tuple[Any, ...], K: int, *, seed: int = 42
) -> tuple[Any, ...]:
    """Build random standard-mode jet args matching ``mock_args``'s structure.

    Each tensor leaf ``t`` in ``mock_args`` becomes a jet tuple
    ``(rand(t.shape), c_1, ..., c_K)`` with all coefficients of shape
    ``t.shape``.
    """
    manual_seed(seed)
    primals = tree_map(_rand_like, mock_args)
    coeffs_by_order = [tree_map(_rand_like, mock_args) for _ in range(K)]
    return zip_jet(primals, *coeffs_by_order)


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
        primal = rand(*t.shape, dtype=t.dtype)
        c_1 = rand(R, *t.shape, dtype=t.dtype)
        middle = [zeros(R, *t.shape, dtype=t.dtype) for _ in range(K - 2)]
        c_K = zeros_like(primal)
        return (primal, c_1, *middle, c_K)

    return tuple(tree_map(make_leaf, arg) for arg in mock_args)


def _rand_like(t: Tensor) -> Tensor:
    return rand(*t.shape, dtype=t.dtype)
