"""Utility functions for testing."""

from torch.utils._pytree import tree_map


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
