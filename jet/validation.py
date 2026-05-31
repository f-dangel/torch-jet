"""Validate user-supplied jet inputs against a mock-arg template.

A jet transform built via :func:`jet.jet` or :func:`jet.collapsed_jet` is
constructed against ``mock_args`` -- a pytree of tensors that describes the
expected leaf positions and shapes. At call time, the user passes ``args``
that mirror ``mock_args``'s structure but with every tensor leaf replaced by a
jet tuple ``(primal, c_1, ..., c_K)``. This module checks the user's ``args``
against ``mock_args``'s structure and the per-mode shape rules:

- **standard** (``collapsed=False``): every ``c_k`` has shape ``S = mock.shape``.
- **collapsed** (``collapsed=True``): coefficients ``c_1..c_{K-1}`` have shape
  ``(R, *S)`` with shared ``R`` across all leaves; ``c_K`` has shape ``S`` (the
  collapsed slot). Requires ``K >= 2``.

Structural traversal is delegated to PyTorch's pytree machinery (``tree_flatten``
plus ``TreeSpec.flatten_up_to``) which raises ``"Node type/arity mismatch"``
when ``args`` diverges from ``mock_args``'s structure.
"""

from typing import Any

from torch import Tensor
from torch.utils._pytree import tree_flatten


def validate_input_jet(
    mock: Any, args: Any, *, collapsed: bool
) -> tuple[list[tuple[Tensor, ...]], int, int | None]:
    """Validate ``args`` against ``mock``'s structure and shapes.

    Args:
        mock: Pytree of tensors describing the expected leaf positions and shapes.
        args: Pytree mirroring ``mock`` with each tensor leaf replaced by a jet
            tuple ``(primal, c_1, ..., c_K)`` of tensors.
        collapsed: Whether to apply the collapsed-mode shape rules.

    Returns:
        ``(jet_leaves, K, R)`` where ``jet_leaves`` is a flat list of jet
        tuples in mock-traversal order, ``K`` is the inferred derivative order
        (consistent across all leaves), and ``R`` is the inferred direction
        dimension for collapsed mode (``None`` in standard mode).

    Raises:
        ValueError: If ``args``' pytree structure differs from ``mock``'s
            (raised by ``flatten_up_to``), if arity (``K``) is inconsistent
            across leaves, or if any coefficient has the wrong shape.
    """
    mock_leaves, in_spec = tree_flatten(mock)
    if not mock_leaves:
        # capture_graph happily traces a zero-tensor-input function (e.g.
        # `f({}) -> tensor(1.0)` produces a 0-placeholder graph emitting a
        # constant), but a jet over such a function is meaningless: K and R
        # would stay None and downstream code (e.g. ``range(K)`` inside
        # ``_zero_coeffs``) would crash with TypeError. Reject up front.
        raise ValueError("No jet leaves found; mock_args has no tensors.")
    # flatten_up_to raises Node type/arity mismatch if args' pytree structure
    # diverges from mock's; at each tensor leaf in mock it takes the entire
    # subtree at the corresponding position in args (i.e. the jet tuple).
    arg_leaves = in_spec.flatten_up_to(args)
    K_seen: int | None = None
    R_seen: int | None = None
    for mock_t, arg in zip(mock_leaves, arg_leaves):
        K_seen, R_seen = _validate_jet_leaf(mock_t, arg, collapsed, K_seen, R_seen)
    if K_seen is None:
        # Unreachable: mock_leaves is non-empty (checked above), so the
        # for-loop ran at least once and _validate_jet_leaf set K_seen.
        # Explicit raise (not assert) so the contract survives ``python -O``.
        raise RuntimeError("internal: K_seen is None after validation loop.")
    return arg_leaves, K_seen, R_seen


def _validate_jet_leaf(
    mock: Tensor,
    arg: Any,
    collapsed: bool,
    K_seen: int | None,
    R_seen: int | None,
) -> tuple[int, int | None]:
    """Check that ``arg`` is a valid jet tuple matching ``mock``'s shape.

    ``K_seen`` and ``R_seen`` carry the values observed at earlier leaves
    (``None`` on the first call).

    Args:
        mock: The expected primal shape (one tensor leaf of ``mock_args``).
        arg: Candidate jet tuple ``(primal, c_1, ..., c_K)`` of tensors.
        collapsed: Whether to apply the collapsed-mode shape rules.
        K_seen: Derivative order observed at an earlier leaf, or ``None`` if
            this is the first leaf.
        R_seen: Direction dim observed at an earlier leaf in collapsed mode,
            or ``None``.

    Returns:
        ``(K, R)`` for this leaf, with ``R = R_seen`` if it was already set
        (so the caller can thread the pair to the next leaf).

    Raises:
        ValueError: If ``arg`` is not a tuple of tensors, if its ``K`` or
            ``R`` disagrees with ``K_seen`` / ``R_seen``, if ``collapsed``
            mode is requested with ``K < 2``, or if any coefficient has the
            wrong shape.
    """
    if not isinstance(mock, Tensor):
        raise ValueError(f"mock_args leaf must be a Tensor, got {type(mock).__name__}.")
    if not isinstance(arg, tuple):
        raise ValueError(
            f"expected a jet tuple (primal, c_1, ..., c_K), got {type(arg).__name__}."
        )
    if len(arg) < 1:
        raise ValueError(
            f"jet tuple must have at least 1 entry (primal), got length {len(arg)}."
        )
    if not all(isinstance(e, Tensor) for e in arg):
        raise ValueError(
            f"every entry of a jet tuple must be a Tensor; got types "
            f"{[type(e).__name__ for e in arg]}."
        )

    K = len(arg) - 1
    # Check the absolute K >= 2 floor first so a user who mixes K=1 with K=2
    # in collapsed mode sees the root cause ('collapsed requires K >= 2') on
    # the K=1 leaf, not the derived 'K disagrees' message.
    if collapsed and K < 2:
        raise ValueError(f"collapsed mode requires K >= 2, got K={K}.")
    if K_seen is not None and K != K_seen:
        raise ValueError(
            f"derivative order K={K} disagrees with K={K_seen} from an "
            f"earlier leaf; all jet leaves must share K."
        )

    primal, *coeffs = arg
    if primal.shape != mock.shape:
        raise ValueError(
            f"primal shape {tuple(primal.shape)} does not match mock shape "
            f"{tuple(mock.shape)}."
        )
    R = _check_coeffs(coeffs, mock, collapsed)
    # Cross-leaf R consistency. R is None in standard mode (no batched coeffs)
    # and in collapsed mode it is always non-None given K >= 2 (enforced above)
    # because c_1 is then batched and _check_coeffs's first iteration sets R.
    if R is not None and R_seen is not None and R != R_seen:
        raise ValueError(
            f"leaf's R={R} disagrees with R={R_seen} from an earlier leaf; "
            f"all batched coefficients must share R."
        )
    return K, R_seen if R_seen is not None else R


def _check_coeffs(coeffs: list[Tensor], mock: Tensor, collapsed: bool) -> int | None:
    """Validate coefficient shapes against ``mock``'s shape.

    - Standard mode: every ``c_k`` has shape ``mock.shape``.
    - Collapsed mode: ``c_1..c_{K-1}`` have shape ``(R, *mock.shape)`` with
      shared ``R`` within the leaf; ``c_K`` has ``mock.shape`` (collapsed
      slot).

    Returns the leaf's ``R`` (collapsed mode with ``K >= 2``) for the caller
    to cross-check against other leaves, or ``None`` otherwise.

    Raises:
        ValueError: If any coefficient violates its expected shape -- in
            standard mode, any ``c_k`` whose shape differs from
            ``mock.shape``; in collapsed mode, a batched ``c_k`` (k < K)
            without the ``(R, *mock.shape)`` shape or whose leading dim
            disagrees with other batched coefficients in the same leaf, or
            the collapsed slot ``c_K`` whose shape differs from
            ``mock.shape``.
    """
    K = len(coeffs)
    R: int | None = None
    for k, c in enumerate(coeffs, start=1):
        # Batched: (R, *S). Otherwise: S (which covers all of standard mode
        # and the collapsed slot c_K).
        if collapsed and k < K:
            if c.ndim != mock.ndim + 1 or c.shape[1:] != mock.shape:
                raise ValueError(
                    f"coefficient c_{k} has shape {tuple(c.shape)}, "
                    f"expected (R, *{tuple(mock.shape)})."
                )
            if R is None:
                R = c.shape[0]
            elif c.shape[0] != R:
                raise ValueError(
                    f"coefficient c_{k} has leading dim {c.shape[0]}, "
                    f"expected {R} (must match earlier batched coefficients "
                    f"in this leaf)."
                )
        elif c.shape != mock.shape:
            raise ValueError(
                f"coefficient c_{k} has shape {tuple(c.shape)}, "
                f"expected {tuple(mock.shape)}."
            )
    return R
