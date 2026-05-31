"""Taylor-mode automatic differentiation (jets) in PyTorch."""

from math import factorial
from typing import Any, Callable

from torch import Tensor, tensor, zeros_like
from torch.autograd import grad
from torch.func import vmap
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from jet.collapsed_operations import CollapsedJetTuple
from jet.jet_interpreter import JetInterpreter
from jet.operations import JetTuple
from jet.tracing import capture_graph
from jet.utils import Value

__all__ = [
    "capture_graph",
    "collapsed_jet",
    "jet",
    "rev_jet",
]

_JetTypes = (JetTuple, CollapsedJetTuple)


def _is_jet_or_tensor(x: Any) -> bool:
    """Return True for ``JetTuple``/``CollapsedJetTuple`` and plain tensors."""
    return isinstance(x, (*_JetTypes, Tensor))


def _walk_and_validate(
    mock: Any, args: Any, *, collapsed: bool, path: str = ""
) -> tuple[list[tuple[Tensor, ...]], int]:
    """Walk ``mock`` and ``args`` in parallel; validate and collect jet leaves.

    ``args`` mirrors the pytree structure of ``mock`` but every ``Tensor`` leaf
    in ``mock`` is replaced by a tuple ``(primal, c_1, ..., c_K)`` of Taylor
    coefficients in ``args``. The traversal uses ``mock`` as the template -- so
    a jet leaf is *positional*, not detected by a "tuple of K+1 tensors"
    heuristic. This eliminates the ambiguity where a pytree container of K+1
    tensors would otherwise be misread as a jet leaf.

    Shape rules per mode (``S = mock.shape``):

    - **standard** (``collapsed=False``): every coefficient has shape ``S``.
    - **collapsed**: coefficients ``c_1..c_{K-1}`` have shape ``(R, *S)`` with a
      common leading ``R`` across all leaves; coefficient ``c_K`` has shape ``S``
      (already summed over directions). Requires ``K >= 2``.

    Args:
        mock: Pytree of tensors describing the expected leaf positions and shapes.
        args: Pytree mirroring ``mock`` with each tensor leaf replaced by a jet
            tuple.
        collapsed: Whether to apply the collapsed-mode shape rules.
        path: Internal -- pytree path of the current node, for error messages.

    Returns:
        ``(jet_leaves, K)`` where ``jet_leaves`` is a flat list of jet tuples
        in mock-traversal order and ``K`` is the inferred derivative order
        (consistent across all leaves).

    Raises:
        ValueError: If structures disagree, arity is inconsistent, or any
            coefficient has the wrong shape.
    """
    jet_leaves: list[tuple[Tensor, ...]] = []
    state: dict[str, int | None] = {"K": None, "R": None}

    def _walk(m: Any, a: Any, p: str) -> None:
        if isinstance(m, Tensor):
            _validate_jet_leaf(m, a, p, collapsed=collapsed, state=state)
            jet_leaves.append(a)
            return
        if isinstance(m, dict):
            if not isinstance(a, dict) or set(m.keys()) != set(a.keys()):
                raise ValueError(
                    f"At {p or '<root>'}: expected dict with keys "
                    f"{sorted(m.keys())}, got {type(a).__name__} "
                    f"{sorted(a.keys()) if isinstance(a, dict) else ''}."
                )
            for k in m:
                _walk(m[k], a[k], f"{p}[{k!r}]")
            return
        if isinstance(m, (tuple, list)):
            if type(a) is not type(m) or len(a) != len(m):
                raise ValueError(
                    f"At {p or '<root>'}: expected {type(m).__name__} of "
                    f"length {len(m)}, got {type(a).__name__} of length "
                    f"{len(a) if hasattr(a, '__len__') else '?'}."
                )
            for i, (mc, ac) in enumerate(zip(m, a)):
                _walk(mc, ac, f"{p}[{i}]")
            return
        raise ValueError(
            f"At {p or '<root>'}: unsupported mock node type {type(m).__name__}."
        )

    _walk(mock, args, path)
    if state["K"] is None:
        raise ValueError("No jet leaves found; mock_primals has no tensors.")
    return jet_leaves, state["K"]


def _validate_jet_leaf(
    mock: Tensor,
    arg: Any,
    path: str,
    *,
    collapsed: bool,
    state: dict[str, int | None],
) -> None:
    """Check that ``arg`` is a valid jet tuple matching ``mock``'s shape.

    Updates ``state["K"]`` (derivative order, must be shared) and, for collapsed
    mode, ``state["R"]`` (leading direction dim, must be shared across all
    leaves and all batched coefficients).
    """
    if not isinstance(arg, tuple):
        raise ValueError(
            f"At {path or '<root>'}: expected a jet tuple "
            f"(primal, c_1, ..., c_K), got {type(arg).__name__}."
        )
    if len(arg) < 1:
        raise ValueError(
            f"At {path or '<root>'}: jet tuple must have at least 1 entry "
            f"(primal), got length {len(arg)}."
        )
    if not all(isinstance(e, Tensor) for e in arg):
        raise ValueError(
            f"At {path or '<root>'}: every entry of a jet tuple must be a "
            f"Tensor; got types "
            f"{[type(e).__name__ for e in arg]}."
        )

    K = len(arg) - 1
    if state["K"] is None:
        state["K"] = K
    elif K != state["K"]:
        raise ValueError(
            f"At {path or '<root>'}: derivative order K={K} disagrees with "
            f"K={state['K']} from an earlier leaf; all jet leaves must share K."
        )
    if collapsed and K < 2:
        raise ValueError(
            f"At {path or '<root>'}: collapsed mode requires K >= 2, got K={K}."
        )

    primal, *coeffs = arg
    if primal.shape != mock.shape:
        raise ValueError(
            f"At {path or '<root>'}: primal shape {tuple(primal.shape)} "
            f"does not match mock shape {tuple(mock.shape)}."
        )
    if collapsed:
        _check_collapsed_coeffs(coeffs, K, mock, path, state)
    else:
        _check_standard_coeffs(coeffs, mock, path)


def _check_standard_coeffs(coeffs: list[Tensor], mock: Tensor, path: str) -> None:
    """Standard mode: every coefficient has the primal's shape."""
    for k, c in enumerate(coeffs, start=1):
        if c.shape != mock.shape:
            raise ValueError(
                f"At {path or '<root>'}: coefficient c_{k} shape "
                f"{tuple(c.shape)} does not match primal shape "
                f"{tuple(mock.shape)}."
            )


def _check_collapsed_coeffs(
    coeffs: list[Tensor],
    K: int,
    mock: Tensor,
    path: str,
    state: dict[str, int | None],
) -> None:
    """Collapsed mode: ``c_1..c_{K-1}`` are ``(R, *S)``; ``c_K`` is ``S``."""
    for k, c in enumerate(coeffs, start=1):
        if k == K:
            if c.shape != mock.shape:
                raise ValueError(
                    f"At {path or '<root>'}: collapsed coefficient c_K=c_{k} "
                    f"(collapsed slot) has shape {tuple(c.shape)}, expected "
                    f"{tuple(mock.shape)}."
                )
            continue
        if c.ndim != mock.ndim + 1 or c.shape[1:] != mock.shape:
            raise ValueError(
                f"At {path or '<root>'}: collapsed coefficient c_{k} has "
                f"shape {tuple(c.shape)}, expected (R, *{tuple(mock.shape)})."
            )
        if state["R"] is None:
            state["R"] = c.shape[0]
        elif c.shape[0] != state["R"]:
            raise ValueError(
                f"At {path or '<root>'}: collapsed coefficient c_{k} has "
                f"leading dim {c.shape[0]}, expected {state['R']} (must be "
                f"shared across all batched coefficients)."
            )


def _normalize_output(result: Any, derivative_order: int) -> Any:
    """Convert the interpreter's pytree-of-jets into a pytree of plain tuples.

    Each ``JetTuple``/``CollapsedJetTuple`` leaf becomes a plain
    ``(f_0, f_1, ..., f_K)`` tuple. Constant outputs (plain tensors that do not
    depend on the inputs) are expanded to ``(c, 0, ..., 0)`` so that every
    output leaf has the same ``(primal, *coeffs)`` shape.
    """
    flat, spec = tree_flatten(result, is_leaf=_is_jet_or_tensor)
    leaves = [
        tuple(node)
        if isinstance(node, _JetTypes)
        else (node, *([zeros_like(node)] * derivative_order))
        for node in flat
    ]
    return tree_unflatten(leaves, spec)


def jet(
    f: Callable[..., Any],
    mock_primals: tuple[Any, ...],
) -> Callable[..., Any]:
    """Overload a function with its Taylor-mode equivalent.

    ``Any`` in the type signatures denotes a *pytree of tensors*, i.e. an
    arbitrarily nested structure of ``Tensor``, ``tuple``, ``list``, or ``dict``
    whose leaves are tensors.

    The returned function is K-polymorphic: the derivative order is inferred
    per call from the number of coefficients in the input jet tuples. To
    freeze the order into an FX ``GraphModule`` (e.g. for graph passes like
    CSE), apply :func:`capture_graph` to the returned callable yourself.

    Args:
        f: Function to overload. May accept and return pytrees of tensors.
        mock_primals: Mock input tensors (or pytrees of tensors) for tracing
            ``f``'s compute graph, provided as a tuple matching the positional
            arguments of ``f``. Only shapes and dtypes matter, not the values.

    Returns:
        A callable ``jet_f(*args)`` taking one positional argument per argument
        of ``f``. Each argument is a pytree mirroring the corresponding
        ``mock_primals`` entry but with every tensor leaf replaced by a tuple
        ``(primal, c_1, ..., c_K)`` bundling the primal with its ``K`` Taylor
        coefficients. Returns a pytree mirroring ``f``'s output structure with
        each tensor leaf replaced by ``(f_0, f_1, ..., f_K)``.

    Examples:
        **Single-input**::

            >>> from torch import sin, zeros, Tensor
            >>> from jet import jet
            >>> jet_f = jet(sin, (zeros(1),))
            >>> x0, x1, x2 = Tensor([0.123]), Tensor([-0.456]), Tensor([0.789])
            >>> f0, f1, f2 = jet_f((x0, x1, x2))

        **Multi-input**::

            >>> from torch import cos
            >>> f = lambda x, y: sin(x) * cos(y)
            >>> jet_f = jet(f, (zeros(3), zeros(3)))
            >>> x, y = Tensor([0.1, 0.2, 0.3]), Tensor([0.4, 0.5, 0.6])
            >>> vx, vy = Tensor([1.0, 0.0, 0.0]), Tensor([0.0, 1.0, 0.0])
            >>> f0, f1 = jet_f((x, vx), (y, vy))
    """
    mod = capture_graph(f, mock_primals)
    interp = JetInterpreter(mod)

    def jet_f(*args: Any) -> Any:
        leaves, K = _walk_and_validate(mock_primals, args, collapsed=False)
        result = interp.run(*leaves)
        return _normalize_output(result, K)

    return jet_f


def collapsed_jet(
    f: Callable[..., Value],
    mock_primals: tuple[Any, ...],
) -> Callable[..., tuple[Value, ...]]:
    """Overload ``f`` with its collapsed Taylor-mode equivalent.

    Like :func:`jet`, the returned callable takes one positional argument per
    argument of ``f``; each is a pytree mirroring the corresponding
    ``mock_primals`` entry with every tensor leaf replaced by a tuple
    ``(primal, c_1, ..., c_K)``. Unlike :func:`jet`, the coefficients have
    mixed shapes across orders:

    - orders 1..K-1: tensors with a leading direction dimension ``R``,
    - order K: tensors without the ``R`` dimension (already collapsed, i.e.
      summed over the directions).

    The K-th output coefficient is likewise returned collapsed. This exploits
    that the highest-order coefficient enters linearly, so it can be summed
    eagerly to propagate smaller tensors through the graph.

    The returned callable is both ``K``-polymorphic (``K >= 2`` inferred per
    call) and ``R``-polymorphic (any ``R`` per call). To freeze ``K`` and
    ``R`` into an FX ``GraphModule``, apply :func:`capture_graph` to it
    yourself.

    Args:
        f: Function to overload. May accept and return pytrees of tensors.
        mock_primals: Mock input tensors (or pytrees of tensors) for tracing
            ``f``'s compute graph, provided as a tuple matching the positional
            arguments of ``f``. Only shapes and dtypes matter, not the values.

    Returns:
        A callable ``cjet_f(*args)`` returning a pytree mirroring ``f``'s
        output structure, with each tensor leaf replaced by a tuple
        ``(f_0, f_1, ..., f_K)``. Orders 1..K-1 carry the leading ``R``
        dimension; order K is collapsed.

    Raises:
        ValueError: At call time, if ``K < 2`` is inferred from the inputs.
    """
    mod = capture_graph(f, mock_primals)
    interp = JetInterpreter(mod, collapsed=True)

    def cjet_f(*args: Any) -> Any:
        leaves, K = _walk_and_validate(mock_primals, args, collapsed=True)
        result = interp.run(*leaves)
        return _normalize_output(result, K)

    return cjet_f


def rev_jet(f: Callable[..., Any], detach: bool = True) -> Callable[..., Any]:
    """Implement Taylor-mode via nested reverse-mode autodiff.

    Serves as a reference implementation for testing ``jet``. See :func:`jet`
    for a description of the pytree-of-jets convention used by the returned
    function. The derivative order is inferred per call from the input jets.

    Args:
        f: Function to overload. May accept and return pytrees of tensors.
        detach: Whether to detach the output from the computation graph.
            Default: ``True``.

    Returns:
        A callable ``jet_f(*args)`` with the same convention as :func:`jet`.
    """
    grad_kwargs = {
        "allow_unused": True,
        "materialize_grads": True,
        "create_graph": True,
    }

    def _grad(f: Tensor, X: Tensor) -> Tensor:
        """Gradient of ``f`` w.r.t. ``X`` if ``f`` requires grad, else zeros."""
        return grad(f, X, **grad_kwargs)[0] if f.requires_grad else zeros_like(X)

    def _is_jet_leaf(x: Any) -> bool:
        return (
            isinstance(x, tuple)
            and len(x) >= 1
            and all(isinstance(e, Tensor) for e in x)
        )

    def jet_f(*args: Any) -> Any:
        """Compute the function and its Taylor coefficients."""
        leaves, in_spec = tree_flatten(args, is_leaf=_is_jet_leaf)
        derivative_order = len(leaves[0]) - 1

        primals = [leaf[0] for leaf in leaves]
        taylor_coeffs_by_order = [
            [leaf[order + 1] for leaf in leaves] for order in range(derivative_order)
        ]
        ref_tensor = primals[0]

        def path(t: Tensor) -> Any:
            """Construct the Taylor path
            x_0 + t * x_1 + t^2 / 2 * x_2 + ... + t^k / k! x_k.
            It tracks ``f``'s dependence on the primal values and Taylor coefficients.
            """  # noqa: D205
            taylor_series = [
                primal
                + sum(
                    t**order / factorial(order) * taylor_coeffs_by_order[order - 1][i]
                    for order in range(1, derivative_order + 1)
                )
                for i, primal in enumerate(primals)
            ]
            return tree_unflatten(taylor_series, in_spec)

        t = tensor(
            0.0,
            requires_grad=True,
            dtype=ref_tensor.dtype,
            device=ref_tensor.device,
        )
        f_paths = f(*path(t))

        f_paths, out_spec = tree_flatten(f_paths)
        num_output_paths = len(f_paths)

        flat_taylor_coeffs_out = [
            [zeros_like(f_path).flatten() for f_path in f_paths]
            for _ in range(derivative_order)
        ]

        for path_idx in range(num_output_paths):
            f_path = f_paths[path_idx]
            for i, path_node in enumerate(f_path.flatten()):
                dnf_dt = path_node
                for order in range(derivative_order):
                    dnf_dt = _grad(dnf_dt, t)
                    flat_taylor_coeffs_out[order][path_idx][i] = (
                        dnf_dt.detach() if detach else dnf_dt
                    )

        out_jets = [
            (
                f_path.detach() if detach else f_path,
                *(
                    flat_taylor_coeffs_out[order][path_idx].reshape_as(f_path)
                    for order in range(derivative_order)
                ),
            )
            for path_idx, f_path in enumerate(f_paths)
        ]
        return tree_unflatten(out_jets, out_spec)

    return jet_f


def _make_uncollapsed_cjet(
    f: Callable[..., Value],
    mock_args: tuple[Any, ...],
    randomization: tuple[str, int] | None,
) -> Callable[..., tuple[Value, ...]]:
    """Build a ``collapsed_jet``-compatible function using standard jet + vmap + sum.

    The returned function has the same calling convention as ``collapsed_jet``:
    it accepts one positional argument per argument of ``f``, each a pytree
    whose tensor leaves are tuples ``(primal, c_1, ..., c_K)`` where
    coefficients of orders 1..K-1 are batched (leading direction dim R) and the
    order-K coefficient is collapsed (no R dim). It returns output with the
    K-th coefficient already summed over directions.

    ``K`` is inferred per call from the input jets.
    """
    jet_f = jet(f, mock_args)

    def _is_jet_leaf(x: Any) -> bool:
        return (
            isinstance(x, tuple)
            and len(x) >= 2
            and all(isinstance(e, Tensor) for e in x)
        )

    def cjet_f(*args: Any) -> Any:
        leaves, in_spec = tree_flatten(args, is_leaf=_is_jet_leaf)
        K = len(leaves[0]) - 1
        num_leaves = len(leaves)
        primals = [leaf[0] for leaf in leaves]
        collapsed = [leaf[K] for leaf in leaves]
        # Batched coefficients (orders 1..K-1) carry the leading direction dim R.
        batched_flat = [leaf[order] for leaf in leaves for order in range(1, K)]

        def single_direction(*flat_batched: Tensor) -> Any:
            # Rebuild per-leaf jets from this direction's batched coefficients,
            # reusing the shared (un-batched) collapsed order-K coefficient.
            per_leaf_jets = [
                (
                    primals[i],
                    *flat_batched[i * (K - 1) : (i + 1) * (K - 1)],
                    collapsed[i],
                )
                for i in range(num_leaves)
            ]
            return jet_f(*tree_unflatten(per_leaf_jets, in_spec))

        vmapped = vmap(
            single_direction,
            randomness="error" if randomization is None else "different",
        )
        result = vmapped(*batched_flat)

        # De-batch order 0 (identical across directions) and collapse order K.
        def _collapse_leaf(leaf: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
            return (leaf[0][0], *leaf[1:K], leaf[K].sum(0))

        return tree_map(_collapse_leaf, result, is_leaf=_is_jet_leaf)

    return cjet_f
