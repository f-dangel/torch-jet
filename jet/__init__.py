"""Taylor-mode automatic differentiation (jets) in PyTorch."""

from math import factorial
from typing import Any, Callable

from torch import Tensor, tensor, zeros_like
from torch.autograd import grad
from torch.func import vmap
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from jet.collapsed_jet_interpreter import CollapsedJetInterpreter
from jet.collapsed_operations import CollapsedJetTuple
from jet.jet_interpreter import JetInterpreter
from jet.operations import JetTuple
from jet.tracing import _assert_traceable_signature, capture_flat_graph
from jet.utils import Value

_JetTypes = (JetTuple, CollapsedJetTuple)


def _is_jet_or_tensor(x: Any) -> bool:
    """Return True for JetTuples/CollapsedJetTuples and plain tensors."""
    return isinstance(x, (*_JetTypes, Tensor))


def _is_jet_leaf(x: Any, derivative_order: int | None = None) -> bool:
    """Return whether ``x`` is a jet-leaf: a tuple of ``(primal, c_1, ..., c_K)``.

    In the input pytrees, every tensor leaf is replaced by such a tuple, while
    pytree containers (tuple/list/dict) only ever hold jet-leaves -- never bare
    tensors. Hence "a tuple whose elements are all tensors" unambiguously
    distinguishes a jet-leaf from a container.

    Args:
        x: The candidate pytree node.
        derivative_order: The order ``K`` of the Taylor expansion. If given, the
            tuple must have exactly ``K + 1`` entries; if ``None``, any non-empty
            tuple of tensors qualifies (used when ``K`` is inferred from input).

    Returns:
        ``True`` if ``x`` is a jet-leaf, ``False`` otherwise.
    """
    if not (
        isinstance(x, tuple) and len(x) >= 1 and all(isinstance(e, Tensor) for e in x)
    ):
        return False
    return derivative_order is None or len(x) == derivative_order + 1


def _normalize_output(result: Any, derivative_order: int) -> Any:
    """Convert the interpreter's pytree-of-jets into a pytree of plain tuples.

    Each ``JetTuple``/``CollapsedJetTuple`` leaf becomes a plain
    ``(f_0, f_1, ..., f_K)`` tuple. Constant outputs (plain tensors that do not
    depend on the inputs) are expanded to ``(c, 0, ..., 0)`` so that every output
    leaf has the same ``(primal, *coeffs)`` shape.

    Args:
        result: The pytree returned by the interpreter, whose leaves are
            ``JetTuple``/``CollapsedJetTuple`` instances or plain tensors.
        derivative_order: The order ``K`` of the Taylor expansion.

    Returns:
        A pytree with the same structure as ``result`` whose leaves are plain
        ``(primal, c_1, ..., c_K)`` tuples.
    """
    flat, spec = tree_flatten(result, is_leaf=_is_jet_or_tensor)
    leaves = [
        tuple(node)
        if isinstance(node, _JetTypes)
        else (node, *(zeros_like(node) for _ in range(derivative_order)))
        for node in flat
    ]
    return tree_unflatten(leaves, spec)


def jet(
    f: Callable[..., Any],
    derivative_order: int,
    mock_primals: tuple[Any, ...],
) -> GraphModule:
    """Overload a function with its Taylor-mode equivalent.

    ``Any`` in the type signatures denotes a *pytree of tensors*, i.e. an
    arbitrarily nested structure of ``Tensor``, ``tuple``, ``list``, or ``dict``
    whose leaves are tensors.

    Args:
        f: Function to overload. May accept and return pytrees of tensors.
        derivative_order: The order of the Taylor expansion.
        mock_primals: Mock input tensors (or pytrees of tensors) for tracing,
            provided as a tuple matching the positional arguments of ``f``.
            Only shapes matter, not the actual values.

    Returns:
        A ``GraphModule`` ``jet_f(*args)`` taking one positional argument per
        argument of ``f``. Each argument is a pytree whose tensor leaves are
        replaced by a tuple ``(primal, c_1, ..., c_K)`` bundling the primal with
        its ``K = derivative_order`` Taylor coefficients. Returns a pytree
        mirroring ``f``'s output structure, with each tensor leaf replaced by a
        tuple ``(f_0, f_1, ..., f_K)``.

    Examples:
        **Single-input**::

            >>> from torch import sin, zeros, Tensor
            >>> from jet import jet
            >>> jet2_f = jet(sin, 2, (zeros(1),))
            >>> x0, x1, x2 = Tensor([0.123]), Tensor([-0.456]), Tensor([0.789])
            >>> f0, f1, f2 = jet2_f((x0, x1, x2))

        **Multi-input**::

            >>> from torch import cos
            >>> f = lambda x, y: sin(x) * cos(y)
            >>> jet1_f = jet(f, 1, (zeros(3), zeros(3)))
            >>> x, y = Tensor([0.1, 0.2, 0.3]), Tensor([0.4, 0.5, 0.6])
            >>> vx, vy = Tensor([1.0, 0.0, 0.0]), Tensor([0.0, 1.0, 0.0])
            >>> f0, f1 = jet1_f((x, vx), (y, vy))
    """
    _assert_traceable_signature(mock_primals)
    mod, _ = capture_flat_graph(f, mock_primals)

    interp = JetInterpreter(mod, derivative_order)

    def jet_f(*args: Any) -> Any:
        leaves, _ = tree_flatten(
            args, is_leaf=lambda x: _is_jet_leaf(x, derivative_order)
        )
        result = interp.run(*leaves)
        return _normalize_output(result, derivative_order)

    mock_jets = tree_map(
        lambda t: (t, *(zeros_like(t) for _ in range(derivative_order))),
        mock_primals,
    )
    return make_fx(jet_f)(*mock_jets)


def rev_jet(
    f: Callable[..., Any],
    derivative_order: int | None = None,
    detach: bool = True,
) -> Callable[..., Any]:
    """Implement Taylor-mode via nested reverse-mode autodiff.

    Serves as a reference implementation for testing ``jet``. See :func:`jet`
    for a description of the pytree-of-jets convention used by the returned
    function.

    Args:
        f: Function to overload. May accept and return pytrees of tensors.
        derivative_order: Order of the Taylor expansion. If ``None`` (default),
            it is inferred from the input jets.
        detach: Whether to detach the output from the computation graph.
            Default: ``True``.

    Returns:
        A function ``jet_f(*args)`` with the same convention as :func:`jet`: one
        positional argument per argument of ``f``, each a pytree whose tensor
        leaves are tuples ``(primal, c_1, ..., c_K)``; returns a pytree mirroring
        ``f``'s output with each tensor leaf replaced by ``(f_0, f_1, ..., f_K)``.
    """
    grad_kwargs = {
        "allow_unused": True,
        "materialize_grads": True,
        "create_graph": True,
    }

    def _grad(f: Tensor, X: Tensor) -> Tensor:
        """Compute the gradient if f requires grad, otherwise return zeros.

        Args:
            f: The function output for which to compute the gradient.
            X: The input tensor at which to compute the gradient.

        Returns:
            The gradient of f w.r.t. X if f requires grad, otherwise a tensor
            of zeros.  Has the same shape as X.
        """
        return grad(f, X, **grad_kwargs)[0] if f.requires_grad else zeros_like(X)

    def jet_f(
        *args: Any,
        derivative_order: int | None = derivative_order,
    ) -> Any:
        """Compute the function and its Taylor coefficients.

        Args:
            args: One positional argument per argument of ``f``, each a pytree
                whose tensor leaves are ``(primal, c_1, ..., c_K)`` tuples.
            derivative_order: Order of the Taylor expansion. If ``None``, it is
                inferred from the input jets.

        Returns:
            A pytree mirroring ``f``'s output structure, with each tensor leaf
            replaced by a tuple ``(f_0, f_1, ..., f_K)``.
        """
        leaves, in_spec = tree_flatten(
            args, is_leaf=lambda x: _is_jet_leaf(x, derivative_order)
        )
        if derivative_order is None:
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


def collapsed_jet(
    f: Callable[..., Value], derivative_order: int, mock_args: tuple[Any, ...]
) -> Callable[..., tuple[Value, ...]]:
    """Overload ``f`` with its collapsed Taylor-mode equivalent.

    Like :func:`jet`, the returned function takes one positional argument per
    argument of ``f``; each is a pytree whose tensor leaves are replaced by a
    tuple ``(primal, c_1, ..., c_K)``. Unlike :func:`jet`, the coefficients have
    mixed shapes across orders:

    - orders 1..K-1: tensors with a leading direction dimension ``R``,
    - order K: tensors without the ``R`` dimension (already collapsed, i.e.
      summed over the directions).

    The K-th output coefficient is likewise returned collapsed. This exploits
    that the highest-order coefficient enters linearly, so it can be summed
    eagerly to propagate smaller tensors through the graph.

    Args:
        f: Function to overload. May accept and return pytrees of tensors.
        derivative_order: The order ``K`` of the Taylor expansion. Must be
            ``>= 2``.
        mock_args: Mock input tensors (or pytrees of tensors) for tracing,
            provided as a tuple matching the positional arguments of ``f``.
            Only shapes matter, not the actual values.

    Returns:
        A function ``cjet_f(*args)`` returning a pytree mirroring ``f``'s output
        structure, with each tensor leaf replaced by a tuple
        ``(f_0, f_1, ..., f_K)``. Orders 1..K-1 carry the leading ``R``
        dimension; order K is collapsed.

    Raises:
        ValueError: If ``derivative_order < 2`` (collapsing requires at least
            one batched coefficient to carry direction information).
    """
    if derivative_order < 2:
        raise ValueError(
            f"collapsed_jet requires derivative_order >= 2, got {derivative_order}."
        )
    _assert_traceable_signature(mock_args)
    mod, _ = capture_flat_graph(f, mock_args)

    interp = CollapsedJetInterpreter(mod, derivative_order)

    def cjet_f(*args: Any) -> Any:
        leaves, _ = tree_flatten(
            args, is_leaf=lambda x: _is_jet_leaf(x, derivative_order)
        )
        result = interp.run(*leaves)
        return _normalize_output(result, derivative_order)

    return cjet_f


def _make_uncollapsed_cjet(
    f: Callable[..., Value],
    derivative_order: int,
    mock_args: tuple[Any, ...],
    randomization: tuple[str, int] | None,
) -> Callable[..., tuple[Value, ...]]:
    """Build a collapsed_jet-compatible function using standard jet + vmap + sum.

    The returned function has the same calling convention as ``collapsed_jet``:
    it accepts one positional argument per argument of ``f``, each a pytree whose
    tensor leaves are tuples ``(primal, c_1, ..., c_K)`` where coefficients of
    orders 1..K-1 are batched (leading direction dim R) and the order-K
    coefficient is collapsed (no R dim). It returns output with the K-th
    coefficient already summed over directions.

    Supports pytree inputs and outputs, matching the generality of ``jet()``
    and ``collapsed_jet()``.

    Args:
        f: The function to trace.
        derivative_order: The order of the Taylor expansion.
        mock_args: Mock arguments for tracing.
        randomization: Randomization tuple or ``None``.

    Returns:
        A callable with the same interface as ``collapsed_jet(f, ...)``.
    """
    K = derivative_order
    jet_f = jet(f, K, mock_args)

    def cjet_f(*args: Any) -> Any:
        leaves, in_spec = tree_flatten(args, is_leaf=lambda x: _is_jet_leaf(x, K))
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
        out_leaves, out_spec = tree_flatten(
            result, is_leaf=lambda x: _is_jet_leaf(x, K)
        )
        collapsed_leaves = [
            (leaf[0][0], *(leaf[order] for order in range(1, K)), leaf[K].sum(0))
            for leaf in out_leaves
        ]
        return tree_unflatten(collapsed_leaves, out_spec)

    return cjet_f
