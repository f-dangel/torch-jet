"""Taylor-mode (jet) transforms: ``jet``, ``_rev_jet``, ``_uncollapsed_via_vmap``.

These live in a leaf module (rather than ``jet/__init__.py``) so that
``jet/__init__.py`` can stay a pure re-export hub. No module imports the
package ``__init__``, which keeps the import graph acyclic.
"""

from math import factorial
from typing import Callable

from torch import Tensor, tensor, zeros_like
from torch.autograd import grad
from torch.func import vmap
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from jet.jet_interpreter import JetInterpreter
from jet.tracing import capture_graph
from jet.utils import Jet, PyTree, _is_jet_leaf
from jet.validation import validate_input_jet


def jet(
    f: Callable[[*tuple[PyTree[Tensor], ...]], PyTree[Tensor]],
    mock_args: tuple[PyTree[Tensor], ...],
    collapsed: bool = False,
) -> Callable[[*tuple[PyTree[Jet], ...]], PyTree[Jet]]:
    """Overload a function with its Taylor-mode equivalent.

    The returned function is K-polymorphic: the derivative order is inferred
    per call from the number of coefficients in the input jet tuples. The
    ``collapsed`` flag selects between two propagation regimes (the
    coefficient shape contract for each mode is given below); to freeze the
    order into an FX ``GraphModule`` (e.g. for graph passes like CSE), apply
    :func:`capture_graph` to the returned callable yourself.

    - **standard mode** (``collapsed=False``): every coefficient ``c_k`` has
      the primal's shape ``S``.
    - **collapsed mode** (``collapsed=True``): coefficients ``c_1..c_{K-1}``
      have shape ``(R, *S)`` carrying ``R`` directions; ``c_K`` has shape ``S``
      (already summed over the directions). The K-th output coefficient is
      likewise returned collapsed. This exploits that the highest-order
      coefficient enters linearly, so it can be summed eagerly to propagate
      smaller tensors through the graph. Requires ``K >= 2`` and ``R`` may
      vary per call.

    Args:
        f: Function to overload. May accept and return pytrees of tensors.
        mock_args: Mock input tensors (or pytrees of tensors) for tracing
            ``f``'s compute graph, provided as a tuple matching the positional
            arguments of ``f``. Only shapes and dtypes matter, not the values.
        collapsed: Select between the two propagation regimes above. Default:
            ``False`` (standard mode).

    Returns:
        A callable ``jet_f(*args)`` taking one positional argument per
        argument of ``f``. Each argument is a pytree mirroring the
        corresponding ``mock_args`` entry but with every tensor leaf
        replaced by a tuple ``(primal, c_1, ..., c_K)`` bundling the primal
        with its ``K`` Taylor coefficients. Returns a pytree mirroring ``f``'s
        output structure with each tensor leaf replaced by
        ``(f_0, f_1, ..., f_K)``.

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
    mod, _ = capture_graph(f, mock_args)
    interp = JetInterpreter(mod, collapsed=collapsed)

    def transformed(*args: PyTree[Jet]) -> PyTree[Jet]:
        leaves, K, R = validate_input_jet(mock_args, args, collapsed=collapsed)
        return interp.run(K, R, *leaves)

    return transformed


def _rev_jet(
    f: Callable[[*tuple[PyTree[Tensor], ...]], PyTree[Tensor]], detach: bool = True
) -> Callable[[*tuple[PyTree[Jet], ...]], PyTree[Jet]]:
    """Implement Taylor-mode via nested reverse-mode autodiff.

    Serves as a reference implementation for testing ``jet``. See :func:`jet`
    for the pytree-of-jets convention; the derivative order is inferred per
    call from the input jets.

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

    def jet_f(*args: PyTree[Jet]) -> PyTree[Jet]:
        """Compute the function and its Taylor coefficients."""
        leaves, in_spec = tree_flatten(args, is_leaf=_is_jet_leaf)
        derivative_order = len(leaves[0]) - 1

        primals = [leaf[0] for leaf in leaves]
        taylor_coeffs_by_order = [
            [leaf[order + 1] for leaf in leaves] for order in range(derivative_order)
        ]
        ref_tensor = primals[0]

        def path(t: Tensor) -> tuple[PyTree[Tensor], ...]:
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


def _uncollapsed_via_vmap(
    f: Callable[[*tuple[PyTree[Tensor], ...]], PyTree[Tensor]],
    mock_args: tuple[PyTree[Tensor], ...],
    randomization: tuple[str, int] | None,
) -> Callable[[*tuple[PyTree[Jet], ...]], PyTree[Jet]]:
    """Build a collapsed-jet-compatible function from standard ``jet`` + ``vmap``.

    The returned function has the same calling convention as
    :func:`jet` with ``collapsed=True``: each positional argument is a pytree
    whose tensor leaves are tuples ``(primal, c_1, ..., c_K)`` where
    coefficients of orders 1..K-1 are batched (leading direction dim ``R``)
    and the order-K coefficient is collapsed (no ``R`` dim). The output's K-th
    coefficient is likewise returned collapsed. ``K`` is inferred per call
    from the inputs.

    Used as a reference implementation against which the in-interpreter
    collapsed path (``jet(..., collapsed=True)``) is compared in tests.
    """
    jet_f = jet(f, mock_args)
    # ``in_spec`` is structural (no leaf values), so compute it once from
    # ``mock_args`` and reuse it for every cjet_f call to rebuild per-direction
    # args inside vmap. The validator confirms ``args`` matches this structure.
    _, in_spec = tree_flatten(mock_args)

    def cjet_f(*args: PyTree[Jet]) -> PyTree[Jet]:
        leaves, K, _ = validate_input_jet(mock_args, args, collapsed=True)
        num_leaves = len(leaves)
        primals = [leaf[0] for leaf in leaves]
        collapsed = [leaf[K] for leaf in leaves]
        # Batched coefficients (orders 1..K-1) carry the leading direction dim R.
        batched_flat = [leaf[order] for leaf in leaves for order in range(1, K)]

        def single_direction(*flat_batched: Tensor) -> PyTree[Jet]:
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
        def _collapse_leaf(leaf: Jet) -> Jet:
            return (leaf[0][0], *leaf[1:K], leaf[K].sum(0))

        return tree_map(_collapse_leaf, result, is_leaf=_is_jet_leaf)

    return cjet_f
