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
from jet.tracing import capture_graph
from jet.utils import Value

_JetTypes = (JetTuple, CollapsedJetTuple)


def _is_jet_or_tensor(x: Any) -> bool:
    """Return True for JetTuples/CollapsedJetTuples and plain tensors."""
    return isinstance(x, (*_JetTypes, Tensor))


def _transpose_jet_output(result: Any, derivative_order: int) -> tuple[Any, ...]:
    """Transpose a pytree-of-JetTuples into a tuple-of-pytrees.

    Works for both ``JetTuple`` and ``CollapsedJetTuple`` leaves.

    Args:
        result: The pytree returned by the interpreter, whose leaves are
            ``JetTuple`` or ``CollapsedJetTuple`` instances (or plain tensors
            for constant outputs).
        derivative_order: The derivative order of the Taylor expansion.

    Returns:
        A tuple ``(f0, f1, ..., fk)`` where each ``fi`` has the same pytree
        structure as *result* but with plain tensor leaves corresponding to
        the *i*-th Taylor coefficient.
    """
    flat_tree, out_spec = tree_flatten(result, is_leaf=_is_jet_or_tensor)
    return tuple(
        tree_unflatten(
            [
                (
                    node[order]
                    if isinstance(node, _JetTypes)
                    else (node if order == 0 else zeros_like(node))
                )
                for node in flat_tree
            ],
            out_spec,
        )
        for order in range(derivative_order + 1)
    )


def jet(
    f: Callable[..., Any],
    derivative_order: int,
    mock_primals: tuple[Any, ...],
) -> GraphModule:
    """Overload a function with its Taylor-mode equivalent.

    ``Any`` in the type signatures denotes a *pytree of tensors*, i.e. an
    arbitrarily nested structure of ``Tensor``, ``tuple``, ``list``, or
    ``dict`` whose leaves are tensors.

    Args:
        f: Function to overload. May accept and return pytrees of tensors.
        derivative_order: The order of the Taylor expansion.
        mock_primals: Mock input tensors (or pytrees of tensors) for tracing,
            provided as a tuple matching the positional arguments of ``f``.
            Only shapes matter, not the actual values.

    Returns:
        A ``GraphModule`` ``jet_f(primals, taylor_coeffs)`` where
        ``taylor_coeffs``
        is a tuple with one entry per argument, each containing
        ``derivative_order`` Taylor coefficients (following
        `JAX's convention <https://docs.jax.dev/en/latest/jax.experimental.jet.html>`_).
        Returns ``(primals_out, taylor_coeffs_out)`` where ``primals_out`` has
        the same pytree structure as ``f``'s output and ``taylor_coeffs_out``
        is a tuple of ``derivative_order`` pytrees with the same structure.

    Examples:
        **Single-input**::

            >>> from torch import sin, zeros, Tensor
            >>> from jet import jet
            >>> jet2_f = jet(sin, 2, (zeros(1),))
            >>> x0, x1, x2 = Tensor([0.123]), Tensor([-0.456]), Tensor([0.789])
            >>> f0, (f1, f2) = jet2_f((x0,), ((x1, x2),))

        **Multi-input**::

            >>> from torch import cos
            >>> f = lambda x, y: sin(x) * cos(y)
            >>> jet1_f = jet(f, 1, (zeros(3), zeros(3)))
            >>> x, y = Tensor([0.1, 0.2, 0.3]), Tensor([0.4, 0.5, 0.6])
            >>> vx, vy = Tensor([1.0, 0.0, 0.0]), Tensor([0.0, 1.0, 0.0])
            >>> f0, (f1,) = jet1_f((x, y), ((vx,), (vy,)))
    """
    flat_mock_primals, in_spec = tree_flatten(mock_primals)
    num_leaves = len(flat_mock_primals)

    def flat_f(*flat_tensors: Tensor) -> Any:
        args = tree_unflatten(list(flat_tensors), in_spec)
        return f(*args)

    mod = capture_graph(flat_f, *flat_mock_primals)

    interp = JetInterpreter(mod, derivative_order)

    def jet_f(
        primals: tuple[Any, ...], taylor_coeffs: tuple[tuple[Any, ...], ...]
    ) -> tuple[Any, tuple[Any, ...]]:
        flat_primals = tree_flatten(primals)[0]
        flat_taylor_coeffs_by_order = [
            [
                coefficient
                for arg_taylor_coeffs in taylor_coeffs
                for coefficient in tree_flatten(arg_taylor_coeffs[order])[0]
            ]
            for order in range(derivative_order)
        ]
        input_tuples = [
            (
                flat_primals[i],
                *(
                    coeffs_at_order[i]
                    for coeffs_at_order in flat_taylor_coeffs_by_order
                ),
            )
            for i in range(num_leaves)
        ]
        output = interp.run(*input_tuples)
        output = _transpose_jet_output(output, derivative_order)
        return output[0], output[1:]

    mock_taylor_coeffs = tuple(
        tuple(tree_map(zeros_like, arg) for _ in range(derivative_order))
        for arg in mock_primals
    )
    return make_fx(jet_f)(mock_primals, mock_taylor_coeffs)


def rev_jet(
    f: Callable[..., Any],
    derivative_order: int | None = None,
    detach: bool = True,
) -> Callable[
    [tuple[Any, ...], tuple[tuple[Any, ...], ...]], tuple[Any, tuple[Any, ...]]
]:
    """Implement Taylor-mode via nested reverse-mode autodiff.

    Serves as a reference implementation for testing ``jet``. See :func:`jet`
    for a description of the ``Any`` pytree convention used in the type
    signatures.

    Args:
        f: Function to overload. May accept and return pytrees of tensors.
        derivative_order: Order of the Taylor expansion. Default: ``None``.
        detach: Whether to detach the output from the computation graph.
            Default: ``True``.

    Returns:
        A function ``jet_f(primals, taylor_coeffs)`` that returns
        ``(primals_out, taylor_coeffs_out)``.
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
        primals: tuple[Any, ...],
        taylor_coeffs: tuple[tuple[Any, ...], ...],
        *,
        derivative_order: int | None = derivative_order,
    ) -> tuple[Any, tuple[Any, ...]]:
        """Compute the function and its Taylor coefficients.

        Args:
            primals: Tuple of primal values matching ``f``'s positional args.
            taylor_coeffs: Tuple with one entry per argument, each containing
                ``derivative_order`` Taylor coefficients.
            derivative_order: Order of the Taylor expansion.

        Returns:
            ``(primals_out, taylor_coeffs_out)`` where *primals_out* has the
            pytree structure of ``f``'s output and *taylor_coeffs_out* is a
            tuple of ``derivative_order`` pytrees with the same structure.
        """
        if derivative_order is None:
            derivative_order = len(taylor_coeffs[0])
        else:
            assert all(
                len(arg_taylor_coeffs) == derivative_order
                for arg_taylor_coeffs in taylor_coeffs
            )

        primals, in_spec = tree_flatten(primals)
        taylor_coeffs_by_order = [
            [
                coefficient
                for arg_taylor_coeffs in taylor_coeffs
                for coefficient in tree_flatten(arg_taylor_coeffs[order])[0]
            ]
            for order in range(derivative_order)
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

        primals_out = tree_unflatten(
            [f_path.detach() if detach else f_path for f_path in f_paths], out_spec
        )

        taylor_coeffs_out = tuple(
            tree_unflatten(
                [
                    (coeffs.detach() if detach else coeffs).reshape_as(f_path)
                    for coeffs, f_path in zip(flat_taylor_coeffs_out[order], f_paths)
                ],
                out_spec,
            )
            for order in range(derivative_order)
        )
        return primals_out, taylor_coeffs_out

    return jet_f


def collapsed_jet(
    f: Callable[..., Value],
    derivative_order: int,
    mock_args: tuple,
    verbose: bool = False,
) -> Callable[..., tuple[Value, ...]]:
    """Overload f with collapsed Taylor-mode equivalent.

    Same API as ``jet()``, but expects mixed-shape series (orders 1..K):
      - series[0..K-2] (orders 1..K-1): tensors with leading batch dim R
      - series[K-1] (order K): tensors without batch dim (collapsed)

    The K-th output coefficient is automatically collapsed (summed over
    directions), so no ``.sum(0)`` or PullSum graph rewrites are needed.

    Raises:
        ValueError: If ``derivative_order < 2`` (collapsing requires at least
            one batched coefficient to carry direction information).
    """
    if derivative_order < 2:
        raise ValueError(
            f"collapsed_jet requires derivative_order >= 2, got {derivative_order}."
        )
    flat_mocks, in_spec = tree_flatten(mock_args)
    num_leaves = len(flat_mocks)

    def flat_f(*flat_tensors):
        args = tree_unflatten(list(flat_tensors), in_spec)
        return f(*args)

    mod = capture_graph(flat_f, *flat_mocks)
    if verbose:
        print(f"Traced graph:\n{mod.graph}")

    interp = CollapsedJetInterpreter(mod, derivative_order)

    def cjet_f(primals, series):
        flat_primals = tree_flatten(primals)[0]
        flat_series = [tree_flatten(s)[0] for s in series]
        input_tuples = [
            (flat_primals[i], *(fs[i] for fs in flat_series)) for i in range(num_leaves)
        ]
        result = interp.run(*input_tuples)
        all_orders = _transpose_jet_output(result, derivative_order)
        return all_orders[0], all_orders[1:]

    return cjet_f


def _make_uncollapsed_cjet(f, derivative_order, mock_args, randomization):
    """Build a collapsed_jet-compatible function using standard jet + vmap + sum.

    The returned function has the same calling convention as ``collapsed_jet``:
    it accepts ``(primals, series)`` where ``series[0..K-2]`` (orders 1..K-1)
    are batched and ``series[K-1]`` (order K) is collapsed, and returns output
    with the K-th coefficient already summed over directions.

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
    num_args = len(mock_args)

    def cjet_f(primals, series):
        # Flatten batched series entries (orders 1..K-1) for vmap
        batched_flat = []
        batched_specs = []
        for order in range(K - 1):
            flat, spec = tree_flatten(series[order])
            batched_flat.extend(flat)
            batched_specs.append((len(flat), spec))

        collapsed = series[K - 1]

        def single_direction(*flat_batched):
            # Reconstruct per-order pytrees from flat batched leaves
            idx = 0
            all_orders = []
            for n_leaves, spec in batched_specs:
                all_orders.append(
                    tree_unflatten(list(flat_batched[idx : idx + n_leaves]), spec)
                )
                idx += n_leaves
            all_orders.append(collapsed)

            # Transpose: series[order][arg] -> taylor_coeffs[arg][order]
            taylor_coeffs = tuple(
                tuple(all_orders[order][arg_idx] for order in range(K))
                for arg_idx in range(num_args)
            )
            return jet_f(primals, taylor_coeffs)

        vmapped = vmap(
            single_direction,
            randomness="error" if randomization is None else "different",
            out_dims=(None, tuple(0 for _ in range(K))),
        )
        F0, Fs = vmapped(*batched_flat)
        FK_summed = tree_map(lambda t: t.sum(0), Fs[-1])
        return F0, (*Fs[:-1], FK_summed)

    return cjet_f
