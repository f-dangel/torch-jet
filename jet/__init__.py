"""Taylor-mode automatic differentiation (jets) in PyTorch."""

from math import factorial
from typing import Any, Callable

from torch import Tensor, tensor, zeros_like
from torch.autograd import grad
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from jet.jet_interpreter import JetInterpreter
from jet.operations import JetTuple
from jet.tracing import capture_graph


def _is_jet_or_tensor(x: Any) -> bool:
    """Return True for JetTuples and plain tensors (pytree leaves for transposition)."""
    return isinstance(x, (JetTuple, Tensor))


def _transpose_jet_output(result: Any, derivative_order: int) -> tuple[Any, ...]:
    """Transpose a pytree-of-JetTuples into a tuple-of-pytrees.

    When the traced function returns a pytree (tuple, dict, etc.), the
    interpreter produces that same pytree structure but with ``JetTuple``
    leaves.  This helper transposes the structure so that we get one pytree
    per Taylor order.

    Args:
        result: The pytree returned by the interpreter, whose leaves are
            ``JetTuple`` instances (or plain tensors for constant outputs).
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
                    if isinstance(node, JetTuple)
                    else (node if order == 0 else zeros_like(node))
                )
                for node in flat_tree
            ],
            out_spec,
        )
        for order in range(derivative_order + 1)
    )


def _flatten_input_jet(
    input_jet: tuple[tuple[Any, ...], ...],
) -> tuple[list[list[Any]], Any]:
    """Flatten an order-major input jet and validate matching leaf structure."""
    assert len(input_jet) > 0

    flat_input_jet = []
    flat_order0_args, in_spec = tree_flatten(input_jet[0])
    assert all(isinstance(coefficient, Tensor) for coefficient in flat_order0_args)

    reference_shapes = [coefficient.shape for coefficient in flat_order0_args]
    num_leaves = len(flat_order0_args)
    flat_input_jet.append(flat_order0_args)

    for order_args in input_jet[1:]:
        flat_order_args, order_spec = tree_flatten(order_args)
        assert order_spec == in_spec
        assert len(flat_order_args) == num_leaves
        assert all(isinstance(coefficient, Tensor) for coefficient in flat_order_args)
        assert all(
            coefficient.shape == reference_shape
            for coefficient, reference_shape in zip(flat_order_args, reference_shapes)
        )
        flat_input_jet.append(flat_order_args)

    return flat_input_jet, in_spec


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
        A ``GraphModule`` ``jet_f(input_jet)`` that consumes one order-major
        full jet ``(order0_args, ..., orderK_args)`` and returns the
        corresponding order-major ``output_jet``. The 0-th entry of
        ``input_jet`` contains the primals, and each later entry contains the
        Taylor coefficients for that order.

    Examples:
        **Single-input**::

            >>> from torch import sin, zeros, Tensor
            >>> from jet import jet
            >>> jet2_f = jet(sin, 2, (zeros(1),))
            >>> x0, x1, x2 = Tensor([0.123]), Tensor([-0.456]), Tensor([0.789])
            >>> f0, f1, f2 = jet2_f(((x0,), (x1,), (x2,)))

        **Multi-input**::

            >>> from torch import cos
            >>> f = lambda x, y: sin(x) * cos(y)
            >>> jet1_f = jet(f, 1, (zeros(3), zeros(3)))
            >>> x, y = Tensor([0.1, 0.2, 0.3]), Tensor([0.4, 0.5, 0.6])
            >>> vx, vy = Tensor([1.0, 0.0, 0.0]), Tensor([0.0, 1.0, 0.0])
            >>> f0, f1 = jet1_f(((x, y), (vx, vy)))
    """
    flat_mock_primals, in_spec = tree_flatten(mock_primals)
    num_leaves = len(flat_mock_primals)

    def flat_f(*flat_tensors: Tensor) -> Any:
        args = tree_unflatten(list(flat_tensors), in_spec)
        return f(*args)

    mod = capture_graph(flat_f, *flat_mock_primals)

    interp = JetInterpreter(mod, derivative_order)

    def jet_f(input_jet: tuple[tuple[Any, ...], ...]) -> tuple[Any, ...]:
        inferred_derivative_order = len(input_jet) - 1
        assert inferred_derivative_order == derivative_order

        flat_input_jet, _ = _flatten_input_jet(input_jet)
        input_tuples = [
            tuple(coeffs_at_order[i] for coeffs_at_order in flat_input_jet)
            for i in range(num_leaves)
        ]
        output = interp.run(*input_tuples)
        output_jet = _transpose_jet_output(output, inferred_derivative_order)
        return output_jet

    mock_input_jet = (mock_primals,) + tuple(
        tuple(tree_map(zeros_like, arg) for arg in mock_primals)
        for _ in range(derivative_order)
    )
    return make_fx(jet_f)(mock_input_jet)


def rev_jet(
    f: Callable[..., Any],
    derivative_order: int | None = None,
    detach: bool = True,
) -> Callable[[tuple[tuple[Any, ...], ...]], tuple[Any, ...]]:
    """Implement Taylor-mode via nested reverse-mode autodiff.

    Serves as a reference implementation for testing ``jet``. See :func:`jet`
    for a description of the ``Any`` pytree convention used in the type
    signatures.

    Args:
        f: Function to overload. May accept and return pytrees of tensors.
        derivative_order: Optional consistency check for the Taylor expansion
            order. If specified, it must match the number of entries in
            ``input_jet`` minus one. Default: ``None``.
        detach: Whether to detach the output from the computation graph.
            Default: ``True``.

    Returns:
        A function ``jet_f(input_jet)`` that returns the corresponding
        order-major ``output_jet``.
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

    def jet_f(input_jet: tuple[tuple[Any, ...], ...]) -> tuple[Any, ...]:
        """Compute the function and its Taylor coefficients.

        Args:
            input_jet: Order-major full jet ``(order0_args, ..., orderK_args)``
                whose entries match ``f``'s positional-argument pytree
                structure.

        Returns:
            The order-major ``output_jet`` with the same convention.
        """
        inferred_derivative_order = len(input_jet) - 1
        if derivative_order is not None:
            assert derivative_order == inferred_derivative_order

        flat_input_jet, in_spec = _flatten_input_jet(input_jet)
        num_leaves = len(flat_input_jet[0])
        ref_tensor = flat_input_jet[0][0]

        def path(t: Tensor) -> Any:
            """Construct the Taylor path
            x_0 + t * x_1 + t^2 / 2 * x_2 + ... + t^k / k! x_k.
            It tracks ``f``'s dependence on the primal values and Taylor coefficients.
            """  # noqa: D205
            taylor_series = [
                sum(
                    t**order / factorial(order) * flat_input_jet[order][i]
                    for order in range(inferred_derivative_order + 1)
                )
                for i in range(num_leaves)
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
            for _ in range(inferred_derivative_order)
        ]

        for path_idx in range(num_output_paths):
            f_path = f_paths[path_idx]
            for i, path_node in enumerate(f_path.flatten()):
                dnf_dt = path_node
                for order in range(inferred_derivative_order):
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
            for order in range(inferred_derivative_order)
        )
        output_jet = (primals_out, *taylor_coeffs_out)
        return output_jet

    return jet_f
