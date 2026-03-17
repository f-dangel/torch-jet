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
        A ``GraphModule`` ``jet_f(primals, input_jets)`` where ``input_jets``
        is a tuple with one entry per argument, each containing
        ``derivative_order`` Taylor coefficients (following
        `JAX's convention <https://docs.jax.dev/en/latest/jax.experimental.jet.html>`_).
        Returns ``(primals_out, output_jets)`` where ``primals_out`` has the
        same pytree structure as ``f``'s output and ``output_jets`` is a
        tuple of ``derivative_order`` pytrees with the same structure.

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
        primals: tuple[Any, ...], input_jets: tuple[tuple[Any, ...], ...]
    ) -> tuple[Any, tuple[Any, ...]]:
        flat_primals = tree_flatten(primals)[0]
        flat_input_jets = [
            [
                coefficient
                for in_jet in input_jets
                for coefficient in tree_flatten(in_jet[order])[0]
            ]
            for order in range(derivative_order)
        ]
        input_tuples = [
            (flat_primals[i], *(flat_in_jet[i] for flat_in_jet in flat_input_jets))
            for i in range(num_leaves)
        ]
        # contains output of f and jets of f
        output = interp.run(*input_tuples)
        output = _transpose_jet_output(output, derivative_order)
        return output[0], output[1:]

    mock_input_jets = tuple(
        tuple(tree_map(zeros_like, arg) for _ in range(derivative_order))
        for arg in mock_primals
    )
    return make_fx(jet_f)(mock_primals, mock_input_jets)


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
        A function ``jet_f(primals, input_jets)`` that returns
        ``(primals_out, output_jets)``.
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
        input_jets: tuple[tuple[Any, ...], ...],
        *,
        derivative_order: int | None = derivative_order,
    ) -> tuple[Any, tuple[Any, ...]]:
        """Compute the function and its Taylor coefficients.

        Args:
            primals: Tuple of primal values matching ``f``'s positional args.
            input_jets: Tuple with one entry per argument, each containing
                ``derivative_order`` Taylor coefficients.
            derivative_order: Order of the Taylor expansion.

        Returns:
            ``(primals_out, output_jets)`` where *primals_out* has the pytree
            structure of ``f``'s output and *output_jets* is a tuple of
            ``derivative_order`` pytrees with the same structure.
        """
        # prepare inputs
        if derivative_order is None:
            derivative_order = len(input_jets[0])
        else:
            assert all(len(in_jet) == derivative_order for in_jet in input_jets)

        primals, in_spec = tree_flatten(primals)
        input_jets = [
            [
                coefficient
                for in_jet in input_jets
                for coefficient in tree_flatten(in_jet[order])[0]
            ]
            for order in range(derivative_order)
        ]
        ref_tensor = primals[0]

        def path(t: Tensor) -> Any:
            """Construct the Taylor path
            x_0 + t * x_1 + t^2 / 2 * x_2 + ... + t^k / k! x_k.
            It tracks ``f``'s dependence on the primal values and input jets.
            """  # noqa: D205
            taylor_series = [
                primal
                + sum(
                    t**order / factorial(order) * input_jets[order - 1][i]
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
        # evaluate f at all paths
        f_paths = f(*path(t))

        # Handle output: may be a tensor or pytree
        f_paths, out_spec = tree_flatten(f_paths)
        num_output_paths = len(f_paths)

        output_jets = [
            [zeros_like(f_path).flatten() for f_path in f_paths]
            for _ in range(derivative_order)
        ]

        # compute the nested derivatives of all nodes in the f_paths tree.
        for path_idx in range(num_output_paths):
            f_path = f_paths[path_idx]
            for i, path_node in enumerate(f_path.flatten()):
                dnf_dt = path_node
                for order in range(derivative_order):
                    dnf_dt = _grad(dnf_dt, t)
                    output_jets[order][path_idx][i] = (
                        dnf_dt.detach() if detach else dnf_dt
                    )

        # Reconstruct per-order outputs
        primals_out = tree_unflatten(
            [f_path.detach() if detach else f_path for f_path in f_paths], out_spec
        )

        output_jets = tuple(
            tree_unflatten(
                [
                    (out_jets.detach() if detach else out_jets).reshape_as(f_path)
                    for out_jets, f_path in zip(output_jets[order], f_paths)
                ],
                out_spec,
            )
            for order in range(derivative_order)
        )
        return primals_out, output_jets

    return jet_f
