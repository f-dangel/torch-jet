"""Taylor-mode automatic differentiation (jets) in PyTorch."""

from math import factorial
from typing import Callable

from torch import Tensor, tensor, zeros_like
from torch.autograd import grad
from torch.utils._pytree import tree_flatten, tree_unflatten

from jet.jet_interpreter import JetInterpreter
from jet.operations import JetTuple
from jet.tracing import capture_graph
from jet.utils import Primal as Primal
from jet.utils import PrimalAndCoefficients as PrimalAndCoefficients
from jet.utils import Value, ValueAndCoefficients


def _is_jet_or_tensor(x):
    """Return True for JetTuples and plain tensors (pytree leaves for transposition)."""
    return isinstance(x, (JetTuple, Tensor))


def _transpose_jet_output(result, derivative_order):
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
    flat, out_spec = tree_flatten(result, is_leaf=_is_jet_or_tensor)
    k = derivative_order + 1
    outputs = []
    for order in range(k):
        flat_order = [
            jt[order]
            if isinstance(jt, JetTuple)
            else (jt if order == 0 else zeros_like(jt))
            for jt in flat
        ]
        outputs.append(tree_unflatten(flat_order, out_spec))
    return tuple(outputs)


def jet(
    f: Callable[..., Value],
    derivative_order: int,
    mock_args: tuple,
    verbose: bool = False,
) -> Callable[..., ValueAndCoefficients]:
    """Overload a function with its Taylor-mode equivalent.

    Args:
        f: Function to overload.
        derivative_order: The order of the Taylor expansion.
        mock_args: Mock input tensors (or pytrees of tensors) for tracing,
            provided as a tuple matching the positional arguments of ``f``.
            Only shapes matter, not the actual values.
        verbose: Whether to print the traced graph. Default: ``False``.

    Returns:
        A function ``jet_f(primals, series)`` that returns
        ``(primals_out, series_out)`` where ``primals_out`` has the same
        pytree structure as ``f``'s output and ``series_out`` is a tuple
        of ``derivative_order`` pytrees with the same structure.

    Examples:
        **Single-input**::

            >>> from torch import sin, zeros, Tensor
            >>> from jet import jet
            >>> jet2_f = jet(sin, 2, (zeros(1),))
            >>> x0, x1, x2 = Tensor([0.123]), Tensor([-0.456]), Tensor([0.789])
            >>> f0, (f1, f2) = jet2_f((x0,), ((x1,), (x2,)))

        **Multi-input**::

            >>> from torch import cos
            >>> f = lambda x, y: sin(x) * cos(y)
            >>> jet1_f = jet(f, 1, (zeros(3), zeros(3)))
            >>> x, y = Tensor([0.1, 0.2, 0.3]), Tensor([0.4, 0.5, 0.6])
            >>> vx, vy = Tensor([1.0, 0.0, 0.0]), Tensor([0.0, 1.0, 0.0])
            >>> f0, (f1,) = jet1_f((x, y), ((vx, vy),))
    """
    flat_mocks, in_spec = tree_flatten(mock_args)
    num_leaves = len(flat_mocks)

    def flat_f(*flat_tensors):
        args = tree_unflatten(list(flat_tensors), in_spec)
        return f(*args)

    mod = capture_graph(flat_f, *flat_mocks)

    if verbose:
        print(f"Traced graph:\n{mod.graph}")

    interp = JetInterpreter(mod, derivative_order)

    def jet_f(primals, series):
        flat_primals = tree_flatten(primals)[0]
        flat_series = [tree_flatten(s)[0] for s in series]
        input_tuples = [
            (flat_primals[i], *(fs[i] for fs in flat_series)) for i in range(num_leaves)
        ]
        result = interp.run(*input_tuples)
        all_orders = _transpose_jet_output(result, derivative_order)
        return all_orders[0], all_orders[1:]

    return jet_f


def rev_jet(
    f: Callable[..., Value],
    derivative_order: int | None = None,
    detach: bool = True,
) -> Callable[..., ValueAndCoefficients]:
    """Implement Taylor-mode via nested reverse-mode autodiff.

    Serves as a reference implementation for testing ``jet``.

    Args:
        f: Function to overload.
        derivative_order: Order of the Taylor expansion. Default: ``None``.
        detach: Whether to detach the output from the computation graph.
            Default: ``True``.

    Returns:
        A function ``jet_f(primals, series)`` that returns
        ``(primals_out, series_out)``.
    """
    grad_kwargs = {
        "allow_unused": True,
        "materialize_grads": True,
        "create_graph": True,
    }

    def _maybe_grad(f: Tensor, X: Tensor) -> Tensor:
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
        primals, series, *, derivative_order: int | None = derivative_order
    ) -> ValueAndCoefficients:
        """Compute the function and its Taylor coefficients.

        Args:
            primals: Pytree of primal values matching ``f``'s positional args.
            series: Tuple of ``derivative_order`` pytrees, each with the same
                structure as *primals*.
            derivative_order: Order of the Taylor expansion.

        Returns:
            ``(primals_out, series_out)`` where *primals_out* has the pytree
            structure of ``f``'s output and *series_out* is a tuple of
            ``derivative_order`` pytrees with the same structure.
        """
        if derivative_order is None:
            derivative_order = len(series)
        else:
            assert derivative_order == len(series)

        flat_primals, in_spec = tree_flatten(primals)
        flat_series = [tree_flatten(s)[0] for s in series]
        k = derivative_order
        ref_tensor = flat_primals[0]

        def path(t: Tensor):
            flat_x_t = [
                p
                + sum(
                    t**n / factorial(n) * flat_series[n - 1][i] for n in range(1, k + 1)
                )
                for i, p in enumerate(flat_primals)
            ]
            unflat_args = tree_unflatten(flat_x_t, in_spec)
            return f(*unflat_args)

        t = tensor(
            0.0,
            requires_grad=True,
            dtype=ref_tensor.dtype,
            device=ref_tensor.device,
        )
        f_x = path(t)

        # Handle output: may be a tensor or pytree
        flat_f_x, out_spec = tree_flatten(f_x)
        num_out_leaves = len(flat_f_x)

        vs_out = [[zeros_like(leaf).flatten() for leaf in flat_f_x] for _ in range(k)]

        for leaf_idx in range(num_out_leaves):
            leaf = flat_f_x[leaf_idx]
            for i, elem in enumerate(leaf.flatten()):
                dnf_dt = elem
                for n in range(k):
                    dnf_dt = _maybe_grad(dnf_dt, t)
                    vs_out[n][leaf_idx][i] = dnf_dt.detach() if detach else dnf_dt

        # Reconstruct per-order outputs
        flat_f_x_det = [v.detach() if detach else v for v in flat_f_x]
        primals_out = tree_unflatten(flat_f_x_det, out_spec)

        series_out = []
        for n in range(k):
            flat_n = [
                (v.detach() if detach else v).reshape_as(leaf)
                for v, leaf in zip(vs_out[n], flat_f_x)
            ]
            series_out.append(tree_unflatten(flat_n, out_spec))

        return primals_out, tuple(series_out)

    return jet_f
