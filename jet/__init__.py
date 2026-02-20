"""Taylor-mode automatic differentiation (jets) in PyTorch."""

from math import factorial
from typing import Callable

from torch import Tensor, tensor, zeros_like
from torch.autograd import grad

from jet.jet_interpreter import JetInterpreter
from jet.operations import JetTuple
from jet.tracing import capture_graph
from jet.utils import Primal, PrimalAndCoefficients, Value, ValueAndCoefficients


def jet(
    f: Callable[[Primal], Value],
    derivative_order: int,
    mock_x: Tensor,
    verbose: bool = False,
) -> Callable[[PrimalAndCoefficients], ValueAndCoefficients]:
    """Overload a function with its Taylor-mode equivalent.

    Args:
        f: Function to overload. Maps a tensor to another tensor.
        derivative_order: The order of the Taylor expansion.
        mock_x: A mock input tensor for tracing. Only the shape matters, not
            the actual values.
        verbose: Whether to print the traced graph. Default: `False`.

    Returns:
        The overloaded function that computes the function and its Taylor coefficients
            from the input tensor and its Taylor coefficients.

    Examples:
        >>> from torch import sin, cos, zeros, Tensor
        >>> from jet import jet
        >>> f = sin
        >>> jet2_f = jet(f, 2, zeros(1))
        >>> # Set up the Taylor coefficients
        >>> x0, x1, x2 = Tensor([0.123]), Tensor([-0.456]), Tensor([0.789])
        >>> # Compute the function value and its Taylor coefficients
        >>> f0, f1, f2 = jet2_f(x0, x1, x2)
        >>> # Manually verify the Taylor coefficients (Faa di Bruno)
        >>> df, d2f = cos(x0), -sin(x0) # derivatives of the sin function
        >>> assert f0.allclose(sin(x0))
        >>> assert f1.allclose(df * x1)
        >>> assert f2.allclose(df * x2 + d2f * x1 ** 2)
    """
    mod = capture_graph(f, mock_x)

    if verbose:
        print(f"Traced graph:\n{mod.graph}")

    interp = JetInterpreter(mod, derivative_order)

    def jet_f(x: Primal, *vs: Primal) -> ValueAndCoefficients:
        result = interp.run((x, *vs))
        if not isinstance(result, JetTuple):
            return (result,) + tuple(zeros_like(result) for _ in vs)
        return tuple(result)

    return jet_f


def rev_jet(
    f: Callable[[Primal], Value],
    derivative_order: int | None = None,
    detach: bool = True,
) -> Callable[[PrimalAndCoefficients], ValueAndCoefficients]:
    """Implement Taylor-mode via nested reverse-mode autodiff.

    Args:
        f: Function to overload. Maps a tensor to another tensor.
        derivative_order: Order of the Taylor expansion. Default: `None`.
        detach: Whether to detach the output of the function and its Taylor coefficients
            from the computation graph. Default: `True`.

    Returns:
        The overloaded function that computes the function and its Taylor coefficients
        from the input tensor and its Taylor coefficients.
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
            The gradient of f w.r.t. X if f requires grad, otherwise a tensor of zeros.
            Has the same shape as X.
        """
        return grad(f, X, **grad_kwargs)[0] if f.requires_grad else zeros_like(X)

    def jet_f(
        x: Primal, *vs: Primal, derivative_order: int | None = derivative_order
    ) -> ValueAndCoefficients:
        """Compute the function and its Taylor coefficients.

        Args:
            x: Input tensor.
            *vs: Taylor coefficients.
            derivative_order: Order of the Taylor expansion. If `None`, the order is the number of
                Taylor coefficients.

        Returns:
            Tuple containing the function value and its Taylor coefficients.
        """
        if derivative_order is None:
            derivative_order = len(vs)
        else:
            assert derivative_order == len(vs)

        def path(t: Tensor):
            x_t = x + sum(
                t**n / factorial(n) * v_n for n, v_n in enumerate(vs, start=1)
            )
            return f(x_t)

        t = tensor(0.0, requires_grad=True, dtype=x.dtype, device=x.device)
        f_x = path(t)

        vs_out = [zeros_like(f_x).flatten() for _ in vs]

        for i, dnf_dt in enumerate(f_x.flatten()):
            for n in range(derivative_order):
                dnf_dt = _maybe_grad(dnf_dt, t)
                vs_out[n][i] = dnf_dt.detach() if detach else dnf_dt

        f_x = f_x.detach() if detach else f_x
        vs_out = tuple((v.detach() if detach else v).reshape_as(f_x) for v in vs_out)

        return (f_x, *vs_out)

    return jet_f
