"""Implementation of AD primitives in Taylor-mode arithmetic."""

from typing import Optional

from torch import Tensor, cos, sin, tanh, zeros_like
from torch.nn.functional import linear

from jet.utils import (
    PrimalAndCoefficients,
    ValueAndCoefficients,
    integer_partitions,
    multiplicity,
    tensor_prod,
)


def jet_sin(arg: PrimalAndCoefficients) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the sine function.

    Args:
        arg: Input tensor and its Taylor coefficients.

    Returns:
        Tuple containing the value of the sine function and its Taylor coefficients.
    """
    (x, vs) = arg

    # pre-compute derivatives
    sin_x = sin(x)
    dsin = {0: sin_x, 1: cos(x)}

    def dn(*vs) -> Tensor:
        """Contract the derivative tensor along the vectors."""
        n = len(vs)
        sign = 1 if n % 4 in [0, 1] else -1
        func = dsin[0] if n % 2 == 0 else dsin[1]
        return tensor_prod(sign * func, *vs)

    vs_out = [zeros_like(sin_x) for _ in vs]
    order = len(vs)

    for k in range(order):
        for sigma in integer_partitions(k + 1):
            vs_contract = [vs[i - 1] for i in sigma]
            nu = multiplicity(sigma)
            vs_out[k].add_(dn(*vs_contract), alpha=nu)

    return sin_x, tuple(vs_out)


def jet_tanh(arg: PrimalAndCoefficients) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the hyperbolic tangent function.

    Args:
        arg: Input tensor and its Taylor coefficients.

    Returns:
        Tuple containing the value of the hyperbolic tangent function and its
        Taylor coefficients.
    """
    (x, vs) = arg

    # pre-compute derivatives
    tanh_x = x.tanh()
    sech_x = 1 / x.cosh()
    order = len(vs)
    dtanh = {0: tanh_x}
    if order >= 1:
        dtanh[1] = sech_x**2
    if order >= 2:
        dtanh[2] = -2 * dtanh[0] * dtanh[1]
    if order >= 3:
        dtanh[3] = -2 * dtanh[1] ** 2 + 4 * dtanh[0] ** 2 * dtanh[1]
    if order > 3:
        raise NotImplementedError(
            f"Tanh only supports derivatives up to third order. Got {order}."
        )

    def dn(*vs) -> Tensor:
        """Contract the derivative tensor along the vectors."""
        return tensor_prod(dtanh[len(vs)], *vs)

    vs_out = [zeros_like(tanh_x) for _ in vs]

    for k in range(order):
        for sigma in integer_partitions(k + 1):
            vs_contract = [vs[i - 1] for i in sigma]
            nu = multiplicity(sigma)
            vs_out[k].add_(dn(*vs_contract), alpha=nu)

    return tanh_x, tuple(vs_out)


def jet_linear(
    arg: PrimalAndCoefficients, weight: Tensor, bias: Optional[Tensor] = None
) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the linear function.

    Args:
        arg: Input tensor and its Taylor coefficients.

    Returns:
        Tuple containing the value of the linear function and its Taylor coefficients.
    """
    (x, vs) = arg

    linear_x = linear(x, weight, bias=bias)
    vs_out = tuple(linear(v, weight) for v in vs)

    return linear_x, vs_out


MAPPING = {
    sin: jet_sin,
    tanh: jet_tanh,
    linear: jet_linear,
}
