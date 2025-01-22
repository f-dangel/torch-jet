"""Implementation of AD primitives in Taylor-mode arithmetic."""

from typing import Optional

from torch import Tensor, cat, cos, sigmoid, sin, stack, tanh, zeros, zeros_like
from torch.nn import Module
from torch.nn.functional import linear

from jet.utils import (
    PrimalAndCoefficients,
    ValueAndCoefficients,
    integer_partitions,
    multiplicity,
    tensor_prod,
)

# class JetSin(Module):

#     def forward(self, arg):
#         (x, vs) = arg

#         # pre-compute derivatives
#         sin_x = sin(x)
#         dsin = {0: sin_x, 1: cos(x)}

#         def dn(*vs) -> Tensor:
#             """Contract the derivative tensor along the vectors."""
#             n = len(vs)
#             sign = 1 if n % 4 in [0, 1] else -1
#             func = dsin[0] if n % 2 == 0 else dsin[1]
#             return tensor_prod(sign * func, *vs)

#         vs_out = [zeros_like(sin_x) for _ in vs]
#         order = len(vs)

#         for k in range(order):
#             for sigma in integer_partitions(k + 1):
#                 vs_contract = [vs[i - 1] for i in sigma]
#                 nu = multiplicity(sigma)
#                 vs_out[k].add_(dn(*vs_contract), alpha=nu)

#         return sin_x, tuple(vs_out)


def jet_sin(
    s: PrimalAndCoefficients, K=None, vmap: bool = False
) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the sine function.

    Args:
        s: The stacked input and Taylor coefficients.

    Returns:
        The stacked output and Taylor coefficients
    """
    dim = 1 if vmap else 0
    x, vs = s.split([1, K], dim=dim)
    x = x.squeeze(dim)

    # pre-compute derivatives
    sin_x = sin(x)
    dsin = {0: sin_x, 1: cos(x)}

    def dn(*vs) -> Tensor:
        """Contract the derivative tensor along the vectors."""
        n = len(vs)
        sign = 1 if n % 4 in [0, 1] else -1
        func = dsin[0] if n % 2 == 0 else dsin[1]
        return tensor_prod(sign * func, *vs)

    vs_out = stack([zeros_like(sin_x) for _ in range(K)], dim=dim)

    for k in range(K):
        for sigma in integer_partitions(k + 1):
            vs_contract = [vs.select(dim, i - 1) for i in sigma]
            nu = multiplicity(sigma)
            vs_out.select(dim, k).add_(dn(*vs_contract), alpha=nu)

    return cat([sin_x.unsqueeze(dim), vs_out], dim=dim)


def jet_tanh(
    s: PrimalAndCoefficients, K=None, vmap: bool = False
) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the hyperbolic tangent function.

    Args:
        s: The stacked input and Taylor coefficients.

    Returns:
        The stacked output and Taylor coefficients.
    """
    dim = 1 if vmap else 0
    x, vs = s.split([1, K], dim=dim)
    x = x.squeeze(dim)

    # pre-compute derivatives
    tanh_x = x.tanh()
    sech_x = 1 / x.cosh()
    dtanh = {0: tanh_x}
    if K >= 1:
        dtanh[1] = sech_x**2
    if K >= 2:
        dtanh[2] = -2 * dtanh[0] * dtanh[1]
    if K >= 3:
        dtanh[3] = -2 * dtanh[1] ** 2 + 4 * dtanh[0] ** 2 * dtanh[1]
    if K > 3:
        raise NotImplementedError(
            f"Tanh only supports derivatives up to third order. Got {K}."
        )

    def dn(*vs) -> Tensor:
        """Contract the derivative tensor along the vectors."""
        return tensor_prod(dtanh[len(vs)], *vs)

    vs_out = stack([zeros_like(tanh_x) for _ in range(K)], dim=dim)

    for k in range(K):
        for sigma in integer_partitions(k + 1):
            vs_contract = [vs.select(dim, i - 1) for i in sigma]
            nu = multiplicity(sigma)
            vs_out.select(dim, k).add_(dn(*vs_contract), alpha=nu)

    return cat([tanh_x.unsqueeze(dim), vs_out], dim=dim)


def jet_sigmoid(
    s: PrimalAndCoefficients, K=None, vmap: bool = False
) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the sigmoid function.

    Args:
        s: The stacked input and Taylor coefficients.

    Returns:
        The stacked output and Taylor coefficients.
    """
    dim = 1 if vmap else 0
    x, vs = s.split([1, K], dim=dim)
    x = x.squeeze(dim)

    # pre-compute derivatives
    sigmoid_x = sigmoid(x)
    dsigmoid = {0: sigmoid_x}
    if K >= 1:
        dsigmoid[1] = sigmoid_x * (1 - sigmoid_x)
    if K >= 2:
        dsigmoid[2] = dsigmoid[1] * (1 - 2 * sigmoid_x)
    if K >= 3:
        raise NotImplementedError(
            f"Sigmoid only supports derivatives up to second order. Got {K}."
        )

    def dn(*vs) -> Tensor:
        """Contract the derivative tensor along the vectors."""
        return tensor_prod(dsigmoid[len(vs)], *vs)

    vs_out = stack([zeros_like(sigmoid_x) for _ in range(K)], dim=dim)

    for k in range(K):
        for sigma in integer_partitions(k + 1):
            vs_contract = [vs.select(dim, i - 1) for i in sigma]
            nu = multiplicity(sigma)
            vs_out.select(dim, k).add_(dn(*vs_contract), alpha=nu)

    return cat([sigmoid_x.unsqueeze(dim), vs_out], dim=dim)


def jet_linear(
    s: PrimalAndCoefficients,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    K=None,
    vmap: bool = False,
) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the linear function.

    Args:
        s: The stacked input and Taylor coefficients.

    Returns:
        The stacked output and Taylor coefficients.
    """
    dim = 1 if vmap else 0
    x, vs = s.split([1, K], dim=dim)

    linear_x = linear(x, weight, bias=bias)
    vs_out = linear(vs, weight)

    return cat([linear_x, vs_out], dim=dim)


MAPPING = {
    sin: jet_sin,
    tanh: jet_tanh,
    sigmoid: jet_sigmoid,
    linear: jet_linear,
}
