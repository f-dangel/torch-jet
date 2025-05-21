"""Implementation of AD primitives in Taylor-mode arithmetic."""

from typing import Dict, List, Optional, Tuple

from scipy.special import factorial, stirling2
from torch import Tensor, cos, mul, sigmoid, sin, tanh
from torch.nn.functional import linear

from jet.utils import (
    Primal,
    PrimalAndCoefficients,
    Value,
    ValueAndCoefficients,
    integer_partitions,
    multiplicity,
)


def _faa_di_bruno(vs: Tuple[Primal, ...], K: int, dn: Dict[int, Primal]) -> List[Value]:
    """Apply Faà di Bruno's formula for elementwise functions.

    Args:
        vs: The incoming Taylor coefficients.
        K: The order of the Taylor expansion.
        dn: A dictionary mapping the degree to the function's derivative.

    Returns:
        The outgoing Taylor coefficients.
    """
    vs_out = []
    for k in range(K):
        for idx, sigma in enumerate(integer_partitions(k + 1)):
            vs_count = {i: sigma.count(i) for i in sigma}
            vs_contract = [
                vs[i - 1] ** count if count > 1 else vs[i - 1]
                for i, count in vs_count.items()
            ]
            term = vs_contract[0]
            for v in vs_contract[1:]:
                term = mul(term, v)
            term = mul(term, dn[len(sigma)])

            nu = multiplicity(sigma)
            # avoid multiplication by one
            term = nu * term if nu != 1.0 else term
            vs_out.append(term if idx == 0 else vs_out.pop(-1) + term)
    return vs_out


def jet_sin(s: PrimalAndCoefficients, K: int, vmap: bool) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the sine function.

    Args:
        s: The primal and its Taylor coefficients.
        K: The order of the Taylor expansion.
        vmap: Whether to `vmap` the primal value and its Taylor coefficients.

    Returns:
        The value and its Taylor coefficients.
    """
    x, vs = s[0], s[1:]

    # pre-compute derivatives
    sin_x = sin(x)
    dsin = {0: sin_x}
    for k in range(1, K + 1):
        if k == 1:
            dsin[k] = cos(x)
        elif k in {2, 3}:
            dsin[k] = -1 * dsin[k - 2]
        else:
            dsin[k] = dsin[k - 4]

    vs_out = _faa_di_bruno(vs, K, dsin)

    return (sin_x, *vs_out)


def jet_tanh(s: PrimalAndCoefficients, K: int, vmap: bool) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the hyperbolic tangent function.

    Args:
        s: The primal and its Taylor coefficients.
        K: The order of the Taylor expansion.
        vmap: Whether to `vmap` the primal value and its Taylor coefficients.

    Returns:
        The value and its Taylor coefficients.
    """
    x, vs = s[0], s[1:]

    # pre-compute derivatives
    tanh_x = tanh(x)
    dtanh = {0: tanh_x}

    # Use the explicit form of the derivative polynomials for tanh from "Derivative
    # polynomials for tanh, tan, sech and sec in explicit form" by Boyadzhiev (2006)
    # (https://www.fq.math.ca/Papers1/45-4/quartboyadzhiev04_2007.pdf);
    # see also this answer: https://math.stackexchange.com/a/4226178
    if K >= 1:
        tanh_inc = tanh_x + 1
        tanh_dec = tanh_x - 1

        # required powers of tanh_dec
        tanh_dec_powers = {1: tanh_dec}
        if K >= 2:
            for k in range(2, K + 1):
                tanh_dec_powers[k] = tanh_dec**k

        # Equations (3.3) and (3.4) from the above paper
        for m in range(1, K + 1):
            # Use that the Stirling number S(m>0, 0) = 0 to start the summation at 1
            term = None
            for k in range(1, m + 1):
                scale = factorial(k, exact=True) / 2**k * stirling2(m, k, exact=True)
                # avoid multiplication by one
                term_k = (
                    (scale * tanh_dec_powers[k]) if scale != 1.0 else tanh_dec_powers[k]
                )
                term = term_k if term is None else term + term_k
            dtanh[m] = (-2) ** m * tanh_inc * term

    vs_out = _faa_di_bruno(vs, K, dtanh)

    return (tanh_x, *vs_out)


def jet_sigmoid(s: PrimalAndCoefficients, K: int, vmap: bool) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the sigmoid function.

    Args:
        s: The primal and its Taylor coefficients.
        K: The order of the Taylor expansion.
        vmap: Whether to `vmap` the primal value and its Taylor coefficients.

    Returns:
        The value and its Taylor coefficients.
    """
    x, vs = s[0], s[1:]

    # pre-compute derivatives
    sigmoid_x = sigmoid(x)
    dsigmoid = {0: sigmoid_x}

    # Use the Stirling form of the sigmoid derivatives, see Equation 20
    # of "On the Derivatives of the Sigmoid" by Minai and Williams (1993)
    # (https://eecs.ceas.uc.edu/~minaiaa/papers/minai_sigmoids_NN93.pdf)
    if K >= 1:
        # The Stirling form requires sigmoid powers
        sigmoid_powers = {1: sigmoid_x}
        for n in range(2, K + 2):
            sigmoid_powers[n] = sigmoid_x**n

        for n in range(1, K + 1):
            term = None
            for k in range(1, n + 2):
                scale = (
                    (-1) ** (k - 1)
                    * factorial(k - 1, exact=True)
                    * stirling2(n + 1, k, exact=True)
                )
                # avoid multiplication by one
                term_k = (
                    scale * sigmoid_powers[k] if scale != 1.0 else sigmoid_powers[k]
                )
                term = term_k if term is None else term + term_k
            dsigmoid[n] = term

    vs_out = _faa_di_bruno(vs, K, dsigmoid)

    return (sigmoid_x, *vs_out)


def jet_linear(
    s: PrimalAndCoefficients,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    K=None,
    vmap: bool = False,
) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the linear function.

    Args:
        s: The primal and its Taylor coefficients.
        weight: The weight matrix.
        bias: The (optional) bias vector.
        K: The order of the Taylor expansion.
        vmap: Whether to `vmap` the primal value and its Taylor coefficients.

    Returns:
        The value and its Taylor coefficients.
    """
    x, vs = s[0], s[1:]

    linear_x = linear(x, weight, bias=bias)
    vs_out = [linear(vs[k], weight) for k in range(K)]

    return (linear_x, *vs_out)


MAPPING = {
    sin: jet_sin,
    tanh: jet_tanh,
    sigmoid: jet_sigmoid,
    linear: jet_linear,
}
