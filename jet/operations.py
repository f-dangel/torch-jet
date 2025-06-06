"""Implementation of AD primitives in Taylor-mode arithmetic."""

import operator
from typing import Dict, List, Optional, Tuple

from scipy.special import comb, factorial, stirling2
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
            if dn[len(sigma)] is None:
                continue

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


def jet_sin(
    s: PrimalAndCoefficients, K: int, vmap: bool, is_taylor: tuple[bool, ...]
) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the sine function.

    Args:
        s: The primal and its Taylor coefficients.
        K: The order of the Taylor expansion.
        vmap: Whether to `vmap` the primal value and its Taylor coefficients.
        is_taylor: A tuple indicating which arguments are Taylor coefficients (`True`)
            and which are constants (`False`).

    Returns:
        The value and its Taylor coefficients.
    """
    assert is_taylor == (True,)

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


def jet_cos(
    s: PrimalAndCoefficients, K: int, vmap: bool, is_taylor: tuple[bool, ...]
) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the cosine function.

    Args:
        s: The primal and its Taylor coefficients.
        K: The order of the Taylor expansion.
        vmap: Whether to `vmap` the primal value and its Taylor coefficients.
        is_taylor: A tuple indicating which arguments are Taylor coefficients (`True`)
            and which are constants (`False`).

    Returns:
        The value and its Taylor coefficients.
    """
    assert is_taylor == (True,)

    x, vs = s[0], s[1:]

    # pre-compute derivatives
    cos_x = cos(x)
    dcos = {0: cos_x}
    for k in range(1, K + 1):
        if k == 1:
            dcos[k] = -1 * sin(x)
        elif k in {2, 3}:
            dcos[k] = -1 * dcos[k - 2]
        else:
            dcos[k] = dcos[k - 4]

    vs_out = _faa_di_bruno(vs, K, dcos)

    return (cos_x, *vs_out)


def jet_tanh(
    s: PrimalAndCoefficients, K: int, vmap: bool, is_taylor: tuple[bool, ...]
) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the hyperbolic tangent function.

    Args:
        s: The primal and its Taylor coefficients.
        K: The order of the Taylor expansion.
        vmap: Whether to `vmap` the primal value and its Taylor coefficients.
        is_taylor: A tuple indicating which arguments are Taylor coefficients (`True`)
            and which are constants (`False`).

    Returns:
        The value and its Taylor coefficients.
    """
    assert is_taylor == (True,)

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


def jet_sigmoid(
    s: PrimalAndCoefficients, K: int, vmap: bool, is_taylor: tuple[bool, ...]
) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the sigmoid function.

    Args:
        s: The primal and its Taylor coefficients.
        K: The order of the Taylor expansion.
        vmap: Whether to `vmap` the primal value and its Taylor coefficients.
        is_taylor: A tuple indicating which arguments are Taylor coefficients (`True`)
            and which are constants (`False`).

    Returns:
        The value and its Taylor coefficients.
    """
    assert is_taylor == (True,)

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
    is_taylor: tuple[bool, ...] = (True, False, False),
) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the linear function.

    Args:
        s: The primal and its Taylor coefficients.
        weight: The weight matrix.
        bias: The (optional) bias vector.
        K: The order of the Taylor expansion.
        vmap: Whether to `vmap` the primal value and its Taylor coefficients.
        is_taylor: A tuple indicating which arguments are Taylor coefficients (`True`)
            and which are constants (`False`).

    Returns:
        The value and its Taylor coefficients.
    """
    if is_taylor not in {(True, False, False), (True, False)}:
        raise NotImplementedError(f"Not implemented for {is_taylor=}.")

    return tuple(
        linear(s[k], weight, bias=bias if k == 0 else None) for k in range(K + 1)
    )


def jet_pow(
    s: PrimalAndCoefficients,
    exponent: float | int,
    K: int,
    vmap: bool,
    is_taylor: tuple[bool, ...],
) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the power function with integer exponent.

    Args:
        s: The primal and its Taylor coefficients.
        exponent: The integer exponent.
        K: The order of the Taylor expansion.
        vmap: Whether to `vmap` the primal value and its Taylor coefficients.
        is_taylor: A tuple indicating which arguments are Taylor coefficients (`True`)
            and which are constants (`False`).

    Returns:
        The value and its Taylor coefficients.
    """
    if is_taylor != (True, False):
        raise NotImplementedError
    assert isinstance(exponent, (float, int))

    x, vs = s[0], s[1:]

    # Compute the primal value
    pow_x = x**exponent

    # Pre-compute derivatives
    dpow = {0: pow_x}
    for k in range(1, K + 1):
        if exponent - k < 0 and int(exponent) == exponent:
            dpow[k] = None
        elif exponent == k:
            dpow[k] = factorial(exponent, exact=True)
        else:
            scale = 1
            for i in range(1, k + 1):
                scale *= exponent + 1 - i
            dpow[k] = scale * x if exponent - k == 1 else scale * x ** (exponent - k)

    # Compute Taylor coefficients using Faà di Bruno's formula
    vs_out = _faa_di_bruno(vs, K, dpow)

    return (pow_x, *vs_out)


def jet_add(
    s1: Primal | PrimalAndCoefficients | float | int,
    s2: Primal | PrimalAndCoefficients | float | int,
    K: int,
    vmap: bool,
    is_taylor: tuple[bool, ...],
) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for addition.

    Args:
        s1: The first primal and its Taylor coefficients, or a scalar.
        s2: The second primal and its Taylor coefficients, or a scalar.
        K: The order of the Taylor expansion.
        vmap: Whether to `vmap` the primal value and its Taylor coefficients.
        is_taylor: A tuple indicating which arguments are Taylor coefficients (`True`)
            and which are constants (`False`).

    Returns:
        The value and its Taylor coefficients.
    """
    coeff1, coeff2 = is_taylor

    if (coeff1, coeff2) == (True, True):
        return tuple(s1[k] + s2[k] for k in range(K + 1))
    elif (coeff1, coeff2) == (True, False):
        return (s1[0] + s2,) + tuple(s1[k] for k in range(1, K + 1))
    elif (coeff1, coeff2) == (False, True):
        return (s2[0] + s1,) + tuple(s2[k] for k in range(1, K + 1))


def jet_sub(
    s1: Primal | PrimalAndCoefficients | float | int,
    s2: Primal | PrimalAndCoefficients | float | int,
    K: int,
    vmap: bool,
    is_taylor: tuple[bool, ...],
) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for subtraction.

    Args:
        s1: The first primal and its Taylor coefficients, or a scalar.
        s2: The second primal and its Taylor coefficients, or a scalar.
        K: The order of the Taylor expansion.
        vmap: Whether to `vmap` the primal value and its Taylor coefficients.
        is_taylor: A tuple indicating which arguments are Taylor coefficients (`True`)
            and which are constants (`False`).

    Returns:
        The value and its Taylor coefficients.
    """
    (coeff1, coeff2) = is_taylor

    if (coeff1, coeff2) == (True, True):
        return tuple(s1[k] - s2[k] for k in range(K + 1))
    elif (coeff1, coeff2) == (True, False):
        return (s1[0] - s2,) + tuple(s1[k] for k in range(1, K + 1))
    elif (coeff1, coeff2) == (False, True):
        return (s2[0] - s1,) + tuple(s2[k] for k in range(1, K + 1))


def jet_mul(
    s1: Primal | PrimalAndCoefficients,
    s2: Primal | PrimalAndCoefficients,
    K: int,
    vmap: bool,
    is_taylor: tuple[bool, ...],
) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for multiplication of two variables.

    Args:
        s1: The first primal and its Taylor coefficients.
        s2: The second primal and its Taylor coefficients.
        K: The order of the Taylor expansion.
        vmap: Whether to `vmap` the primal value and its Taylor coefficients.
        is_taylor: A tuple indicating which arguments are Taylor coefficients (`True`)
            and which are constants (`False`).

    Returns:
        The value and its Taylor coefficients.
    """
    (coeff1, coeff2) = is_taylor

    if (coeff1, coeff2) == (True, True):
        s_out = ()
        for k in range(K + 1):
            term = None
            for j in range(k + 1):
                term_j = comb(k, j, exact=True) * s1[j] * s2[k - j]
                term = term_j if term is None else term + term_j
            s_out = s_out + (term,)

        return s_out

    elif (coeff1, coeff2) == (True, False):
        return tuple(s2 * s1[k] for k in range(K + 1))
    elif (coeff1, coeff2) == (False, True):
        return tuple(s1 * s2[k] for k in range(K + 1))


MAPPING = {
    sin: jet_sin,
    cos: jet_cos,
    tanh: jet_tanh,
    sigmoid: jet_sigmoid,
    linear: jet_linear,
    operator.pow: jet_pow,
    operator.add: jet_add,
    operator.sub: jet_sub,
    operator.mul: jet_mul,
    mul: jet_mul,
}
