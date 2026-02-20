"""Implementation of AD primitives in Taylor-mode arithmetic."""

from scipy.special import comb, factorial, stirling2
from torch import addmm, cos, mm, mul, ops, sigmoid, sin, tanh
from torch.utils._pytree import register_pytree_node

from jet.utils import (
    Primal,
    Value,
    integer_partitions,
    multiplicity,
)


class JetTuple(tuple):
    """A tuple subclass marking Taylor-expanded values (primal + coefficients).

    Using a distinct type instead of plain ``tuple`` prevents false positives
    from ATen ops that take tuple arguments (e.g. padding, stride).
    """


# Register with PyTorch's pytree so that vmap, make_fx, etc. can flatten/unflatten
# JetTuple the same way they handle plain tuples.
register_pytree_node(
    JetTuple,
    flatten_fn=lambda x: (list(x), None),
    unflatten_fn=lambda values, context: JetTuple(values),
)


def _faa_di_bruno(
    vs: tuple[Primal, ...], derivative_order: int, dn: dict[int, Primal]
) -> list[Value]:
    """Apply Faà di Bruno's formula for elementwise functions.

    Args:
        vs: The incoming Taylor coefficients.
        derivative_order: The order of the Taylor expansion.
        dn: A dictionary mapping the degree to the function's derivative.

    Returns:
        The outgoing Taylor coefficients.
    """
    vs_out = []
    for k in range(derivative_order):
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
    self_and_taylor_coefficients: JetTuple, *, derivative_order: int
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.sin``.

    Args:
        self_and_taylor_coefficients: The primal and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    x, vs = self_and_taylor_coefficients[0], self_and_taylor_coefficients[1:]

    # pre-compute derivatives
    sin_x = sin(x)
    dsin = {0: sin_x}
    for k in range(1, derivative_order + 1):
        if k == 1:
            dsin[k] = cos(x)
        elif k in {2, 3}:
            dsin[k] = -1 * dsin[k - 2]
        else:
            dsin[k] = dsin[k - 4]

    vs_out = _faa_di_bruno(vs, derivative_order, dsin)

    return JetTuple((sin_x, *vs_out))


def jet_cos(
    self_and_taylor_coefficients: JetTuple, *, derivative_order: int
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.cos``.

    Args:
        self_and_taylor_coefficients: The primal and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    x, vs = self_and_taylor_coefficients[0], self_and_taylor_coefficients[1:]

    # pre-compute derivatives
    cos_x = cos(x)
    dcos = {0: cos_x}
    for k in range(1, derivative_order + 1):
        if k == 1:
            dcos[k] = -1 * sin(x)
        elif k in {2, 3}:
            dcos[k] = -1 * dcos[k - 2]
        else:
            dcos[k] = dcos[k - 4]

    vs_out = _faa_di_bruno(vs, derivative_order, dcos)

    return JetTuple((cos_x, *vs_out))


def jet_tanh(
    self_and_taylor_coefficients: JetTuple, *, derivative_order: int
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.tanh``.

    Args:
        self_and_taylor_coefficients: The primal and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    x, vs = self_and_taylor_coefficients[0], self_and_taylor_coefficients[1:]

    # pre-compute derivatives
    tanh_x = tanh(x)
    dtanh = {0: tanh_x}

    # Use the explicit form of the derivative polynomials for tanh from "Derivative
    # polynomials for tanh, tan, sech and sec in explicit form" by Boyadzhiev (2006)
    # (https://www.fq.math.ca/Papers1/45-4/quartboyadzhiev04_2007.pdf);
    # see also this answer: https://math.stackexchange.com/a/4226178
    if derivative_order >= 1:
        tanh_inc = tanh_x + 1
        tanh_dec = tanh_x - 1

        # required powers of tanh_dec
        tanh_dec_powers = {1: tanh_dec}
        if derivative_order >= 2:
            for k in range(2, derivative_order + 1):
                tanh_dec_powers[k] = tanh_dec**k

        # Equations (3.3) and (3.4) from the above paper
        for m in range(1, derivative_order + 1):
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

    vs_out = _faa_di_bruno(vs, derivative_order, dtanh)

    return JetTuple((tanh_x, *vs_out))


def jet_sigmoid(
    self_and_taylor_coefficients: JetTuple, *, derivative_order: int
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.sigmoid``.

    Args:
        self_and_taylor_coefficients: The primal and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    x, vs = self_and_taylor_coefficients[0], self_and_taylor_coefficients[1:]

    # pre-compute derivatives
    sigmoid_x = sigmoid(x)
    dsigmoid = {0: sigmoid_x}

    # Use the Stirling form of the sigmoid derivatives, see Equation 20
    # of "On the Derivatives of the Sigmoid" by Minai and Williams (1993)
    # (https://eecs.ceas.uc.edu/~minaiaa/papers/minai_sigmoids_NN93.pdf)
    if derivative_order >= 1:
        # The Stirling form requires sigmoid powers
        sigmoid_powers = {1: sigmoid_x}
        for n in range(2, derivative_order + 2):
            sigmoid_powers[n] = sigmoid_x**n

        for n in range(1, derivative_order + 1):
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

    vs_out = _faa_di_bruno(vs, derivative_order, dsigmoid)

    return JetTuple((sigmoid_x, *vs_out))


def jet_pow(
    base_and_taylor_coefficients: JetTuple,
    exponent: float | int,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.pow.Tensor_Scalar``.

    Args:
        base_and_taylor_coefficients: The primal and its Taylor coefficients.
        exponent: The scalar exponent.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    assert isinstance(exponent, (float, int))

    x, vs = base_and_taylor_coefficients[0], base_and_taylor_coefficients[1:]

    # Compute the primal value
    pow_x = x**exponent

    # Pre-compute derivatives
    dpow = {0: pow_x}
    for k in range(1, derivative_order + 1):
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
    vs_out = _faa_di_bruno(vs, derivative_order, dpow)

    return JetTuple((pow_x, *vs_out))


def jet_add(
    a: Primal | JetTuple | float | int,
    b: Primal | JetTuple | float | int,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.add.Tensor``.

    Args:
        a: The first primal and its Taylor coefficients, or a scalar.
        b: The second primal and its Taylor coefficients, or a scalar.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    a_is_jet, b_is_jet = isinstance(a, JetTuple), isinstance(b, JetTuple)

    if a_is_jet and b_is_jet:
        return JetTuple(a[k] + b[k] for k in range(derivative_order + 1))
    elif a_is_jet:
        return JetTuple(
            (a[0] + b,) + tuple(a[k] for k in range(1, derivative_order + 1))
        )
    else:
        return JetTuple(
            (b[0] + a,) + tuple(b[k] for k in range(1, derivative_order + 1))
        )


def jet_sub(
    a: Primal | JetTuple | float | int,
    b: Primal | JetTuple | float | int,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.sub.Tensor``.

    Args:
        a: The first primal and its Taylor coefficients, or a scalar.
        b: The second primal and its Taylor coefficients, or a scalar.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    a_is_jet, b_is_jet = isinstance(a, JetTuple), isinstance(b, JetTuple)

    if a_is_jet and b_is_jet:
        return JetTuple(a[k] - b[k] for k in range(derivative_order + 1))
    elif a_is_jet:
        return JetTuple(
            (a[0] - b,) + tuple(a[k] for k in range(1, derivative_order + 1))
        )
    else:
        return JetTuple(
            (a - b[0],) + tuple(-b[k] for k in range(1, derivative_order + 1))
        )


def jet_mul(
    a: Primal | JetTuple,
    b: Primal | JetTuple,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.mul.Tensor``.

    Args:
        a: The first primal and its Taylor coefficients.
        b: The second primal and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    a_is_jet, b_is_jet = isinstance(a, JetTuple), isinstance(b, JetTuple)

    if a_is_jet and b_is_jet:
        s_out = ()
        for k in range(derivative_order + 1):
            term = None
            for j in range(k + 1):
                term_j = comb(k, j, exact=True) * a[j] * b[k - j]
                term = term_j if term is None else term + term_j
            s_out = s_out + (term,)
        return JetTuple(s_out)

    elif a_is_jet:
        return JetTuple(b * a[k] for k in range(derivative_order + 1))
    else:
        return JetTuple(a * b[k] for k in range(derivative_order + 1))


def jet_sum(
    self_and_taylor_coefficients: JetTuple,
    dim: list[int],
    keepdim: bool = False,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.sum.dim_IntList``.

    Args:
        self_and_taylor_coefficients: The primal and its Taylor coefficients.
        dim: The dimensions along which to sum (list of ints).
        keepdim: Whether to keep the reduced dimension. Default: ``False``.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.

    Raises:
        NotImplementedError: If keepdim is True.
    """
    if keepdim:
        raise NotImplementedError("keepdim=True is not supported.")
    pos = dim[0] if isinstance(dim, list) else dim
    return JetTuple(
        self_and_taylor_coefficients[k].sum(pos) for k in range(derivative_order + 1)
    )


def jet_view(
    self_and_taylor_coefficients: JetTuple,
    shape: list[int],
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.view``.

    Args:
        self_and_taylor_coefficients: The primal and its Taylor coefficients.
        shape: The target shape.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients, each reshaped.
    """
    return JetTuple(
        ops.aten.view.default(self_and_taylor_coefficients[k], shape)
        for k in range(derivative_order + 1)
    )


def jet_unsqueeze(
    self_and_taylor_coefficients: JetTuple,
    dim: int,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.unsqueeze``.

    Args:
        self_and_taylor_coefficients: The primal and its Taylor coefficients.
        dim: The dimension to unsqueeze.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients, each unsqueezed.
    """
    return JetTuple(
        ops.aten.unsqueeze.default(self_and_taylor_coefficients[k], dim)
        for k in range(derivative_order + 1)
    )


def jet_squeeze(
    self_and_taylor_coefficients: JetTuple,
    dim: int,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.squeeze.dim``.

    Args:
        self_and_taylor_coefficients: The primal and its Taylor coefficients.
        dim: The dimension to squeeze.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients, each squeezed.
    """
    return JetTuple(
        ops.aten.squeeze.dim(self_and_taylor_coefficients[k], dim)
        for k in range(derivative_order + 1)
    )


def jet_mm(
    a: Primal | JetTuple,
    b: Primal | JetTuple,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.mm``.

    Args:
        a: The first matrix and its Taylor coefficients.
        b: The second matrix and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    a_is_jet, b_is_jet = isinstance(a, JetTuple), isinstance(b, JetTuple)

    if a_is_jet and b_is_jet:
        s_out = ()
        for k in range(derivative_order + 1):
            term = None
            for j in range(k + 1):
                term_j = comb(k, j, exact=True) * mm(a[j], b[k - j])
                term = term_j if term is None else term + term_j
            s_out = s_out + (term,)
        return JetTuple(s_out)

    elif a_is_jet:
        return JetTuple(mm(a[k], b) for k in range(derivative_order + 1))
    else:
        return JetTuple(mm(a, b[k]) for k in range(derivative_order + 1))


def jet_addmm(
    bias: Primal,
    a: Primal | JetTuple,
    b: Primal | JetTuple,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.addmm``.

    Args:
        bias: The bias tensor (constant).
        a: The first matrix and its Taylor coefficients.
        b: The second matrix and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    a_is_jet, b_is_jet = isinstance(a, JetTuple), isinstance(b, JetTuple)

    if a_is_jet and b_is_jet:
        s_out = (addmm(bias, a[0], b[0]),)
        for k in range(1, derivative_order + 1):
            term = None
            for j in range(k + 1):
                term_j = comb(k, j, exact=True) * mm(a[j], b[k - j])
                term = term_j if term is None else term + term_j
            s_out = s_out + (term,)
        return JetTuple(s_out)

    elif a_is_jet:
        return JetTuple(
            (addmm(bias, a[0], b),)
            + tuple(mm(a[k], b) for k in range(1, derivative_order + 1))
        )
    else:
        return JetTuple(
            (addmm(bias, a, b[0]),)
            + tuple(mm(a, b[k]) for k in range(1, derivative_order + 1))
        )


MAPPING = {
    # Elementwise unary
    ops.aten.sin.default: jet_sin,
    ops.aten.cos.default: jet_cos,
    ops.aten.tanh.default: jet_tanh,
    ops.aten.sigmoid.default: jet_sigmoid,
    # Power
    ops.aten.pow.Tensor_Scalar: jet_pow,
    # Arithmetic
    ops.aten.add.Tensor: jet_add,
    ops.aten.sub.Tensor: jet_sub,
    ops.aten.mul.Tensor: jet_mul,
    # Linear decomposition
    ops.aten.mm.default: jet_mm,
    ops.aten.addmm.default: jet_addmm,
    ops.aten.view.default: jet_view,
    ops.aten.unsqueeze.default: jet_unsqueeze,
    ops.aten.squeeze.dim: jet_squeeze,
    # Sum (dim reduction)
    ops.aten.sum.dim_IntList: jet_sum,
}
