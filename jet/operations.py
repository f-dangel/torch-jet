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


# --- Elementwise unary ---


def jet_sin(self: JetTuple, *, derivative_order: int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.sin(self)``.

    Args:
        self: The primal and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    self0, vs = self[0], self[1:]

    sin_self0 = sin(self0)
    dsin = {0: sin_self0}
    for k in range(1, derivative_order + 1):
        if k == 1:
            dsin[k] = cos(self0)
        elif k in {2, 3}:
            dsin[k] = -1 * dsin[k - 2]
        else:
            dsin[k] = dsin[k - 4]

    vs_out = _faa_di_bruno(vs, derivative_order, dsin)

    return JetTuple((sin_self0, *vs_out))


def jet_cos(self: JetTuple, *, derivative_order: int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.cos(self)``.

    Args:
        self: The primal and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    self0, vs = self[0], self[1:]

    cos_self0 = cos(self0)
    dcos = {0: cos_self0}
    for k in range(1, derivative_order + 1):
        if k == 1:
            dcos[k] = -1 * sin(self0)
        elif k in {2, 3}:
            dcos[k] = -1 * dcos[k - 2]
        else:
            dcos[k] = dcos[k - 4]

    vs_out = _faa_di_bruno(vs, derivative_order, dcos)

    return JetTuple((cos_self0, *vs_out))


def jet_tanh(self: JetTuple, *, derivative_order: int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.tanh(self)``.

    Args:
        self: The primal and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    self0, vs = self[0], self[1:]

    tanh_self0 = tanh(self0)
    dtanh = {0: tanh_self0}

    # Use the explicit form of the derivative polynomials for tanh from "Derivative
    # polynomials for tanh, tan, sech and sec in explicit form" by Boyadzhiev (2006)
    # (https://www.fq.math.ca/Papers1/45-4/quartboyadzhiev04_2007.pdf);
    # see also this answer: https://math.stackexchange.com/a/4226178
    if derivative_order >= 1:
        tanh_inc = tanh_self0 + 1
        tanh_dec = tanh_self0 - 1

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

    return JetTuple((tanh_self0, *vs_out))


def jet_sigmoid(self: JetTuple, *, derivative_order: int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.sigmoid(self)``.

    Args:
        self: The primal and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    self0, vs = self[0], self[1:]

    sigmoid_self0 = sigmoid(self0)
    dsigmoid = {0: sigmoid_self0}

    # Use the Stirling form of the sigmoid derivatives, see Equation 20
    # of "On the Derivatives of the Sigmoid" by Minai and Williams (1993)
    # (https://eecs.ceas.uc.edu/~minaiaa/papers/minai_sigmoids_NN93.pdf)
    if derivative_order >= 1:
        # The Stirling form requires sigmoid powers
        sigmoid_powers = {1: sigmoid_self0}
        for n in range(2, derivative_order + 2):
            sigmoid_powers[n] = sigmoid_self0**n

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

    return JetTuple((sigmoid_self0, *vs_out))


# --- Power ---


def jet_pow(
    self: JetTuple, exponent: float | int, *, derivative_order: int
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.pow(self, exponent)``.

    Args:
        self: The primal and its Taylor coefficients.
        exponent: The scalar exponent.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    assert isinstance(exponent, (float, int))

    self0, vs = self[0], self[1:]

    pow_self0 = self0**exponent

    dpow = {0: pow_self0}
    for k in range(1, derivative_order + 1):
        if exponent - k < 0 and int(exponent) == exponent:
            dpow[k] = None
        elif exponent == k:
            dpow[k] = factorial(exponent, exact=True)
        else:
            scale = 1
            for i in range(1, k + 1):
                scale *= exponent + 1 - i
            dpow[k] = (
                scale * self0 if exponent - k == 1 else scale * self0 ** (exponent - k)
            )

    vs_out = _faa_di_bruno(vs, derivative_order, dpow)

    return JetTuple((pow_self0, *vs_out))


# --- Arithmetic ---


def jet_add(
    self: Primal | JetTuple | float | int,
    other: Primal | JetTuple | float | int,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.add(self, other)``.

    Args:
        self: The first operand and its Taylor coefficients, or a scalar.
        other: The second operand and its Taylor coefficients, or a scalar.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    self_is_jet = isinstance(self, JetTuple)
    other_is_jet = isinstance(other, JetTuple)

    if self_is_jet and other_is_jet:
        return JetTuple(self[k] + other[k] for k in range(derivative_order + 1))
    elif self_is_jet:
        return JetTuple(
            (self[0] + other,) + tuple(self[k] for k in range(1, derivative_order + 1))
        )
    else:
        return JetTuple(
            (other[0] + self,) + tuple(other[k] for k in range(1, derivative_order + 1))
        )


def jet_sub(
    self: Primal | JetTuple | float | int,
    other: Primal | JetTuple | float | int,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.sub(self, other)``.

    Args:
        self: The first operand and its Taylor coefficients, or a scalar.
        other: The second operand and its Taylor coefficients, or a scalar.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    self_is_jet = isinstance(self, JetTuple)
    other_is_jet = isinstance(other, JetTuple)

    if self_is_jet and other_is_jet:
        return JetTuple(self[k] - other[k] for k in range(derivative_order + 1))
    elif self_is_jet:
        return JetTuple(
            (self[0] - other,) + tuple(self[k] for k in range(1, derivative_order + 1))
        )
    else:
        return JetTuple(
            (self - other[0],)
            + tuple(-other[k] for k in range(1, derivative_order + 1))
        )


def jet_mul(
    self: Primal | JetTuple,
    other: Primal | JetTuple,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.mul(self, other)``.

    Args:
        self: The first operand and its Taylor coefficients.
        other: The second operand and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    self_is_jet = isinstance(self, JetTuple)
    other_is_jet = isinstance(other, JetTuple)

    if self_is_jet and other_is_jet:
        s_out = ()
        for k in range(derivative_order + 1):
            term = None
            for j in range(k + 1):
                term_j = comb(k, j, exact=True) * self[j] * other[k - j]
                term = term_j if term is None else term + term_j
            s_out = s_out + (term,)
        return JetTuple(s_out)

    elif self_is_jet:
        return JetTuple(other * self[k] for k in range(derivative_order + 1))
    else:
        return JetTuple(self * other[k] for k in range(derivative_order + 1))


# --- Linear decomposition ---


def jet_mm(
    self: Primal | JetTuple,
    mat2: Primal | JetTuple,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.mm(self, mat2)``.

    Args:
        self: The first matrix and its Taylor coefficients.
        mat2: The second matrix and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    self_is_jet = isinstance(self, JetTuple)
    mat2_is_jet = isinstance(mat2, JetTuple)

    if self_is_jet and mat2_is_jet:
        s_out = ()
        for k in range(derivative_order + 1):
            term = None
            for j in range(k + 1):
                term_j = comb(k, j, exact=True) * mm(self[j], mat2[k - j])
                term = term_j if term is None else term + term_j
            s_out = s_out + (term,)
        return JetTuple(s_out)

    elif self_is_jet:
        return JetTuple(mm(self[k], mat2) for k in range(derivative_order + 1))
    else:
        return JetTuple(mm(self, mat2[k]) for k in range(derivative_order + 1))


def jet_addmm(
    self: Primal,
    mat1: Primal | JetTuple,
    mat2: Primal | JetTuple,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.addmm(self, mat1, mat2)``.

    Args:
        self: The bias tensor (constant).
        mat1: The first matrix and its Taylor coefficients.
        mat2: The second matrix and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    mat1_is_jet = isinstance(mat1, JetTuple)
    mat2_is_jet = isinstance(mat2, JetTuple)

    if mat1_is_jet and mat2_is_jet:
        s_out = (addmm(self, mat1[0], mat2[0]),)
        for k in range(1, derivative_order + 1):
            term = None
            for j in range(k + 1):
                term_j = comb(k, j, exact=True) * mm(mat1[j], mat2[k - j])
                term = term_j if term is None else term + term_j
            s_out = s_out + (term,)
        return JetTuple(s_out)

    elif mat1_is_jet:
        return JetTuple(
            (addmm(self, mat1[0], mat2),)
            + tuple(mm(mat1[k], mat2) for k in range(1, derivative_order + 1))
        )
    else:
        return JetTuple(
            (addmm(self, mat1, mat2[0]),)
            + tuple(mm(mat1, mat2[k]) for k in range(1, derivative_order + 1))
        )


def jet_view(self: JetTuple, size: list[int], *, derivative_order: int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.view(self, size)``.

    Args:
        self: The primal and its Taylor coefficients.
        size: The target shape.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients, each reshaped.
    """
    return JetTuple(
        ops.aten.view.default(self[k], size) for k in range(derivative_order + 1)
    )


def jet_unsqueeze(self: JetTuple, dim: int, *, derivative_order: int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.unsqueeze(self, dim)``.

    Args:
        self: The primal and its Taylor coefficients.
        dim: The dimension to unsqueeze.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients, each unsqueezed.
    """
    return JetTuple(
        ops.aten.unsqueeze.default(self[k], dim) for k in range(derivative_order + 1)
    )


def jet_squeeze(self: JetTuple, dim: int, *, derivative_order: int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.squeeze(self, dim)``.

    Args:
        self: The primal and its Taylor coefficients.
        dim: The dimension to squeeze.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients, each squeezed.
    """
    return JetTuple(
        ops.aten.squeeze.dim(self[k], dim) for k in range(derivative_order + 1)
    )


# --- Sum (dim reduction) ---


def jet_sum(
    self: JetTuple,
    dim: list[int],
    keepdim: bool = False,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.sum(self, dim, keepdim)``.

    Args:
        self: The primal and its Taylor coefficients.
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
    return JetTuple(self[k].sum(pos) for k in range(derivative_order + 1))


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
