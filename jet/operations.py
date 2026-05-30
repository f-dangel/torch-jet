"""Implementation of AD primitives in Taylor-mode arithmetic."""

from typing import Callable

from scipy.special import comb, factorial, stirling2
from torch import addmm, cos, mm, ops, sigmoid, sin, tanh, zeros_like
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


def _order(args: tuple[Value, ...], jet_type: type) -> int:
    """Infer the Taylor-expansion order ``K`` from the jet-typed positional args.

    A jet (whether ``JetTuple`` or ``CollapsedJetTuple``) is exactly
    ``(primal, c_1, ..., c_K)``, so ``K = len(jet) - 1``. Collects ``K`` from
    every ``jet_type`` argument in a single pass and requires exactly one
    distinct value.

    Args:
        args: Positional arguments of a jet op.
        jet_type: The jet tuple subclass to match (``JetTuple`` for standard
            Taylor mode, ``CollapsedJetTuple`` for collapsed).

    Returns:
        The Taylor-expansion order ``K``.

    Raises:
        TypeError: If no positional argument is an instance of ``jet_type``.
        ValueError: If two or more ``jet_type`` arguments have different lengths
            (inconsistent Taylor-expansion orders).
    """
    Ks = {len(arg) - 1 for arg in args if isinstance(arg, jet_type)}
    if not Ks:
        raise TypeError(f"_order: no {jet_type.__name__} in positional arguments")
    if len(Ks) > 1:
        raise ValueError(
            f"all {jet_type.__name__} arguments must share the same derivative "
            f"order; got {sorted(Ks)}"
        )
    return Ks.pop()


def _jet_order(*args: Value) -> int:
    """Infer ``K`` from all ``JetTuple`` positional args. See :func:`_order`."""
    return _order(args, JetTuple)


def _apply_linear(self: JetTuple, op: Callable[[Primal], Primal]) -> JetTuple:
    """Apply a linear ``op`` coefficient-wise to every entry of ``self``.

    Linear ops commute with the Taylor expansion, so the result's coefficients
    are just ``op`` applied to each input coefficient. Mirrors the analogous
    helper in :mod:`jet.collapsed_operations` (which additionally vmaps over
    the batched direction dim ``R``).

    Args:
        self: The primal and its Taylor coefficients.
        op: A linear function from a single coefficient tensor to a tensor.

    Returns:
        The value and its Taylor coefficients, with ``op`` applied to each.
    """
    return JetTuple(op(c) for c in self)


def _apply_linear_coeffs(
    self: JetTuple, op: Callable[[Primal], Primal]
) -> tuple[Primal, ...]:
    """Apply a linear ``op`` to coefficients 1..K of ``self`` (skipping the primal).

    Used by ops that handle the primal separately (e.g. ``jet_addmm``'s bias).
    """
    return tuple(op(c) for c in self[1:])


def _leibniz(
    self: JetTuple,
    other: JetTuple,
    binary_op: Callable[[Primal, Primal], Primal],
) -> tuple[Primal, ...]:
    """Apply the Leibniz product rule for a bilinear ``binary_op`` (orders 1..K).

    The k-th coefficient of ``binary_op(self, other)`` (treated as functions of
    ``t``) is ``sum_{j=0}^{k} C(k, j) * binary_op(self[j], other[k - j])``. This
    helper returns only **coefficients 1..K**; the caller handles the order-0
    coefficient (the primal) explicitly. For ``mul``/``mm`` that's just
    ``binary_op(self[0], other[0])``; for ``addmm`` it's
    ``addmm(bias, mat1[0], mat2[0])`` — skipping the k=0 term inside the helper
    avoids tracing a wasted ``binary_op(self[0], other[0])`` node into the
    captured FX graph.

    Both operands must share the same Taylor-expansion order ``K``; ``K`` is
    inferred as ``len(self) - 1`` and the lengths are checked. Mirrors
    :func:`jet.collapsed_operations._collapsed_leibniz`.

    Args:
        self: The first operand jet.
        other: The second operand jet (same length as ``self``).
        binary_op: A bilinear function from two coefficient tensors to a tensor
            (e.g. elementwise ``*``, or ``torch.mm``).

    Returns:
        The Taylor coefficients of orders 1..K (a tuple of length ``K``).

    Raises:
        ValueError: If ``self`` and ``other`` have different lengths.
    """
    if len(self) != len(other):
        raise ValueError(
            f"_leibniz: operands must share the same derivative order; "
            f"got lengths {len(self)} and {len(other)}"
        )
    K = len(self) - 1
    coeffs = ()
    for k in range(1, K + 1):
        term = None
        for j in range(k + 1):
            term_j = comb(k, j, exact=True) * binary_op(self[j], other[k - j])
            term = term_j if term is None else term + term_j
        coeffs = coeffs + (term,)
    return coeffs


def _partition_term(
    vs: tuple[Primal, ...], sigma: tuple[int, ...], dn: dict[int, Primal]
) -> Value | None:
    r"""Compute one term of the Faà di Bruno sum for a given partition.

    In Faà di Bruno's formula, the order-``k`` Taylor coefficient of the
    composition ``f(g(x))`` is a sum over the integer partitions of ``k``. A
    partition ``sigma`` is a tuple of part sizes summing to ``k``. Its number of
    blocks ``len(sigma)`` selects the outer derivative ``dn[len(sigma)]``, while
    each part of size ``i`` contributes a factor of the inner coefficient
    ``vs[i - 1]`` (the order-``i`` coefficient). Repeated parts of the same size
    are raised to the corresponding power, and the product is weighted by the
    combinatorial multiplicity ``nu`` of the partition (the number of set
    partitions of ``{1, ..., k}`` whose block sizes are ``sigma``).

    Args:
        vs: The incoming (inner) Taylor coefficients, indexed by order minus one,
            i.e. ``vs[i - 1]`` is the order-``i`` coefficient.
        sigma: An integer partition of the output order, given as a tuple of part
            sizes (e.g. ``(2, 1, 1)`` for order 4 split into three blocks).
        dn: A dictionary mapping a degree to the outer function's derivative of
            that degree.

    Returns:
        The partition's contribution to the Faà di Bruno sum, or ``None`` when
        the required outer derivative ``dn[len(sigma)]`` is ``None`` (a
        structurally vanishing term that callers skip).
    """
    if dn[len(sigma)] is None:
        return None
    vs_count = {i: sigma.count(i) for i in sigma}
    vs_contract = [
        vs[i - 1] ** count if count > 1 else vs[i - 1] for i, count in vs_count.items()
    ]
    term = vs_contract[0]
    for v in vs_contract[1:]:
        term = term * v
    term = term * dn[len(sigma)]
    nu = multiplicity(sigma)
    return nu * term if nu != 1.0 else term


def _collapsed_highest_order(vs: tuple[Primal, ...], dn: dict[int, Primal]) -> Value:
    """Compute the collapsed (summed) highest-order Faà di Bruno coefficient.

    Separates the linear contribution (which multiplies the collapsed input)
    from the nonlinear contributions (which are summed over the direction
    dimension *R*).

    Args:
        vs: The incoming Taylor coefficients (length ``K``).
        dn: A dictionary mapping the degree to the function's derivative.

    Returns:
        The collapsed highest-order coefficient.
    """
    K = len(vs)
    linear_term = dn[1] * vs[K - 1] if dn[1] is not None else None
    nonlinear_term = None
    for sigma in integer_partitions(K):
        if sigma == (K,):
            continue
        term = _partition_term(vs, sigma, dn)
        if term is not None:
            # Sum out the direction dim R per term so the accumulator (and the
            # tensors flowing through the traced graph) stay small.
            term = term.sum(0)
            nonlinear_term = term if nonlinear_term is None else nonlinear_term + term
    if nonlinear_term is not None and linear_term is not None:
        return linear_term + nonlinear_term
    elif nonlinear_term is not None:
        return nonlinear_term
    elif linear_term is not None:
        return linear_term
    return zeros_like(dn[0])


def _faa_di_bruno(
    vs: tuple[Primal, ...],
    dn: dict[int, Primal],
    collapsed: bool = False,
) -> list[Value]:
    """Apply Faà di Bruno's formula for elementwise functions.

    Args:
        vs: The incoming Taylor coefficients (length ``K``).
        dn: A dictionary mapping the degree to the function's derivative.
        collapsed: If ``True``, treat ``vs[-1]`` as a collapsed (already
            summed) coefficient and ``vs[0:-1]`` as batched with a leading
            direction dimension *R*.  The last output coefficient is then
            computed by separating the linear contribution (which multiplies
            the collapsed input) from the nonlinear contributions (which are
            summed over *R*).

    Returns:
        The outgoing Taylor coefficients.
    """
    K = len(vs)
    vs_out = []
    for k in range(K):
        order = k + 1
        if order == K and collapsed:
            vs_out.append(_collapsed_highest_order(vs, dn))
        else:
            result = None
            for sigma in integer_partitions(order):
                term = _partition_term(vs, sigma, dn)
                if term is not None:
                    result = term if result is None else result + term
            vs_out.append(result)
    return vs_out


# --- Derivative helpers (shared with collapsed mode) ---


def _sin_derivatives(x0: Primal, K: int) -> tuple[Primal, dict[int, Primal]]:
    """Compute ``sin(x0)`` and its derivatives up to order *K*."""
    sin_x0 = sin(x0)
    d = {0: sin_x0}
    for k in range(1, K + 1):
        if k == 1:
            d[k] = cos(x0)
        elif k in {2, 3}:
            d[k] = -1 * d[k - 2]
        else:
            d[k] = d[k - 4]
    return sin_x0, d


def _cos_derivatives(x0: Primal, K: int) -> tuple[Primal, dict[int, Primal]]:
    """Compute ``cos(x0)`` and its derivatives up to order *K*."""
    cos_x0 = cos(x0)
    d = {0: cos_x0}
    for k in range(1, K + 1):
        if k == 1:
            d[k] = -1 * sin(x0)
        elif k in {2, 3}:
            d[k] = -1 * d[k - 2]
        else:
            d[k] = d[k - 4]
    return cos_x0, d


def _tanh_derivatives(x0: Primal, K: int) -> tuple[Primal, dict[int, Primal]]:
    """Compute ``tanh(x0)`` and its derivatives up to order *K*."""
    # Use the explicit form of the derivative polynomials for tanh from "Derivative
    # polynomials for tanh, tan, sech and sec in explicit form" by Boyadzhiev (2006)
    # (https://www.fq.math.ca/Papers1/45-4/quartboyadzhiev04_2007.pdf);
    # see also this answer: https://math.stackexchange.com/a/4226178
    tanh_x0 = tanh(x0)
    d = {0: tanh_x0}
    if K >= 1:
        tanh_inc = tanh_x0 + 1
        tanh_dec = tanh_x0 - 1
        tanh_dec_powers = {1: tanh_dec}
        if K >= 2:
            for k in range(2, K + 1):
                tanh_dec_powers[k] = tanh_dec**k
        # Equations (3.3) and (3.4) from the above paper
        for m in range(1, K + 1):
            term = None
            # Use that the Stirling number S(m>0, 0) = 0 to start the summation at 1
            for k in range(1, m + 1):
                scale = factorial(k, exact=True) / 2**k * stirling2(m, k, exact=True)
                term_k = (
                    (scale * tanh_dec_powers[k]) if scale != 1.0 else tanh_dec_powers[k]
                )
                term = term_k if term is None else term + term_k
            d[m] = (-2) ** m * tanh_inc * term
    return tanh_x0, d


def _sigmoid_derivatives(x0: Primal, K: int) -> tuple[Primal, dict[int, Primal]]:
    """Compute ``sigmoid(x0)`` and its derivatives up to order *K*."""
    # Use the Stirling form of the sigmoid derivatives, see Equation 20
    # of "On the Derivatives of the Sigmoid" by Minai and Williams (1993)
    # (https://eecs.ceas.uc.edu/~minaiaa/papers/minai_sigmoids_NN93.pdf)
    sigmoid_x0 = sigmoid(x0)
    d = {0: sigmoid_x0}
    if K >= 1:
        sigmoid_powers = {1: sigmoid_x0}
        for n in range(2, K + 2):
            sigmoid_powers[n] = sigmoid_x0**n
        for n in range(1, K + 1):
            term = None
            for k in range(1, n + 2):
                scale = (
                    (-1) ** (k - 1)
                    * factorial(k - 1, exact=True)
                    * stirling2(n + 1, k, exact=True)
                )
                term_k = (
                    scale * sigmoid_powers[k] if scale != 1.0 else sigmoid_powers[k]
                )
                term = term_k if term is None else term + term_k
            d[n] = term
    return sigmoid_x0, d


def _pow_derivatives(
    x0: Primal, exponent: float | int, K: int
) -> tuple[Primal, dict[int, Primal | None]]:
    """Compute ``x0 ** exponent`` and its derivatives up to order *K*."""
    pow_x0 = x0**exponent
    d = {0: pow_x0}
    for k in range(1, K + 1):
        if exponent - k < 0 and int(exponent) == exponent:
            d[k] = None
        elif exponent == k:
            d[k] = factorial(exponent, exact=True)
        else:
            scale = 1
            for i in range(1, k + 1):
                scale *= exponent + 1 - i
            d[k] = scale * x0 if exponent - k == 1 else scale * x0 ** (exponent - k)
    return pow_x0, d


# --- Elementwise unary ---


def _jet_elementwise(
    self: JetTuple,
    deriv_fn: Callable[[Primal, int], tuple[Primal, dict[int, Primal]]],
) -> JetTuple:
    """Generic elementwise jet rule using shared derivative helpers.

    Args:
        self: The primal and its Taylor coefficients.
        deriv_fn: Returns the primal and the function's derivatives ``dn`` at
            the primal, e.g. ``_sin_derivatives``.

    Returns:
        The value and its Taylor coefficients.
    """
    K = _jet_order(self)
    self0, vs = self[0], self[1:]
    primal, dn = deriv_fn(self0, K)
    vs_out = _faa_di_bruno(vs, dn)
    return JetTuple((primal, *vs_out))


def jet_sin(self: JetTuple) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.sin(self)``."""
    return _jet_elementwise(self, _sin_derivatives)


def jet_cos(self: JetTuple) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.cos(self)``."""
    return _jet_elementwise(self, _cos_derivatives)


def jet_tanh(self: JetTuple) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.tanh(self)``."""
    return _jet_elementwise(self, _tanh_derivatives)


def jet_sigmoid(self: JetTuple) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.sigmoid(self)``."""
    return _jet_elementwise(self, _sigmoid_derivatives)


# --- Power ---


def jet_pow(self: JetTuple, exponent: float | int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.pow(self, exponent)``.

    Args:
        self: The primal and its Taylor coefficients.
        exponent: The scalar exponent.

    Returns:
        The value and its Taylor coefficients.
    """
    assert isinstance(exponent, (float, int))
    self0, vs = self[0], self[1:]
    pow_self0, dpow = _pow_derivatives(self0, exponent, _jet_order(self))
    vs_out = _faa_di_bruno(vs, dpow)
    return JetTuple((pow_self0, *vs_out))


# --- Arithmetic ---


def jet_add(
    self: Primal | JetTuple | float | int,
    other: Primal | JetTuple | float | int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.add(self, other)``.

    Args:
        self: The first operand and its Taylor coefficients, or a scalar.
        other: The second operand and its Taylor coefficients, or a scalar.

    Returns:
        The value and its Taylor coefficients.
    """
    self_is_jet = isinstance(self, JetTuple)
    other_is_jet = isinstance(other, JetTuple)

    if self_is_jet and other_is_jet:
        _jet_order(self, other)  # validates K-consistency, raises on mismatch
        coeffs = (s + o for s, o in zip(self, other))
    elif self_is_jet:
        coeffs = (self[0] + other, *self[1:])
    else:
        coeffs = (other[0] + self, *other[1:])
    return JetTuple(coeffs)


def jet_sub(
    self: Primal | JetTuple | float | int,
    other: Primal | JetTuple | float | int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.sub(self, other)``.

    Args:
        self: The first operand and its Taylor coefficients, or a scalar.
        other: The second operand and its Taylor coefficients, or a scalar.

    Returns:
        The value and its Taylor coefficients.
    """
    self_is_jet = isinstance(self, JetTuple)
    other_is_jet = isinstance(other, JetTuple)

    if self_is_jet and other_is_jet:
        _jet_order(self, other)  # validates K-consistency, raises on mismatch
        coeffs = (s - o for s, o in zip(self, other))
    elif self_is_jet:
        coeffs = (self[0] - other, *self[1:])
    else:
        coeffs = (self - other[0], *(-c for c in other[1:]))
    return JetTuple(coeffs)


def jet_mul(self: Primal | JetTuple, other: Primal | JetTuple) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.mul(self, other)``.

    Args:
        self: The first operand and its Taylor coefficients.
        other: The second operand and its Taylor coefficients.

    Returns:
        The value and its Taylor coefficients.
    """
    self_is_jet = isinstance(self, JetTuple)
    other_is_jet = isinstance(other, JetTuple)

    if self_is_jet and other_is_jet:
        primal = self[0] * other[0]
        return JetTuple((primal, *_leibniz(self, other, lambda a, b: a * b)))
    elif self_is_jet:
        return _apply_linear(self, lambda c: other * c)
    else:
        return _apply_linear(other, lambda c: self * c)


# --- Linear decomposition ---


def jet_mm(self: Primal | JetTuple, mat2: Primal | JetTuple) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.mm(self, mat2)``.

    Args:
        self: The first matrix and its Taylor coefficients.
        mat2: The second matrix and its Taylor coefficients.

    Returns:
        The value and its Taylor coefficients.
    """
    self_is_jet = isinstance(self, JetTuple)
    mat2_is_jet = isinstance(mat2, JetTuple)

    if self_is_jet and mat2_is_jet:
        primal = mm(self[0], mat2[0])
        return JetTuple((primal, *_leibniz(self, mat2, mm)))
    elif self_is_jet:
        return _apply_linear(self, lambda c: mm(c, mat2))
    else:
        return _apply_linear(mat2, lambda c: mm(self, c))


def jet_addmm(
    self: Primal, mat1: Primal | JetTuple, mat2: Primal | JetTuple
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.addmm(self, mat1, mat2)``.

    Args:
        self: The bias tensor. Must be a constant ``Tensor``, not a ``JetTuple``.
        mat1: The first matrix and its Taylor coefficients.
        mat2: The second matrix and its Taylor coefficients.

    Returns:
        The value and its Taylor coefficients.
    """
    if isinstance(self, JetTuple):
        raise NotImplementedError(
            "jet_addmm does not support a Taylor-expanded bias (self). "
            "Expected a constant Tensor."
        )

    mat1_is_jet = isinstance(mat1, JetTuple)
    mat2_is_jet = isinstance(mat2, JetTuple)

    if mat1_is_jet and mat2_is_jet:
        primal = addmm(self, mat1[0], mat2[0])
        return JetTuple((primal, *_leibniz(mat1, mat2, mm)))
    elif mat1_is_jet:
        primal = addmm(self, mat1[0], mat2)
        return JetTuple((primal, *_apply_linear_coeffs(mat1, lambda c: mm(c, mat2))))
    else:
        primal = addmm(self, mat1, mat2[0])
        return JetTuple((primal, *_apply_linear_coeffs(mat2, lambda c: mm(mat1, c))))


def jet_view(self: JetTuple, size: list[int]) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.view(self, size)``.

    Args:
        self: The primal and its Taylor coefficients.
        size: The target shape.

    Returns:
        The value and its Taylor coefficients, each reshaped.
    """
    return _apply_linear(self, lambda c: ops.aten.view.default(c, size))


def jet_unsqueeze(self: JetTuple, dim: int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.unsqueeze(self, dim)``.

    Args:
        self: The primal and its Taylor coefficients.
        dim: The dimension to unsqueeze.

    Returns:
        The value and its Taylor coefficients, each unsqueezed.
    """
    return _apply_linear(self, lambda c: ops.aten.unsqueeze.default(c, dim))


def jet_squeeze(self: JetTuple, dim: int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.squeeze(self, dim)``.

    Args:
        self: The primal and its Taylor coefficients.
        dim: The dimension to squeeze.

    Returns:
        The value and its Taylor coefficients, each squeezed.
    """
    return _apply_linear(self, lambda c: ops.aten.squeeze.dim(c, dim))


# --- Sum (dim reduction) ---


def jet_sum(self: JetTuple, dim: list[int], keepdim: bool = False) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.sum(self, dim, keepdim)``.

    Args:
        self: The primal and its Taylor coefficients.
        dim: The dimension to sum along, as either an ``int`` or a 1-element
            ``list[int]``. Multi-dimensional reductions are not supported.
        keepdim: Whether to keep the reduced dimension. Default: ``False``.

    Returns:
        The value and its Taylor coefficients.

    Raises:
        NotImplementedError: If keepdim is True.
        ValueError: If ``dim`` is a list with anything other than one element.
    """
    if keepdim:
        raise NotImplementedError("keepdim=True is not supported.")
    (pos,) = (dim,) if isinstance(dim, int) else dim
    return _apply_linear(self, lambda c: c.sum(pos))


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
