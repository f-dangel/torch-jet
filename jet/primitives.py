"""Implementation of AD primitives in Taylor-mode arithmetic.

A *primitive* is an irreducible jet rule -- an op that is not expressed in terms
of other rules (those live in :mod:`jet.compositions`). Each op appears here with
both its standard and collapsed bodies co-located. Collapsed Taylor mode
propagates a single "collapsed jet" with mixed shapes:

  - Coefficient 0 (primal): shape (...)
  - Coefficients 1..K-1: shape (R, ...) -- batched over R directions
  - Coefficient K: shape (...) -- already collapsed (summed over directions)

At each nonlinear collapsed op, the K-th output coefficient is computed as
``out_K = LINEAR_TERM(in_K_collapsed) + NONLINEAR_TERMS(in_1..K-1).sum(0)``.
"""

from operator import add, sub
from typing import Callable, Self

from scipy.special import comb, factorial, stirling2
from torch import (
    Tensor,
    cat,
    cos,
    exp,
    log,
    matmul,
    ops,
    relu,
    sigmoid,
    sin,
    tanh,
    zeros_like,
)
from torch.func import vmap
from torch.utils._pytree import register_pytree_node

from jet.utils import integer_partitions, multiplicity


class JetTuple(tuple):
    """A Taylor jet ``(primal, c_1, ..., c_K)`` carrying a ``collapsed`` flag.

    ``collapsed`` is ``False`` for standard Taylor mode and ``True`` for
    collapsed mode (where coefficients ``c_1..c_{K-1}`` carry a leading
    direction dim ``R`` and ``c_K`` is already summed over it).
    """

    def __new__(cls, iterable=(), *, collapsed: bool) -> Self:
        """Build a jet from ``iterable``, tagging it standard or collapsed."""
        obj = super().__new__(cls, iterable)
        obj.collapsed = collapsed
        return obj


# Register with PyTorch's pytree so that vmap, make_fx, etc. can flatten/unflatten
# JetTuple the same way they handle plain tuples; the ``collapsed`` flag rides in
# the pytree context so it survives the roundtrip.
register_pytree_node(
    JetTuple,
    flatten_fn=lambda x: (list(x), x.collapsed),
    unflatten_fn=lambda values, collapsed: JetTuple(values, collapsed=collapsed),
)


def _jet_order(*args: Tensor) -> int:
    """Infer the Taylor-expansion order ``K`` from the ``JetTuple`` positional args.

    A jet is exactly ``(primal, c_1, ..., c_K)``, so ``K = len(jet) - 1``.
    Collects ``K`` from every ``JetTuple`` argument in a single pass and
    requires exactly one distinct value.

    Args:
        args: Positional arguments of a jet op.

    Returns:
        The Taylor-expansion order ``K``.

    Raises:
        TypeError: If no positional argument is a ``JetTuple``.
        ValueError: If two or more ``JetTuple`` args have different lengths
            (inconsistent Taylor-expansion orders).
    """
    Ks = {len(arg) - 1 for arg in args if isinstance(arg, JetTuple)}
    if not Ks:
        raise TypeError("_jet_order: no JetTuple in positional arguments")
    if len(Ks) > 1:
        raise ValueError(
            f"all JetTuple arguments must share the same derivative order; "
            f"got {sorted(Ks)}"
        )
    return Ks.pop()


def _is_batched(k: int, K: int, collapsed: bool) -> bool:
    """Whether coefficient ``k`` carries the leading direction dim ``R``.

    Only collapsed mode's coefficients ``c_1..c_{K-1}`` are batched over ``R``;
    the primal (``k == 0``) and the collapsed ``K``-th slot are not, and standard
    mode has no batched coefficients at all. This single predicate is what every
    rule gates its R-specific machinery (vmap, dim-shift, R-aware broadcast,
    R-sum) on, so standard mode degenerates to the plain coefficient-wise path.
    """
    return collapsed and 0 < k < K


def _apply_linear(self: JetTuple, op: Callable[[Tensor], Tensor]) -> JetTuple:
    """Apply a linear ``op`` coefficient-wise to every entry of ``self``.

    Linear ops commute with the Taylor expansion, so the result's coefficients
    are just ``op`` applied to each input coefficient. Standard mode applies
    ``op`` to every entry; collapsed mode vmaps the batched coefficients
    ``c_1..c_{K-1}`` over the direction dim ``R`` and applies ``op`` directly to
    the primal and the already-collapsed slot ``c_K``.

    Args:
        self: The primal and its Taylor coefficients.
        op: A linear function from a single coefficient tensor to a tensor.

    Returns:
        The value and its Taylor coefficients, with ``op`` applied to each.
    """
    return JetTuple(
        (op(self[0]), *_apply_linear_coeffs(self, op)), collapsed=self.collapsed
    )


def _apply_linear_coeffs(
    self: JetTuple, op: Callable[[Tensor], Tensor]
) -> tuple[Tensor, ...]:
    """Apply a linear ``op`` to coefficients 1..K of ``self`` (skipping the primal).

    Used by ops that handle the primal separately (e.g. the pooling and
    nll-loss rules, which compute the primal's value and indices first).
    Standard mode applies ``op`` to every coefficient; collapsed mode vmaps the
    batched coefficients ``c_1..c_{K-1}`` over the direction dim ``R`` and
    applies ``op`` directly to the already-collapsed slot ``c_K``.
    """
    K = len(self) - 1
    vop = vmap(op)
    return tuple(
        vop(self[k]) if _is_batched(k, K, self.collapsed) else op(self[k])
        for k in range(1, K + 1)
    )


def _broadcast_coeffs(self: JetTuple, primal: Tensor) -> list[Tensor]:
    """Broadcast a jet's coefficients (orders 1..K) up to ``primal``'s shape.

    For ``jet + constant`` (and ``sub``) where the constant is larger than the
    jet: the constant contributes nothing to the coefficients, but the result
    primal broadcasts up, so each coefficient must broadcast to match. A no-op
    when a coefficient is already ``primal``-shaped (the common case).

    In collapsed mode the batched coefficients (orders ``1..K-1``) carry a
    leading direction dim ``R``, so they broadcast R-aware (insert size-1 dims
    after ``R``, then expand); plain PyTorch broadcasting would left-pad at the
    front and shift ``R`` into a primal dim when ranks differ. The collapsed
    ``K``-th coefficient and the primal carry no ``R`` and broadcast normally --
    which is also exactly the standard-mode path for every coefficient.
    """
    K = len(self) - 1
    out = []
    for k in range(1, K + 1):
        c = self[k]
        if _is_batched(k, K, self.collapsed):  # batched (R, *S) -> (R, *primal.shape)
            target = (c.shape[0], *primal.shape)
            if c.shape != target:
                pad = primal.ndim - (c.ndim - 1)
                c = c.reshape(c.shape[0], *([1] * pad), *c.shape[1:]).broadcast_to(
                    target
                )
        elif c.shape != primal.shape:  # no R: standard, or the collapsed K-th
            c = c.broadcast_to(primal.shape)
        out.append(c)
    return out


def _pointwise(
    self: JetTuple,
    other: JetTuple,
    op: Callable[[Tensor, Tensor], Tensor],
) -> JetTuple:
    """Apply pointwise ``op`` to two jets of equal order and mode.

    Broadcasts both operands' coefficients up to the result primal's shape via
    :func:`_broadcast_coeffs` before ``op``, so different-rank operands align
    over their primal dims (R-aware in collapsed mode).

    ``op`` must be elementwise (it broadcasts over leading dims); the product
    rules use the Leibniz helpers instead (see :func:`_apply_bilinear`).
    """
    _jet_order(self, other)  # validates K-consistency, raises on mismatch
    primal = op(self[0], other[0])
    s_coeffs = _broadcast_coeffs(self, primal)
    o_coeffs = _broadcast_coeffs(other, primal)
    coeffs = (op(s, o) for s, o in zip(s_coeffs, o_coeffs))
    return JetTuple((primal, *coeffs), collapsed=self.collapsed)


def _leibniz(
    self: JetTuple,
    other: JetTuple,
    binary_op: Callable[[Tensor, Tensor], Tensor],
) -> tuple[Tensor, ...]:
    """Apply the Leibniz product rule for a bilinear ``binary_op`` (orders 1..K).

    The k-th coefficient of ``binary_op(self, other)`` (treated as functions of
    ``t``) is ``sum_{j=0}^{k} C(k, j) * binary_op(self[j], other[k - j])``. This
    helper returns only **coefficients 1..K**; the caller handles the order-0
    coefficient (the primal) explicitly. For ``mul``/``mm`` that's just
    ``binary_op(self[0], other[0])``; for ``addmm`` it's
    ``addmm(bias, mat1[0], mat2[0])`` — skipping the k=0 term inside the helper
    avoids tracing a wasted ``binary_op(self[0], other[0])`` node into the
    captured FX graph.

    Mode-aware: in collapsed mode the batched coefficients (orders ``1..K-1``)
    carry a leading direction dim ``R``, so a product touching one is mapped per
    direction with ``vmap``; the collapsed ``K``-th coefficient carries no ``R``,
    so its batched terms (both operands carrying ``R``) are summed back over the
    direction dim. Standard mode has no batched coefficients, so every product is
    a plain ``binary_op`` and nothing is summed -- the ordinary Leibniz rule.

    Both operands must share the same Taylor-expansion order ``K``; ``K`` is
    inferred as ``len(self) - 1`` and the lengths are checked.

    Args:
        self: The first operand jet.
        other: The second operand jet (same length as ``self``).
        binary_op: A bilinear function from two coefficient tensors to a tensor
            (e.g. elementwise ``*``, or ``torch.matmul``).

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
    collapsed = self.collapsed

    def product(j: int, k: int) -> Tensor:
        """``binary_op(self[j], other[k - j])``, vmapped over ``R`` if batched.

        ``binary_op`` may be an op (``conv``) that cannot broadcast over a leading
        batch dim, so ``R`` is mapped explicitly per direction. When neither
        operand is batched (all of standard mode, plus the collapsed linear
        terms) it is a plain call.
        """
        a, b = self[j], other[k - j]
        a_batched = _is_batched(j, K, collapsed)
        b_batched = _is_batched(k - j, K, collapsed)
        if not (a_batched or b_batched):
            return binary_op(a, b)
        in_dims = (0 if a_batched else None, 0 if b_batched else None)
        return vmap(binary_op, in_dims=in_dims)(a, b)

    coeffs = ()
    for k in range(1, K + 1):
        term = None
        for j in range(k + 1):
            term_j = comb(k, j, exact=True) * product(j, k)
            # The collapsed K-th coefficient carries no R, so reduce its batched
            # (R-carrying) terms over the direction dim; standard never enters.
            if k == K and _is_batched(j, K, collapsed):
                term_j = term_j.sum(0)
            term = term_j if term is None else term + term_j
        coeffs += (term,)
    return coeffs


def _apply_bilinear(
    op: Callable[[Tensor, Tensor], Tensor],
    self: Tensor | JetTuple,
    other: Tensor | JetTuple,
) -> Tensor | JetTuple:
    """Lift a bilinear tensor ``op`` to operands each of which may be jet or constant.

    For a product-like bilinear ``op`` (elementwise ``mul``, ``mm``, a bias-free
    convolution), the Taylor rule depends on which operands carry coefficients:

    - both jets: the primal ``op(self[0], other[0])`` plus the Leibniz product
      rule (:func:`_leibniz`) for orders 1..K;
    - one jet: ``op`` is linear in that operand, so it maps coefficient-wise
      (:func:`_apply_linear`);
    - neither: a plain ``op`` on two constants -- the all-constant sub-expression
      that arises *inside* ``addmm`` / ``convolution`` / ``native_batch_norm``.
      The binary jet ops assume at least one jet operand, so this fallback keeps
      the composed rule total.

    Mode-agnostic: both :func:`_leibniz` (both-jet) and :func:`_apply_linear`
    (one-sided) follow the jet's ``collapsed`` flag. Only valid for **bilinear**
    (product-like) ops; ``add`` / ``sub`` follow the additive rule
    (coefficient-wise sum), not Leibniz.

    Args:
        op: A bilinear function of two coefficient tensors.
        self: The first operand; a jet or a constant ``Tensor``.
        other: The second operand; a jet or a constant ``Tensor``.

    Returns:
        The jet of ``op(self, other)``, or a plain constant when both operands
        are constants.
    """
    self_is_jet = isinstance(self, JetTuple)
    other_is_jet = isinstance(other, JetTuple)
    if not (self_is_jet or other_is_jet):
        return op(self, other)
    if self_is_jet and other_is_jet:
        primal = op(self[0], other[0])
        return JetTuple((primal, *_leibniz(self, other, op)), collapsed=self.collapsed)
    if self_is_jet:
        return _apply_linear(self, lambda c: op(c, other))
    return _apply_linear(other, lambda c: op(self, c))


def _partition_term(
    vs: tuple[Tensor, ...], sigma: tuple[int, ...], dn: dict[int, Tensor]
) -> Tensor | None:
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


def _collapsed_highest_order(vs: tuple[Tensor, ...], dn: dict[int, Tensor]) -> Tensor:
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
    vs: tuple[Tensor, ...],
    dn: dict[int, Tensor],
    collapsed: bool = False,
) -> list[Tensor]:
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
            # ``result is None`` means every Faà di Bruno term vanished
            # structurally (all required outer derivatives are ``None``), as
            # happens for a constant outer function like ``x ** 0``. Materialize
            # the zero coefficient with the matching input-coefficient shape
            # (``vs[k]`` is ``(R, *S)`` in collapsed mode, ``S`` in standard).
            vs_out.append(zeros_like(vs[k]) if result is None else result)
    return vs_out


# --- Derivative helpers (shared with collapsed mode) ---


def _sin_derivatives(x0: Tensor, K: int) -> dict[int, Tensor]:
    """Compute ``sin(x0)`` and its derivatives up to order *K* (order 0 is the value)."""
    sin_x0 = sin(x0)
    d = {0: sin_x0}
    for k in range(1, K + 1):
        if k == 1:
            d[k] = cos(x0)
        elif k in {2, 3}:
            d[k] = -1 * d[k - 2]
        else:
            d[k] = d[k - 4]
    return d


def _cos_derivatives(x0: Tensor, K: int) -> dict[int, Tensor]:
    """Compute ``cos(x0)`` and its derivatives up to order *K* (order 0 is the value)."""
    cos_x0 = cos(x0)
    d = {0: cos_x0}
    for k in range(1, K + 1):
        if k == 1:
            d[k] = -1 * sin(x0)
        elif k in {2, 3}:
            d[k] = -1 * d[k - 2]
        else:
            d[k] = d[k - 4]
    return d


def _tanh_derivatives(x0: Tensor, K: int) -> dict[int, Tensor]:
    """Compute ``tanh(x0)`` and its derivatives up to order *K* (order 0 is the value)."""
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
    return d


def _sigmoid_derivatives(x0: Tensor, K: int) -> dict[int, Tensor]:
    """Compute ``sigmoid(x0)`` and its derivatives up to order *K* (order 0 is the value)."""
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
    return d


def _relu_derivatives(x0: Tensor, K: int) -> dict[int, Tensor | None]:
    """Compute ``relu(x0)`` and its derivatives up to order *K*.

    ReLU is piecewise linear, so its first derivative is the indicator
    ``x0 > 0`` and every higher derivative vanishes (we adopt the subgradient
    convention ``relu'(0) = 0``, matching PyTorch's autograd). Returning
    ``None`` for orders ``>= 2`` lets Faà di Bruno skip those structurally-zero
    terms; the only surviving order-``k`` term is the linear one,
    ``(x0 > 0) * c_k``.
    """
    relu_x0 = relu(x0)
    d: dict[int, Tensor | None] = {0: relu_x0}
    if K >= 1:
        d[1] = (x0 > 0).to(x0.dtype)
    for k in range(2, K + 1):
        d[k] = None
    return d


def _exp_derivatives(x0: Tensor, K: int) -> dict[int, Tensor]:
    """Compute ``exp(x0)`` and its derivatives up to order *K* (order 0 is the value).

    Every derivative of ``exp`` is ``exp`` itself, so all orders share the
    single ``exp(x0)`` tensor.
    """
    return dict.fromkeys(range(K + 1), exp(x0))


def _pow_derivatives(
    x0: Tensor, exponent: float | int, K: int
) -> dict[int, Tensor | None]:
    """Compute ``x0 ** exponent`` and its derivatives up to order *K* (order 0 is the value)."""
    d = {0: x0**exponent}
    for k in range(1, K + 1):
        if exponent - k < 0 and int(exponent) == exponent and exponent >= 0:
            d[k] = None
        elif exponent == k:
            d[k] = factorial(int(exponent), exact=True)
        else:
            scale = 1
            for i in range(1, k + 1):
                scale *= exponent + 1 - i
            d[k] = scale * x0 if exponent - k == 1 else scale * x0 ** (exponent - k)
    return d


def _log_derivatives(x0: Tensor, K: int) -> dict[int, Tensor]:
    """Compute ``log(x0)`` and its derivatives up to order *K* (order 0 is the value)."""
    log_x0 = log(x0)
    dpow = _pow_derivatives(x0, -1, K - 1)
    return {k: log_x0 if k == 0 else dpow[k - 1] for k in range(K + 1)}


# --- Elementwise unary ---


def _elementwise(
    self: JetTuple,
    deriv_fn: Callable[[Tensor, int], dict[int, Tensor]],
) -> JetTuple:
    """Generic elementwise jet rule (standard *or* collapsed).

    Computes the value and Taylor coefficients of an elementwise unary op from
    its derivatives ``deriv_fn`` (``dn[0]`` is the primal) via Faà di Bruno,
    reading the standard/collapsed mode from ``self.collapsed``.
    """
    K = _jet_order(self)
    dn = deriv_fn(self[0], K)
    vs_out = _faa_di_bruno(self[1:], dn, collapsed=self.collapsed)
    return JetTuple((dn[0], *vs_out), collapsed=self.collapsed)


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
    return _elementwise(self, lambda x, k: _pow_derivatives(x, exponent, k))


# --- Arithmetic ---


def _addsub(
    self: Tensor | JetTuple | float | int,
    other: Tensor | JetTuple | float | int,
    op: Callable[[Tensor, Tensor], Tensor],
    neg: Callable[[Tensor], Tensor],
) -> Tensor | JetTuple | float | int:
    """Shared body for the additive rules ``add`` and ``sub``.

    Additive ops are linear, so a one-sided jet maps coefficient-wise (the
    constant operand contributes only to the primal). ``neg`` flips the sign of
    the coefficients when the jet is the *second* operand of a subtraction
    (``self - other`` differentiates ``other`` with a minus); it is the identity
    for ``add``. Mode-agnostic: the broadcast helper and constructor follow the
    jet operand's ``collapsed`` flag.
    """
    self_is = isinstance(self, JetTuple)
    other_is = isinstance(other, JetTuple)
    if self_is and other_is:
        return _pointwise(self, other, op)
    if self_is:
        primal = op(self[0], other)
        return JetTuple(
            (primal, *_broadcast_coeffs(self, primal)), collapsed=self.collapsed
        )
    if other_is:
        primal = op(self, other[0])
        coeffs = _broadcast_coeffs(other, primal)
        return JetTuple((primal, *map(neg, coeffs)), collapsed=other.collapsed)
    return op(self, other)


def jet_add(
    self: Tensor | JetTuple | float | int,
    other: Tensor | JetTuple | float | int,
) -> Tensor | JetTuple | float | int:
    """Taylor-mode arithmetic for ``aten.add(self, other)`` (standard or collapsed).

    Args:
        self: The first operand and its Taylor coefficients, or a constant.
        other: The second operand and its Taylor coefficients, or a constant.

    Returns:
        The value and its Taylor coefficients, or a plain constant when both
        operands are constants.
    """
    return _addsub(self, other, add, lambda c: c)


def jet_sub(
    self: Tensor | JetTuple | float | int,
    other: Tensor | JetTuple | float | int,
) -> Tensor | JetTuple | float | int:
    """Taylor-mode arithmetic for ``aten.sub(self, other)`` (standard or collapsed).

    Args:
        self: The first operand and its Taylor coefficients, or a constant.
        other: The second operand and its Taylor coefficients, or a constant.

    Returns:
        The value and its Taylor coefficients, or a plain constant when both
        operands are constants.
    """
    return _addsub(self, other, sub, lambda c: -c)


def jet_mul(self: Tensor | JetTuple, other: Tensor | JetTuple) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.mul(self, other)``.

    Mode-agnostic -- :func:`_apply_bilinear` follows the operands' ``collapsed``
    flag.

    Args:
        self: The first operand and its Taylor coefficients.
        other: The second operand and its Taylor coefficients.

    Returns:
        The value and its Taylor coefficients.
    """
    return _apply_bilinear(lambda a, b: a * b, self, other)


# --- Linear decomposition ---


def jet_mm(self: Tensor | JetTuple, mat2: Tensor | JetTuple) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.mm(self, mat2)``.

    Mode-agnostic. Uses ``matmul`` rather than ``mm`` as the bilinear op: it
    agrees with ``mm`` on the 2-D coefficients standard mode passes, and is the
    form collapsed mode needs (:func:`_leibniz` vmaps it over the direction
    dim ``R``).

    Args:
        self: The first matrix and its Taylor coefficients.
        mat2: The second matrix and its Taylor coefficients.

    Returns:
        The value and its Taylor coefficients.
    """
    return _apply_bilinear(matmul, self, mat2)


def _align_conv_bias(bias: Tensor | JetTuple, ndim: int) -> Tensor | JetTuple:
    """Reshape a 1-D conv bias to broadcast over the output's batch/spatial dims.

    The bias indexes the channel dim (dim 1 of an ``ndim``-D conv output);
    appending ``ndim - 2`` trailing size-1 dims lets ``add`` broadcast it over
    the batch and spatial dims. A jet bias is reshaped coefficient-wise (the
    ``Tensor`` check separates a constant bias).

    Args:
        bias: The 1-D bias; a jet or a constant ``Tensor``.
        ndim: The convolution output's rank.

    Returns:
        The bias reshaped to ``(C_out, 1, ...)``.
    """
    tail = (1,) * (ndim - 2)
    if isinstance(bias, Tensor):
        return bias.reshape(*bias.shape, *tail)
    return _apply_linear(bias, lambda b: b.reshape(*b.shape, *tail))


def _gather_at_indices(c: Tensor, indices: Tensor) -> Tensor:
    """Select pooling coefficients at the primal's arg-max ``indices``.

    ``indices`` (shape ``(*lead, oH, oW)``, as returned by
    ``max_pool2d_with_indices``) index into the flattened spatial plane ``H*W``
    of ``c`` (shape ``(*lead, H, W)``), per leading dim. The leading dims
    ``lead`` are ``(N, C)`` for a batched ``(N, C, H, W)`` input and ``(C,)``
    for an unbatched ``(C, H, W)`` one. Max pooling is piecewise linear, so
    every Taylor coefficient is selected at the same positions the primal's
    max chose.
    """
    *lead, H, W = c.shape
    *_, oH, oW = indices.shape
    flat = c.reshape(*lead, H * W)
    selected = flat.gather(flat.dim() - 1, indices.reshape(*lead, oH * oW))
    return selected.reshape(*lead, oH, oW)


def jet_max_pool2d_with_indices(
    input: JetTuple, *pool_args: object
) -> tuple[JetTuple, Tensor]:
    """Taylor-mode arithmetic for ``aten.max_pool2d_with_indices(input, ...)``.

    Returns ``(values_jet, indices)`` mirroring the ATen op's two outputs; a
    downstream ``getitem`` selects the values jet. Max pooling is piecewise
    linear: the primal picks the arg-max ``indices``, and every coefficient is
    gathered at those same positions. Mode-agnostic -- :func:`_apply_linear_coeffs`
    follows ``input.collapsed``.

    Args:
        input: The input and its Taylor coefficients.
        *pool_args: The remaining ``max_pool2d_with_indices`` structural
            arguments (``kernel_size``, ``stride``, ``padding``, ``dilation``,
            ``ceil_mode``), forwarded unchanged.

    Returns:
        A ``(values_jet, indices)`` tuple.
    """
    values0, indices = ops.aten.max_pool2d_with_indices.default(input[0], *pool_args)
    coeffs = _apply_linear_coeffs(input, lambda c: _gather_at_indices(c, indices))
    return JetTuple((values0, *coeffs), collapsed=input.collapsed), indices


def jet_max_pool2d(input: JetTuple, *pool_args: object) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.max_pool2d`` (values only).

    The fused, indices-free pooling op some backends emit (e.g. MPS). Delegates
    to :func:`jet_max_pool2d_with_indices` and drops the indices output.
    """
    jet, _ = jet_max_pool2d_with_indices(input, *pool_args)
    return jet


# --- Concatenation ---


def jet_cat(tensors: list[Tensor | JetTuple], dim: int = 0) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.cat(tensors, dim)``.

    Concatenation is linear, so each Taylor coefficient is the concatenation of
    the operands' coefficients; constant entries contribute their value to the
    primal and zeros to every higher coefficient. In collapsed mode the batched
    coefficients (orders ``1..K-1``) carry a leading direction dim ``R``, so a
    non-negative concat ``dim`` shifts by one there and constant operands pad
    with ``(R, *shape)`` zeros to match. Standard mode has no batched
    coefficients, so that machinery stays inert (``dim`` unshifted,
    ``zeros_like`` padding).

    Args:
        tensors: The list of operands; each is a jet or a constant ``Tensor``.
        dim: The concatenation dimension.

    Returns:
        The value and its Taylor coefficients.
    """
    K = _jet_order(*tensors)
    first = next(t for t in tensors if isinstance(t, JetTuple))
    collapsed = first.collapsed
    R = first[1].shape[0] if collapsed else 0

    def part(t: object, k: int, batched: bool) -> Tensor:
        if isinstance(t, JetTuple):
            return t[k]
        if k == 0:
            return t
        return t.new_zeros(R, *t.shape) if batched else zeros_like(t)

    out = []
    for k in range(K + 1):
        batched = _is_batched(k, K, collapsed)
        d = dim + 1 if (batched and dim >= 0) else dim
        out.append(cat([part(t, k, batched) for t in tensors], d))
    return JetTuple(tuple(out), collapsed=collapsed)


# --- Loss functions ---


def jet_nll_loss_forward(
    self: JetTuple,
    target: Tensor,
    weight: Tensor | None,
    reduction: int,
    ignore_index: int,
) -> tuple[JetTuple, Tensor]:
    """Taylor-mode arithmetic for ``aten.nll_loss_forward``.

    ``nll_loss_forward(input, target, ...)`` is linear in ``input`` (the
    log-probabilities) once ``target`` / ``weight`` / ``reduction`` are fixed,
    so the same ATen op is applied to every Taylor coefficient. ``target`` and
    ``weight`` are constants (a label and per-class weights), not jets.

    The op returns ``(output, total_weight)``; ``total_weight`` depends only on
    ``target`` / ``weight``, so it is computed once from the primal and passed
    through as a constant. The result is a plain ``tuple`` so the downstream
    ``operator.getitem`` (selecting ``output``) falls through to native
    indexing and recovers the output jet.

    Args:
        self: The log-probabilities and their Taylor coefficients.
        target: The (constant) class-index targets.
        weight: Optional (constant) per-class weights, or ``None``.
        reduction: The ATen reduction enum -- ``0`` (none), ``1`` (mean),
            ``2`` (sum).
        ignore_index: Target value to ignore.

    Returns:
        A ``(output_jet, total_weight)`` tuple.

    Raises:
        NotImplementedError: If ``target`` or ``weight`` is Taylor-expanded;
            both must be constant tensors (the target is a class-index label).
    """
    if isinstance(target, JetTuple) or isinstance(weight, JetTuple):
        raise NotImplementedError(
            "jet_nll_loss_forward does not support a Taylor-expanded target or "
            "weight; both must be constant tensors (the target is a class-index "
            "label)."
        )
    output, total_weight = ops.aten.nll_loss_forward.default(
        self[0], target, weight, reduction, ignore_index
    )
    coeffs = _apply_linear_coeffs(
        self,
        lambda c: ops.aten.nll_loss_forward.default(
            c, target, weight, reduction, ignore_index
        )[0],
    )
    return JetTuple((output, *coeffs), collapsed=self.collapsed), total_weight


# --- Batch norm ---


def _bn_channel_view(primal: Tensor) -> tuple[int, ...]:
    """Per-channel broadcast view ``(1, C, 1, ...)`` for an ndim-D batch-norm input.

    Reshaping a per-channel ``weight`` / ``bias`` / scale / shift to this view
    lets it broadcast over the batch and spatial dims of an ``input`` of any rank
    (1d/2d/3d batch norm). A pure-tensor helper shared with the collapsed rule.
    """
    return (1, primal.shape[1]) + (1,) * (primal.dim() - 2)


# --- Rule-building factories (registered in :mod:`jet._rules`) ---
#
# These build the rule for a category of op; registration into the single
# ``RULES`` registry lives in :mod:`jet._rules`.


def _deflinear(prim: Callable) -> Callable:
    """Build a linear jet rule (standard or collapsed).

    Applies ``prim`` coefficient-wise via the mode-aware :func:`_apply_linear`
    when ``self`` is a ``JetTuple``; a non-jet ``self`` is passed straight to
    ``prim`` (total over constants). ``prim`` must be ``aten``-style -- the
    tensor first, structural args (``size``, ``dim``, ...) after -- and
    ``*args`` / ``**kwargs`` are forwarded to it.
    """

    def rule(self, *args, **kwargs):
        if not isinstance(self, JetTuple):
            return prim(self, *args, **kwargs)
        return _apply_linear(self, lambda c: prim(c, *args, **kwargs))

    return rule


def _defzero(prim: Callable) -> Callable:
    """Build a constant-output jet rule (output independent of the input).

    Mode-agnostic (standard *or* collapsed): the Taylor expansion of a
    constant-output op has all coefficients zero, so only the primal carries
    information. ``prim`` is applied to the primal to produce the output value
    (which carries any ``dtype`` / ``device`` / ``layout`` kwargs the user
    passed). Each zero coefficient matches its input slot's shape -- which
    respects the collapsed per-slot contract (``(R, *S)`` for ``c_1..c_{K-1}``,
    ``S`` for ``c_K``) and reduces to the primal's shape in standard mode -- and
    inherits ``primal_out``'s metadata. The output mode follows ``self.collapsed``.
    """

    def rule(self: JetTuple, *args, **kwargs) -> JetTuple:
        primal_out = prim(self[0], *args, **kwargs)
        coeffs = [primal_out.new_zeros(c.shape) for c in self[1:]]
        return JetTuple([primal_out, *coeffs], collapsed=self.collapsed)

    return rule
