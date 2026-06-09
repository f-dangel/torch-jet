"""Implementation of AD primitives in collapsed Taylor-mode arithmetic.

Collapsed Taylor mode propagates a single "collapsed jet" with mixed shapes:

  - Coefficient 0 (primal): shape (...)
  - Coefficients 1..K-1: shape (R, ...) -- batched over R directions
  - Coefficient K: shape (...) -- already collapsed (summed over directions)

At each nonlinear operation, the K-th output coefficient is computed as:
  out_K = LINEAR_TERM(in_K_collapsed) + NONLINEAR_TERMS(in_1..K-1).sum(0)
"""

from functools import partial
from operator import add, sub
from typing import Callable

from scipy.special import comb
from torch import Tensor, cat, matmul, ops, zeros_like
from torch.func import vmap

from jet.operations import (
    JetTuple,
    _addmm_impl,
    _align_conv_bias,
    _elementwise,
    _exp_derivatives,
    _gather_at_indices,
    _jet_order,
    _log_derivatives,
    _log_softmax_impl,
    _make_linear_rule,
    _native_batch_norm_impl,
    _pow_derivatives,
)

#: Collapsed-mode JetTuple constructor (``JetTuple(values, collapsed=True)``).
_cjet = partial(JetTuple, collapsed=True)


# ---------------------------------------------------------------------------
# Helpers: apply a linear op with vmap for batched coefficients
# ---------------------------------------------------------------------------


def _apply_linear(jet: JetTuple, op: Callable[[Tensor], Tensor]) -> JetTuple:
    """Apply a linear *op* to every entry of *jet*, vmapping batched ones."""
    K = len(jet) - 1
    results = [op(jet[0])]
    vop = vmap(op)
    for k in range(1, K + 1):
        results.append(vop(jet[k]) if k < K else op(jet[k]))
    return _cjet(results)


def _apply_linear_coeffs(
    jet: JetTuple, op: Callable[[Tensor], Tensor]
) -> tuple[Tensor, ...]:
    """Apply *op* to coefficients 1..K only, vmapping batched ones."""
    K = len(jet) - 1
    vop = vmap(op)
    return tuple(vop(jet[k]) if k < K else op(jet[k]) for k in range(1, K + 1))


def _collapsed_pointwise(
    self: JetTuple,
    other: JetTuple,
    op: Callable[[Tensor, Tensor], Tensor],
) -> JetTuple:
    """Apply pointwise ``op`` to two collapsed jets of equal order.

    Broadcasts both operands' coefficients up to the result primal's shape via
    :func:`_broadcast_coeffs` before ``op``, so different-rank operands align
    over their primal dims without colliding the leading direction dim ``R``.

    ``op`` must be elementwise (it broadcasts over leading dims); the product
    rules use per-direction ``vmap`` instead (see :func:`_collapsed_leibniz`)
    for ops like ``conv`` that cannot.
    """
    _jet_order(self, other)  # validates K-consistency, raises on mismatch
    primal = op(self[0], other[0])
    s_coeffs = _broadcast_coeffs(self, primal)
    o_coeffs = _broadcast_coeffs(other, primal)
    coeffs = (op(s, o) for s, o in zip(s_coeffs, o_coeffs))
    return _cjet((primal, *coeffs))


def _broadcast_coeffs(self: JetTuple, primal: Tensor) -> list[Tensor]:
    """Broadcast a collapsed jet's coefficients up to ``primal``'s shape.

    Also used by :func:`_collapsed_pointwise` to align two jets before an
    elementwise ``op``. The batched coefficients (orders ``1..K-1``) carry a
    leading direction dim ``R``, so the broadcast is R-aware: insert size-1 dims
    after ``R``, then expand. PyTorch broadcasting would instead left-pad at the
    front, shifting ``R`` so it collides with a primal dim when ranks differ.
    The collapsed ``K``-th coefficient and the primal carry no ``R`` and
    broadcast normally. No-op when already shaped.
    """
    K = len(self) - 1
    out = []
    for k in range(1, K + 1):
        c = self[k]
        if k < K:  # batched (R, *S) -> (R, *primal.shape)
            target = (c.shape[0], *primal.shape)
            if c.shape != target:
                pad = primal.ndim - (c.ndim - 1)
                reshaped = c.reshape(c.shape[0], *([1] * pad), *c.shape[1:])
                c = reshaped.broadcast_to(target)
        elif c.shape != primal.shape:  # collapsed K-th (no R)
            c = c.broadcast_to(primal.shape)
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Collapsed Leibniz rule (for products: mul, mm)
# ---------------------------------------------------------------------------


def _collapsed_leibniz(
    self: JetTuple,
    other: JetTuple,
    binary_op: Callable[[Tensor, Tensor], Tensor],
) -> tuple[Tensor, ...]:
    """Leibniz product rule with collapsed K-th coefficient (orders 1..K).

    Returns only coefficients 1..K; the caller handles the order-0 primal
    explicitly (mirroring :func:`jet.operations._leibniz` — skipping k=0 here
    avoids a wasted ``binary_op(self[0], other[0])`` node in the captured
    graph when the caller is ``cjet_addmm`` and supplies its own
    ``addmm``-based primal).

    For orders 1..K-1: standard Leibniz.
    For order K: linear terms (using collapsed coefficients) +
                 nonlinear terms (using batched coefficients, summed over R).
    ``K`` is inferred as ``len(self) - 1``; lengths are checked.

    Raises:
        ValueError: If ``self`` and ``other`` have different lengths.
    """
    if len(self) != len(other):
        raise ValueError(
            f"_collapsed_leibniz: operands must share the same derivative "
            f"order; got lengths {len(self)} and {len(other)}"
        )
    K = len(self) - 1

    def apply(a, b, a_batched, b_batched):
        # ``binary_op`` here is a general bilinear op (elementwise ``mul``,
        # ``matmul``, or ``conv``), not necessarily one that broadcasts over a
        # leading batch dim -- ``conv`` in particular cannot -- so the direction
        # dim ``R`` is mapped explicitly with ``vmap`` (per direction), which
        # also keeps ``R`` aligned when the operands' primal ranks differ. At
        # least one operand is always batched here (the two coefficient indices
        # sum to ``k >= 1``), so ``in_dims`` is never all-``None``.
        in_dims = (0 if a_batched else None, 0 if b_batched else None)
        return vmap(binary_op, in_dims=in_dims)(a, b)

    coeffs = ()
    for k in range(1, K + 1):
        if k < K:
            term = None
            for j in range(k + 1):
                term_j = comb(k, j, exact=True) * apply(
                    self[j], other[k - j], j >= 1, (k - j) >= 1
                )
                term = term_j if term is None else term + term_j
            coeffs += (term,)
        else:
            linear = binary_op(self[0], other[K]) + binary_op(self[K], other[0])
            if K >= 2:
                nonlinear = None
                for j in range(1, K):
                    # Sum out the direction dim R per term so the accumulator (and
                    # downstream traced-graph tensors) stay small.
                    term_j = comb(K, j, exact=True) * apply(
                        self[j], other[K - j], True, True
                    ).sum(0)
                    nonlinear = term_j if nonlinear is None else nonlinear + term_j
                coeffs += (linear + nonlinear,)
            else:
                coeffs += (linear,)
    return coeffs


def _apply_bilinear(
    op: Callable[[Tensor, Tensor], Tensor],
    self: Tensor | JetTuple,
    other: Tensor | JetTuple,
) -> Tensor | JetTuple:
    """Lift a bilinear tensor ``op`` to operands each of which may be jet or constant.

    Collapsed mirror of :func:`jet.operations._apply_bilinear`: both-jet uses the
    collapsed Leibniz rule (:func:`_collapsed_leibniz`, which sums the nonlinear
    terms over the direction dim ``R``); one-sided maps coefficient-wise via the
    collapsed :func:`_apply_linear` (vmapping the batched coefficients); neither
    falls back to a plain ``op`` on two constants. Only valid for **bilinear**
    (product-like) ops -- ``add`` / ``sub`` use the additive rule.

    Args:
        op: A bilinear function of two coefficient tensors.
        self: The first operand; a jet or a constant ``Tensor``.
        other: The second operand; a jet or a constant ``Tensor``.

    Returns:
        The collapsed jet of ``op(self, other)``, or a plain constant when both
        operands are constants.
    """
    self_is = isinstance(self, JetTuple)
    other_is = isinstance(other, JetTuple)
    if self_is and other_is:
        primal = op(self[0], other[0])
        return _cjet((primal, *_collapsed_leibniz(self, other, op)))
    if self_is:
        return _apply_linear(self, lambda c: op(c, other))
    if other_is:
        return _apply_linear(other, lambda c: op(self, c))
    return op(self, other)


# ---------------------------------------------------------------------------
# Elementwise nonlinear (shared derivative helpers + collapsed Faà di Bruno)
# ---------------------------------------------------------------------------


def cjet_exp(self: JetTuple) -> JetTuple:
    """Collapsed ``aten.exp`` (bound for reuse in ``cjet_log_softmax``)."""
    return _elementwise(self, _exp_derivatives)


def cjet_log(self: JetTuple) -> JetTuple:
    """Collapsed ``aten.log`` (bound for reuse in ``cjet_log_softmax``)."""
    return _elementwise(self, _log_derivatives)


def cjet_pow(self: JetTuple, exponent: float | int) -> JetTuple:
    """Collapsed jet rule for ``aten.pow``."""
    assert isinstance(exponent, (float, int))
    return _elementwise(self, lambda x, k: _pow_derivatives(x, exponent, k))


# ---------------------------------------------------------------------------
# Arithmetic (add, sub are linear; mul uses collapsed Leibniz)
# ---------------------------------------------------------------------------


def cjet_add(
    self: Tensor | JetTuple | float | int,
    other: Tensor | JetTuple | float | int,
) -> Tensor | JetTuple | float | int:
    """Collapsed jet rule for ``aten.add``."""
    self_is = isinstance(self, JetTuple)
    other_is = isinstance(other, JetTuple)
    if self_is and other_is:
        return _collapsed_pointwise(self, other, add)
    if self_is:
        primal = self[0] + other
        return _cjet((primal, *_broadcast_coeffs(self, primal)))
    if other_is:
        primal = other[0] + self
        return _cjet((primal, *_broadcast_coeffs(other, primal)))
    return self + other


def cjet_sub(
    self: Tensor | JetTuple | float | int,
    other: Tensor | JetTuple | float | int,
) -> Tensor | JetTuple | float | int:
    """Collapsed jet rule for ``aten.sub``."""
    self_is = isinstance(self, JetTuple)
    other_is = isinstance(other, JetTuple)
    if self_is and other_is:
        return _collapsed_pointwise(self, other, sub)
    if self_is:
        primal = self[0] - other
        return _cjet((primal, *_broadcast_coeffs(self, primal)))
    if other_is:
        primal = self - other[0]
        return _cjet((primal, *(-c for c in _broadcast_coeffs(other, primal))))
    return self - other


def cjet_mul(self: Tensor | JetTuple, other: Tensor | JetTuple) -> JetTuple:
    """Collapsed jet rule for ``aten.mul``."""
    return _apply_bilinear(lambda a, b: a * b, self, other)


# ---------------------------------------------------------------------------
# Matrix operations (vmap for one-sided, collapsed Leibniz for both-sided)
# ---------------------------------------------------------------------------


def cjet_mm(self: Tensor | JetTuple, mat2: Tensor | JetTuple) -> JetTuple:
    """Collapsed jet rule for ``aten.mm``."""
    return _apply_bilinear(matmul, self, mat2)


def cjet_addmm(
    self: Tensor | JetTuple,
    mat1: Tensor | JetTuple,
    mat2: Tensor | JetTuple,
) -> JetTuple:
    """Collapsed ``aten.addmm`` (supports a Taylor-expanded bias). See ``_addmm_impl``."""
    return _addmm_impl(self, mat1, mat2, cjet_add, cjet_mm)


def cjet_convolution(
    input: Tensor | JetTuple,
    weight: Tensor | JetTuple,
    bias: Tensor | JetTuple | None,
    *conv_args: object,
) -> JetTuple:
    """Collapsed jet rule for ``aten.convolution``.

    See :func:`jet.operations.jet_convolution`: composes the bilinear bias-free
    convolution with the affine bias addition, ``cjet_add(bias, conv(input,
    weight, None))``. The 1-D bias is reshaped to broadcast over the output's
    batch and spatial dims (channel is dim 1); the direction-dim ``R`` alignment
    of the batched coefficients is handled by :func:`cjet_add`.
    """

    def cv(a: Tensor, b: Tensor) -> Tensor:
        """Bias-free convolution -- the bilinear core of ``aten.convolution``."""
        return ops.aten.convolution.default(a, b, None, *conv_args)

    product = _apply_bilinear(cv, input, weight)
    if bias is None:
        return product
    # conv preserves rank, so the output ndim is the input ndim.
    ndim = (input[0] if isinstance(input, JetTuple) else input).ndim
    return cjet_add(_align_conv_bias(bias, ndim), product)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def _reduce_loss(loss: JetTuple, reduction: int) -> JetTuple:
    """Apply a loss reduction coefficient-wise (collapsed mode).

    ``reduction`` follows ATen's enum -- ``0`` (none, identity), ``1`` (mean),
    ``2`` (sum). Reductions are linear; the collapsed ``_apply_linear`` vmaps
    over the leading direction dim ``R`` for the batched coefficients
    ``c_1..c_{K-1}`` and reduces the primal and collapsed slot ``c_K`` directly.

    Args:
        loss: The per-element loss and its Taylor coefficients.
        reduction: The ATen reduction enum (``0``/``1``/``2``).

    Returns:
        The reduced loss and its Taylor coefficients.

    Raises:
        ValueError: If ``reduction`` is not ``0``, ``1``, or ``2``.
    """
    if reduction == 0:  # 'none'
        return loss
    if reduction == 1:  # 'mean'
        return _apply_linear(loss, lambda c: c.mean())
    if reduction == 2:  # 'sum'
        return _apply_linear(loss, lambda c: c.sum())
    raise ValueError(f"Unsupported reduction {reduction}; expected 0, 1, or 2.")


def cjet_mse_loss(
    self: Tensor | JetTuple,
    target: Tensor | JetTuple,
    reduction: int = 1,
) -> JetTuple:
    """Collapsed jet rule for ``aten.mse_loss(self, target, reduction)``.

    Computes ``reduce((self - target) ** 2)`` by composing the ``sub`` and
    ``pow`` rules. ``target`` is typically a constant tensor (a label), but a
    Taylor-expanded ``target`` is supported too.

    Args:
        self: The prediction and its Taylor coefficients.
        target: The target and its Taylor coefficients, or a constant tensor.
        reduction: The ATen reduction enum -- ``0`` (none), ``1`` (mean, the
            default), ``2`` (sum).

    Returns:
        The value and its Taylor coefficients.
    """
    squared_error = cjet_pow(cjet_sub(self, target), 2)
    return _reduce_loss(squared_error, reduction)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def cjet_max_pool2d_with_indices(
    input: JetTuple, *pool_args: object
) -> tuple[JetTuple, Tensor]:
    """Collapsed jet rule for ``aten.max_pool2d_with_indices``."""
    values0, indices = ops.aten.max_pool2d_with_indices.default(input[0], *pool_args)
    coeffs = _apply_linear_coeffs(input, lambda c: _gather_at_indices(c, indices))
    return _cjet((values0, *coeffs)), indices


def cjet_cat(tensors: list[Tensor | JetTuple], dim: int = 0) -> JetTuple:
    """Collapsed jet rule for ``aten.cat(tensors, dim)``.

    Concatenation is linear -- see :func:`jet.operations.jet_cat`. The batched
    coefficients (orders ``1..K-1``) carry a leading direction dim ``R``, so a
    non-negative concat ``dim`` shifts by one there, and constant operands
    contribute ``(R, *shape)`` zeros to match the batched jet coefficients.
    """
    K = _jet_order(*tensors)
    R = next(t for t in tensors if isinstance(t, JetTuple))[1].shape[0]

    def part(t: object, k: int, batched: bool) -> Tensor:
        if isinstance(t, JetTuple):
            return t[k]
        if k == 0:
            return t
        return t.new_zeros(R, *t.shape) if batched else zeros_like(t)

    out = []
    for k in range(K + 1):
        batched = 0 < k < K
        d = dim + 1 if (batched and dim >= 0) else dim
        out.append(cat([part(t, k, batched) for t in tensors], d))
    return _cjet(tuple(out))


def cjet_log_softmax(self: JetTuple, dim: int, half_to_float: bool = False) -> JetTuple:
    """Collapsed ``aten._log_softmax``. See :func:`jet.operations._log_softmax_impl`."""
    return _log_softmax_impl(self, dim, cjet_sub, cjet_exp, cjet_sum, cjet_log)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def cjet_nll_loss_forward(
    self: JetTuple,
    target: Tensor,
    weight: Tensor | None,
    reduction: int,
    ignore_index: int,
) -> tuple[JetTuple, Tensor]:
    """Collapsed jet rule for ``aten.nll_loss_forward``.

    Same linear application as :func:`jet.operations.jet_nll_loss_forward`.

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
            "cjet_nll_loss_forward does not support a Taylor-expanded target or "
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
    return _cjet((output, *coeffs)), total_weight


# ---------------------------------------------------------------------------
# Batch norm
# ---------------------------------------------------------------------------


def cjet_native_batch_norm(
    input: Tensor | JetTuple,
    weight: Tensor | JetTuple | None,
    bias: Tensor | JetTuple | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[JetTuple, Tensor, Tensor]:
    """Collapsed ``aten.native_batch_norm`` (eval). See ``_native_batch_norm_impl``."""
    return _native_batch_norm_impl(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        cjet_sub,
        cjet_mul,
        cjet_view,
        cjet_add,
    )


# ---------------------------------------------------------------------------
# Rule-building factories (registered in :mod:`jet._rules`)
# ---------------------------------------------------------------------------
#
# Mirror :func:`jet.operations._deflinear` / :func:`jet.operations._defzero`,
# but use the collapsed ``_apply_linear`` (which vmaps over the leading
# direction dim for batched coefficients) and respect the collapsed per-slot
# shape contract for the zero rule. Registration into ``RULES`` lives in
# :mod:`jet._rules`.


def _deflinear(prim: Callable) -> Callable:
    """Build a collapsed linear jet rule. See :func:`jet.operations._make_linear_rule`.

    Uses the collapsed ``_apply_linear``, which vmaps over the leading ``R`` dim
    for batched coefficients ``c_1..c_{K-1}`` and applies ``prim`` directly to
    the primal and the collapsed slot ``c_K``.
    """
    return _make_linear_rule(prim, _apply_linear)


def _defzero(prim: Callable) -> Callable:
    """Build a constant-output jet rule (collapsed mode).

    ``prim`` is applied to the primal; coefficients are filled with zero
    tensors that take their shape from each input coefficient slot (to
    preserve the per-slot shape contract — ``(R, *S)`` for ``c_1..c_{K-1}``,
    ``S`` for ``c_K``) and their dtype / device / layout from ``primal_out``
    so any ``dtype=`` / ``device=`` etc. kwargs passed to ``prim`` propagate
    to the coefficients too.
    """

    def rule(self: JetTuple, *args, **kwargs) -> JetTuple:
        primal_out = prim(self[0], *args, **kwargs)
        coeffs = [primal_out.new_zeros(c.shape) for c in self[1:]]
        return _cjet([primal_out, *coeffs])

    return rule


# Bound to names so composite rules can reuse them: ``cjet_sum`` in
# ``cjet_log_softmax``; ``cjet_view`` to reshape a possibly-constant operand.
cjet_sum = _deflinear(ops.aten.sum.dim_IntList)
cjet_view = _deflinear(ops.aten.view.default)
