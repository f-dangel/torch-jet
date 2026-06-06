"""Implementation of AD primitives in collapsed Taylor-mode arithmetic.

Collapsed Taylor mode propagates a single "collapsed jet" with mixed shapes:

  - Coefficient 0 (primal): shape (...)
  - Coefficients 1..K-1: shape (R, ...) -- batched over R directions
  - Coefficient K: shape (...) -- already collapsed (summed over directions)

At each nonlinear operation, the K-th output coefficient is computed as:
  out_K = LINEAR_TERM(in_K_collapsed) + NONLINEAR_TERMS(in_1..K-1).sum(0)
"""

from operator import add, sub
from typing import Callable

from scipy.special import comb
from torch import Tensor, cat, matmul, ops, zeros_like
from torch.func import vmap
from torch.utils._pytree import register_pytree_node

from jet.operations import (
    _align_conv_bias,
    _bn_channel_view,
    _cos_derivatives,
    _exp_derivatives,
    _faa_di_bruno,
    _gather_at_indices,
    _log_derivatives,
    _order,
    _pow_derivatives,
    _relu_derivatives,
    _sigmoid_derivatives,
    _sin_derivatives,
    _tanh_derivatives,
)

# ---------------------------------------------------------------------------
# CollapsedJetTuple
# ---------------------------------------------------------------------------


class CollapsedJetTuple(tuple):
    """JetTuple where the last coefficient is collapsed (summed over directions)."""


register_pytree_node(
    CollapsedJetTuple,
    flatten_fn=lambda x: (list(x), None),
    unflatten_fn=lambda values, context: CollapsedJetTuple(values),
)


def _cjet_order(*args: Tensor | CollapsedJetTuple | float | int) -> int:
    """Infer ``K`` from all ``CollapsedJetTuple`` positional args.

    Thin wrapper around :func:`jet.operations._order` that pre-binds the jet
    type to ``CollapsedJetTuple``.
    """
    return _order(args, CollapsedJetTuple)


# ---------------------------------------------------------------------------
# Helpers: apply a linear op with vmap for batched coefficients
# ---------------------------------------------------------------------------


def _apply_linear(
    jet: CollapsedJetTuple, op: Callable[[Tensor], Tensor]
) -> CollapsedJetTuple:
    """Apply a linear *op* to every entry of *jet*, vmapping batched ones."""
    K = len(jet) - 1
    results = [op(jet[0])]
    vop = vmap(op)
    for k in range(1, K + 1):
        results.append(vop(jet[k]) if k < K else op(jet[k]))
    return CollapsedJetTuple(results)


def _apply_linear_coeffs(
    jet: CollapsedJetTuple, op: Callable[[Tensor], Tensor]
) -> tuple[Tensor, ...]:
    """Apply *op* to coefficients 1..K only, vmapping batched ones."""
    K = len(jet) - 1
    vop = vmap(op)
    return tuple(vop(jet[k]) if k < K else op(jet[k]) for k in range(1, K + 1))


def _collapsed_pointwise(
    self: CollapsedJetTuple,
    other: CollapsedJetTuple,
    op: Callable[[Tensor, Tensor], Tensor],
) -> CollapsedJetTuple:
    """Apply pointwise ``op`` to two collapsed jets of equal order.

    Broadcasts both operands' coefficients up to the result primal's shape via
    :func:`_broadcast_coeffs` before ``op``, so different-rank operands align
    over their primal dims without colliding the leading direction dim ``R``.

    ``op`` must be elementwise (it broadcasts over leading dims); the product
    rules use per-direction ``vmap`` instead (see :func:`_collapsed_leibniz`)
    for ops like ``conv`` that cannot.
    """
    _cjet_order(self, other)  # validates K-consistency, raises on mismatch
    primal = op(self[0], other[0])
    s_coeffs = _broadcast_coeffs(self, primal)
    o_coeffs = _broadcast_coeffs(other, primal)
    coeffs = (op(s, o) for s, o in zip(s_coeffs, o_coeffs))
    return CollapsedJetTuple((primal, *coeffs))


def _broadcast_coeffs(self: CollapsedJetTuple, primal: Tensor) -> list[Tensor]:
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
    self: CollapsedJetTuple,
    other: CollapsedJetTuple,
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
    self: Tensor | CollapsedJetTuple,
    other: Tensor | CollapsedJetTuple,
) -> Tensor | CollapsedJetTuple:
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
    self_is = isinstance(self, CollapsedJetTuple)
    other_is = isinstance(other, CollapsedJetTuple)
    if self_is and other_is:
        primal = op(self[0], other[0])
        return CollapsedJetTuple((primal, *_collapsed_leibniz(self, other, op)))
    if self_is:
        return _apply_linear(self, lambda c: op(c, other))
    if other_is:
        return _apply_linear(other, lambda c: op(self, c))
    return op(self, other)


# ---------------------------------------------------------------------------
# Elementwise nonlinear (shared derivative helpers + collapsed Faà di Bruno)
# ---------------------------------------------------------------------------


def _cjet_elementwise(
    self: CollapsedJetTuple,
    deriv_fn: Callable[[Tensor, int], dict[int, Tensor]],
) -> CollapsedJetTuple:
    """Generic collapsed elementwise using shared helpers."""
    K = _cjet_order(self)
    self0, vs = self[0], self[1:]
    dn = deriv_fn(self0, K)
    vs_out = _faa_di_bruno(vs, dn, collapsed=True)
    return CollapsedJetTuple((dn[0], *vs_out))


def cjet_sin(self: CollapsedJetTuple) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.sin``."""
    return _cjet_elementwise(self, _sin_derivatives)


def cjet_cos(self: CollapsedJetTuple) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.cos``."""
    return _cjet_elementwise(self, _cos_derivatives)


def cjet_tanh(self: CollapsedJetTuple) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.tanh``."""
    return _cjet_elementwise(self, _tanh_derivatives)


def cjet_sigmoid(self: CollapsedJetTuple) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.sigmoid``."""
    return _cjet_elementwise(self, _sigmoid_derivatives)


def cjet_relu(self: CollapsedJetTuple) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.relu``."""
    return _cjet_elementwise(self, _relu_derivatives)


def cjet_exp(self: CollapsedJetTuple) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.exp``."""
    return _cjet_elementwise(self, _exp_derivatives)


def cjet_log(self: CollapsedJetTuple) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.log``."""
    return _cjet_elementwise(self, _log_derivatives)


def cjet_pow(self: CollapsedJetTuple, exponent: float | int) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.pow``."""
    assert isinstance(exponent, (float, int))
    self0, vs = self[0], self[1:]
    dpow = _pow_derivatives(self0, exponent, _cjet_order(self))
    vs_out = _faa_di_bruno(vs, dpow, collapsed=True)
    return CollapsedJetTuple((dpow[0], *vs_out))


# ---------------------------------------------------------------------------
# Arithmetic (add, sub are linear; mul uses collapsed Leibniz)
# ---------------------------------------------------------------------------


def cjet_add(
    self: Tensor | CollapsedJetTuple | float | int,
    other: Tensor | CollapsedJetTuple | float | int,
) -> Tensor | CollapsedJetTuple | float | int:
    """Collapsed jet rule for ``aten.add``."""
    self_is = isinstance(self, CollapsedJetTuple)
    other_is = isinstance(other, CollapsedJetTuple)
    if self_is and other_is:
        return _collapsed_pointwise(self, other, add)
    if self_is:
        primal = self[0] + other
        return CollapsedJetTuple((primal, *_broadcast_coeffs(self, primal)))
    if other_is:
        primal = other[0] + self
        return CollapsedJetTuple((primal, *_broadcast_coeffs(other, primal)))
    return self + other


def cjet_sub(
    self: Tensor | CollapsedJetTuple | float | int,
    other: Tensor | CollapsedJetTuple | float | int,
) -> Tensor | CollapsedJetTuple | float | int:
    """Collapsed jet rule for ``aten.sub``."""
    self_is = isinstance(self, CollapsedJetTuple)
    other_is = isinstance(other, CollapsedJetTuple)
    if self_is and other_is:
        return _collapsed_pointwise(self, other, sub)
    if self_is:
        primal = self[0] - other
        return CollapsedJetTuple((primal, *_broadcast_coeffs(self, primal)))
    if other_is:
        primal = self - other[0]
        return CollapsedJetTuple(
            (primal, *(-c for c in _broadcast_coeffs(other, primal)))
        )
    return self - other


def cjet_mul(
    self: Tensor | CollapsedJetTuple,
    other: Tensor | CollapsedJetTuple,
) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.mul``."""
    return _apply_bilinear(lambda a, b: a * b, self, other)


# ---------------------------------------------------------------------------
# Matrix operations (vmap for one-sided, collapsed Leibniz for both-sided)
# ---------------------------------------------------------------------------


def cjet_mm(
    self: Tensor | CollapsedJetTuple, mat2: Tensor | CollapsedJetTuple
) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.mm``."""
    return _apply_bilinear(matmul, self, mat2)


def cjet_addmm(
    self: Tensor | CollapsedJetTuple,
    mat1: Tensor | CollapsedJetTuple,
    mat2: Tensor | CollapsedJetTuple,
) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.addmm`` (supports a Taylor-expanded bias).

    See :func:`jet.operations.jet_addmm`: composes the matrix-product rule with
    the affine bias addition, ``cjet_add(self, _apply_bilinear(matmul, mat1, mat2))``.
    """
    return cjet_add(self, _apply_bilinear(matmul, mat1, mat2))


def cjet_convolution(
    input: Tensor | CollapsedJetTuple,
    weight: Tensor | CollapsedJetTuple,
    bias: Tensor | CollapsedJetTuple | None,
    *conv_args: object,
) -> CollapsedJetTuple:
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
    ndim = (input[0] if isinstance(input, CollapsedJetTuple) else input).ndim
    return cjet_add(_align_conv_bias(bias, ndim), product)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def _reduce_loss(loss: CollapsedJetTuple, reduction: int) -> CollapsedJetTuple:
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
    self: Tensor | CollapsedJetTuple,
    target: Tensor | CollapsedJetTuple,
    reduction: int = 1,
) -> CollapsedJetTuple:
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
# COLLAPSED_MAPPING
# ---------------------------------------------------------------------------


def cjet_max_pool2d_with_indices(
    input: CollapsedJetTuple, *pool_args: object
) -> tuple[CollapsedJetTuple, Tensor]:
    """Collapsed jet rule for ``aten.max_pool2d_with_indices``."""
    values0, indices = ops.aten.max_pool2d_with_indices.default(input[0], *pool_args)
    coeffs = _apply_linear_coeffs(input, lambda c: _gather_at_indices(c, indices))
    return CollapsedJetTuple((values0, *coeffs)), indices


def cjet_max_pool2d(input: CollapsedJetTuple, *pool_args: object) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.max_pool2d`` (values only; e.g. MPS)."""
    jet, _ = cjet_max_pool2d_with_indices(input, *pool_args)
    return jet


def cjet_cat(
    tensors: list[Tensor | CollapsedJetTuple], dim: int = 0
) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.cat(tensors, dim)``.

    Concatenation is linear -- see :func:`jet.operations.jet_cat`. The batched
    coefficients (orders ``1..K-1``) carry a leading direction dim ``R``, so a
    non-negative concat ``dim`` shifts by one there, and constant operands
    contribute ``(R, *shape)`` zeros to match the batched jet coefficients.
    """
    K = _cjet_order(*tensors)
    R = next(t for t in tensors if isinstance(t, CollapsedJetTuple))[1].shape[0]

    def part(t: object, k: int, batched: bool) -> Tensor:
        if isinstance(t, CollapsedJetTuple):
            return t[k]
        if k == 0:
            return t
        return t.new_zeros(R, *t.shape) if batched else zeros_like(t)

    out = []
    for k in range(K + 1):
        batched = 0 < k < K
        d = dim + 1 if (batched and dim >= 0) else dim
        out.append(cat([part(t, k, batched) for t in tensors], d))
    return CollapsedJetTuple(tuple(out))


def cjet_log_softmax(
    self: CollapsedJetTuple, dim: int, half_to_float: bool = False
) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten._log_softmax(self, dim, half_to_float)``.

    Args:
        self: The logits and their Taylor coefficients.
        dim: The dimension along which to normalize.
        half_to_float: Whether inputs were promoted from half precision.
            Accepted for ATen-signature compatibility; does not affect the
            float32/float64 paths.

    Returns:
        The value and its Taylor coefficients.
    """
    shift = self[0].amax(dim, keepdim=True)
    shifted = cjet_sub(self, shift)
    exp_jet = cjet_exp(shifted)
    sum_exp = cjet_sum(exp_jet, dim, keepdim=True)
    log_sum_exp = cjet_log(sum_exp)
    return cjet_sub(shifted, log_sum_exp)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def cjet_nll_loss_forward(
    self: CollapsedJetTuple,
    target: Tensor,
    weight: Tensor | None,
    reduction: int,
    ignore_index: int,
) -> tuple[CollapsedJetTuple, Tensor]:
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
    if isinstance(target, CollapsedJetTuple) or isinstance(weight, CollapsedJetTuple):
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
    return CollapsedJetTuple((output, *coeffs)), total_weight


# ---------------------------------------------------------------------------
# Batch norm
# ---------------------------------------------------------------------------


def cjet_native_batch_norm(
    input: Tensor | CollapsedJetTuple,
    weight: Tensor | CollapsedJetTuple | None,
    bias: Tensor | CollapsedJetTuple | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[CollapsedJetTuple, Tensor, Tensor]:
    """Collapsed jet rule for ``aten.native_batch_norm`` (eval mode).

    Mirrors :func:`jet.operations.jet_native_batch_norm` with the collapsed
    arithmetic helpers (sharing only the pure-tensor :func:`_bn_channel_view`):
    normalizes each channel with the frozen running statistics,
    ``(input - running_mean) / sqrt(running_var + eps) * weight + bias``. Any of
    ``input`` / ``weight`` / ``bias`` may be a jet, a constant, or
    (``weight`` / ``bias``) ``None``; any input rank is supported. Training mode
    is deferred until PyTorch fixes its fused ``native_batch_norm``'s incorrect
    higher-order autograd in training (pytorch/pytorch#186256); eval mode without
    running statistics is likewise unsupported.
    """
    if training:
        raise NotImplementedError(
            "Taylor-mode native_batch_norm supports eval mode only. Training mode "
            "is deferred until PyTorch fixes the fused op's incorrect higher-order "
            "autograd in training (pytorch/pytorch#186256)."
        )
    if running_mean is None or running_var is None:
        raise NotImplementedError(
            "Taylor-mode native_batch_norm requires running statistics in eval "
            "mode; missing running_mean/running_var falls back to batch "
            "statistics, which is not yet implemented."
        )

    primal = input[0] if isinstance(input, CollapsedJetTuple) else input
    shape = _bn_channel_view(primal)
    rstd = (running_var + eps).rsqrt()
    out = cjet_sub(input, cjet_view(running_mean, shape))
    out = cjet_mul(out, cjet_view(rstd, shape))
    if weight is not None:
        out = cjet_mul(out, cjet_view(weight, shape))
    if bias is not None:
        out = cjet_add(out, cjet_view(bias, shape))
    empty = primal.new_empty(0)
    return out, empty, empty


# ---------------------------------------------------------------------------
# COLLAPSED_MAPPING
# ---------------------------------------------------------------------------


COLLAPSED_MAPPING: dict = {
    # Elementwise nonlinear
    ops.aten.sin.default: cjet_sin,
    ops.aten.cos.default: cjet_cos,
    ops.aten.tanh.default: cjet_tanh,
    ops.aten.sigmoid.default: cjet_sigmoid,
    ops.aten.relu.default: cjet_relu,
    ops.aten.exp.default: cjet_exp,
    ops.aten.log.default: cjet_log,
    # Power
    ops.aten.pow.Tensor_Scalar: cjet_pow,
    # Arithmetic
    ops.aten.add.Tensor: cjet_add,
    ops.aten.sub.Tensor: cjet_sub,
    ops.aten.mul.Tensor: cjet_mul,
    # Matrix ops
    ops.aten.mm.default: cjet_mm,
    ops.aten.addmm.default: cjet_addmm,
    # Convolution (bilinear in input/weight; bias is the affine term)
    ops.aten.convolution.default: cjet_convolution,
    # Pooling (piecewise linear: gather coefficients at the primal's arg-max)
    ops.aten.max_pool2d_with_indices.default: cjet_max_pool2d_with_indices,
    ops.aten.max_pool2d.default: cjet_max_pool2d,
    # Concatenation (linear; jets nested in the operand list)
    ops.aten.cat.default: cjet_cat,
    # Loss functions
    ops.aten.mse_loss.default: cjet_mse_loss,
    ops.aten.nll_loss_forward.default: cjet_nll_loss_forward,
    # Normalization
    ops.aten._log_softmax.default: cjet_log_softmax,
    # Batch norm (affine in eval; composed batch statistics in training)
    ops.aten.native_batch_norm.default: cjet_native_batch_norm,
}


# --- JAX-style helpers: bulk-register categories of ops ---
#
# Mirrors :func:`jet.operations.deflinear` / :func:`jet.operations.defzero`,
# but uses the collapsed ``_apply_linear`` (which vmaps over the leading
# direction dim for batched coefficients) and respects the collapsed
# per-slot shape contract for ``defzero``.


def deflinear(prim: Callable) -> Callable:
    """Register ``prim`` as a linear op (collapsed mode).

    Collapsed ``_apply_linear`` vmaps over the leading ``R`` dim for batched
    coefficients ``c_1..c_{K-1}`` and applies ``prim`` directly to the primal
    and the collapsed slot ``c_K``. Returns the registered rule so it can also
    be bound to a name and reused inside composite rules (e.g. ``cjet_sum`` in
    ``cjet_log_softmax``, or ``cjet_view`` to reshape a possibly-constant
    operand). The rule is total over constants: a non-jet argument is passed
    straight to ``prim``.
    """

    def rule(
        self: Tensor | CollapsedJetTuple, *args, **kwargs
    ) -> Tensor | CollapsedJetTuple:
        if not isinstance(self, CollapsedJetTuple):
            return prim(self, *args, **kwargs)
        return _apply_linear(self, lambda c: prim(c, *args, **kwargs))

    COLLAPSED_MAPPING[prim] = rule
    return rule


def defzero(prim: Callable) -> None:
    """Register ``prim`` as a constant-output op (collapsed mode).

    ``prim`` is applied to the primal; coefficients are filled with zero
    tensors that take their shape from each input coefficient slot (to
    preserve the per-slot shape contract — ``(R, *S)`` for ``c_1..c_{K-1}``,
    ``S`` for ``c_K``) and their dtype / device / layout from ``primal_out``
    so any ``dtype=`` / ``device=`` etc. kwargs passed to ``prim`` propagate
    to the coefficients too.
    """

    def rule(self: CollapsedJetTuple, *args, **kwargs) -> CollapsedJetTuple:
        primal_out = prim(self[0], *args, **kwargs)
        coeffs = [primal_out.new_zeros(c.shape) for c in self[1:]]
        return CollapsedJetTuple([primal_out, *coeffs])

    COLLAPSED_MAPPING[prim] = rule


# Linear ops (pointwise-linear, shape-only, reductions).
for _prim in (
    ops.aten.neg.default,
    ops.aten.div.Scalar,
    ops.aten.t.default,
    ops.aten._unsafe_view.default,
    ops.aten.unsqueeze.default,
    ops.aten.squeeze.dim,
    ops.aten.squeeze.dims,
    ops.aten.sum.default,
    ops.aten._adaptive_avg_pool2d.default,
    ops.aten.avg_pool2d.default,
    ops.aten.mean.default,
    ops.aten.mean.dim,
):
    deflinear(_prim)

# Bound to names so composite rules can reuse them: ``cjet_sum`` in
# ``cjet_log_softmax``; ``cjet_view`` to reshape a possibly-constant operand.
cjet_sum = deflinear(ops.aten.sum.dim_IntList)
cjet_view = deflinear(ops.aten.view.default)

# Constant-output ops.
for _prim in (ops.aten.zeros_like.default,):
    defzero(_prim)
