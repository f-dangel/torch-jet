"""Implementation of AD primitives in collapsed Taylor-mode arithmetic.

Collapsed Taylor mode propagates a single "collapsed jet" with mixed shapes:

  - Coefficient 0 (primal): shape (...)
  - Coefficients 1..K-1: shape (R, ...) -- batched over R directions
  - Coefficient K: shape (...) -- already collapsed (summed over directions)

At each nonlinear operation, the K-th output coefficient is computed as:
  out_K = LINEAR_TERM(in_K_collapsed) + NONLINEAR_TERMS(in_1..K-1).sum(0)
"""

from typing import Callable

from scipy.special import comb
from torch import Tensor, addmm, matmul, mm, ops
from torch.func import vmap
from torch.utils._pytree import register_pytree_node

from jet.operations import (
    _cos_derivatives,
    _exp_derivatives,
    _faa_di_bruno,
    _gather_at_indices,
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
        # Each coefficient ``c_k`` for ``k >= 1`` carries a leading direction
        # dim ``R``; ``c_0`` does not. Plain ``binary_op(a, b)`` would
        # right-align via PyTorch broadcasting, which collides ``R`` against
        # a middle primal dim of the other operand whenever ``a``'s and
        # ``b``'s primal shapes have different ranks (e.g. ``mul`` of operands
        # with primal shapes ``(3,)`` and ``(2, 3)``, where ``R`` coincides
        # with the size-2 primal dim of the other). ``vmap`` over ``R`` with
        # the right ``in_dims`` aligns ``R`` per-direction explicitly and
        # broadcasts the suffixes correctly.
        if not (a_batched or b_batched):
            return binary_op(a, b)
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


# ---------------------------------------------------------------------------
# Elementwise nonlinear (shared derivative helpers + collapsed Faà di Bruno)
# ---------------------------------------------------------------------------


def _cjet_elementwise(
    self: CollapsedJetTuple,
    deriv_fn: Callable[[Tensor, int], tuple[Tensor, dict[int, Tensor]]],
) -> CollapsedJetTuple:
    """Generic collapsed elementwise using shared helpers."""
    K = _cjet_order(self)
    self0, vs = self[0], self[1:]
    primal, dn = deriv_fn(self0, K)
    vs_out = _faa_di_bruno(vs, dn, collapsed=True)
    return CollapsedJetTuple((primal, *vs_out))


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


def cjet_pow(self: CollapsedJetTuple, exponent: float | int) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.pow``."""
    assert isinstance(exponent, (float, int))
    self0, vs = self[0], self[1:]
    primal, dpow = _pow_derivatives(self0, exponent, _cjet_order(self))
    vs_out = _faa_di_bruno(vs, dpow, collapsed=True)
    return CollapsedJetTuple((primal, *vs_out))


# ---------------------------------------------------------------------------
# Arithmetic (add, sub are linear; mul uses collapsed Leibniz)
# ---------------------------------------------------------------------------


def cjet_add(
    self: Tensor | CollapsedJetTuple | float | int,
    other: Tensor | CollapsedJetTuple | float | int,
) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.add``."""
    self_is = isinstance(self, CollapsedJetTuple)
    other_is = isinstance(other, CollapsedJetTuple)
    if self_is and other_is:
        _cjet_order(self, other)  # validates K-consistency, raises on mismatch
        coeffs = (s + o for s, o in zip(self, other))
    elif self_is:
        coeffs = (self[0] + other, *self[1:])
    else:
        coeffs = (other[0] + self, *other[1:])
    return CollapsedJetTuple(coeffs)


def cjet_sub(
    self: Tensor | CollapsedJetTuple | float | int,
    other: Tensor | CollapsedJetTuple | float | int,
) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.sub``."""
    self_is = isinstance(self, CollapsedJetTuple)
    other_is = isinstance(other, CollapsedJetTuple)
    if self_is and other_is:
        _cjet_order(self, other)  # validates K-consistency, raises on mismatch
        coeffs = (s - o for s, o in zip(self, other))
    elif self_is:
        coeffs = (self[0] - other, *self[1:])
    else:
        coeffs = (self - other[0], *(-c for c in other[1:]))
    return CollapsedJetTuple(coeffs)


def cjet_mul(
    self: Tensor | CollapsedJetTuple,
    other: Tensor | CollapsedJetTuple,
) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.mul``."""
    self_is = isinstance(self, CollapsedJetTuple)
    other_is = isinstance(other, CollapsedJetTuple)
    if self_is and other_is:
        primal = self[0] * other[0]
        return CollapsedJetTuple(
            (primal, *_collapsed_leibniz(self, other, lambda a, b: a * b))
        )
    elif self_is:
        return _apply_linear(self, lambda c: other * c)
    else:
        return _apply_linear(other, lambda c: self * c)


# ---------------------------------------------------------------------------
# Matrix operations (vmap for one-sided, collapsed Leibniz for both-sided)
# ---------------------------------------------------------------------------


def cjet_mm(
    self: Tensor | CollapsedJetTuple, mat2: Tensor | CollapsedJetTuple
) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.mm``."""
    self_is = isinstance(self, CollapsedJetTuple)
    mat2_is = isinstance(mat2, CollapsedJetTuple)
    if self_is and mat2_is:
        primal = matmul(self[0], mat2[0])
        return CollapsedJetTuple((primal, *_collapsed_leibniz(self, mat2, matmul)))
    elif self_is:
        return _apply_linear(self, lambda c: mm(c, mat2))
    else:
        return _apply_linear(mat2, lambda c: mm(self, c))


def cjet_addmm(
    self: Tensor,
    mat1: Tensor | CollapsedJetTuple,
    mat2: Tensor | CollapsedJetTuple,
) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.addmm``."""
    if isinstance(self, CollapsedJetTuple):
        raise NotImplementedError(
            "cjet_addmm does not support a Taylor-expanded bias (self). "
            "Expected a constant Tensor."
        )
    mat1_is = isinstance(mat1, CollapsedJetTuple)
    mat2_is = isinstance(mat2, CollapsedJetTuple)
    if mat1_is and mat2_is:
        primal = addmm(self, mat1[0], mat2[0])
        return CollapsedJetTuple((primal, *_collapsed_leibniz(mat1, mat2, matmul)))
    elif mat1_is:
        primal = addmm(self, mat1[0], mat2)
        return CollapsedJetTuple(
            (primal, *_apply_linear_coeffs(mat1, lambda c: mm(c, mat2)))
        )
    else:
        primal = addmm(self, mat1, mat2[0])
        return CollapsedJetTuple(
            (primal, *_apply_linear_coeffs(mat2, lambda c: mm(mat1, c)))
        )


def cjet_convolution(
    input: Tensor | CollapsedJetTuple,
    weight: Tensor | CollapsedJetTuple,
    bias: Tensor | None,
    *conv_args: object,
) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.convolution``.

    Args:
        input: The convolution input; a jet or a constant ``Tensor``.
        weight: The convolution kernel; a jet or a constant ``Tensor``.
        bias: The bias; a constant ``Tensor`` or ``None``.
        *conv_args: The remaining ``aten.convolution`` structural arguments,
            forwarded unchanged.

    Returns:
        The value and its Taylor coefficients.

    Raises:
        NotImplementedError: If ``bias`` is Taylor-expanded, or if neither
            ``input`` nor ``weight`` is Taylor-expanded.
    """
    if isinstance(bias, CollapsedJetTuple):
        raise NotImplementedError(
            "cjet_convolution does not support a Taylor-expanded bias. "
            "Expected a constant Tensor or None."
        )
    conv = ops.aten.convolution.default

    def cv(a: Tensor, b: Tensor) -> Tensor:
        """Bias-free convolution -- the bilinear core of ``aten.convolution``."""
        return conv(a, b, None, *conv_args)

    input_is_jet = isinstance(input, CollapsedJetTuple)
    weight_is_jet = isinstance(weight, CollapsedJetTuple)

    if input_is_jet and weight_is_jet:
        primal = conv(input[0], weight[0], bias, *conv_args)
        return CollapsedJetTuple((primal, *_collapsed_leibniz(input, weight, cv)))
    elif input_is_jet:
        primal = conv(input[0], weight, bias, *conv_args)
        return CollapsedJetTuple(
            (primal, *_apply_linear_coeffs(input, lambda c: cv(c, weight)))
        )
    elif weight_is_jet:
        primal = conv(input, weight[0], bias, *conv_args)
        return CollapsedJetTuple(
            (primal, *_apply_linear_coeffs(weight, lambda c: cv(input, c)))
        )
    raise NotImplementedError(
        "cjet_convolution expects input and/or weight to be Taylor-expanded."
    )


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


COLLAPSED_MAPPING: dict = {
    # Elementwise nonlinear
    ops.aten.sin.default: cjet_sin,
    ops.aten.cos.default: cjet_cos,
    ops.aten.tanh.default: cjet_tanh,
    ops.aten.sigmoid.default: cjet_sigmoid,
    ops.aten.relu.default: cjet_relu,
    ops.aten.exp.default: cjet_exp,
    # Power
    ops.aten.pow.Tensor_Scalar: cjet_pow,
    # Arithmetic
    ops.aten.add.Tensor: cjet_add,
    ops.aten.sub.Tensor: cjet_sub,
    ops.aten.mul.Tensor: cjet_mul,
    # Matrix ops
    ops.aten.mm.default: cjet_mm,
    ops.aten.addmm.default: cjet_addmm,
    # Convolution (affine: bias on primal, coefficients convolved bias-free)
    ops.aten.convolution.default: cjet_convolution,
    # Pooling (piecewise linear: gather coefficients at the primal's arg-max)
    ops.aten.max_pool2d_with_indices.default: cjet_max_pool2d_with_indices,
    ops.aten.max_pool2d.default: cjet_max_pool2d,
    # Loss functions
    ops.aten.mse_loss.default: cjet_mse_loss,
}


# --- JAX-style helpers: bulk-register categories of ops ---
#
# Mirrors :func:`jet.operations.deflinear` / :func:`jet.operations.defzero`,
# but uses the collapsed ``_apply_linear`` (which vmaps over the leading
# direction dim for batched coefficients) and respects the collapsed
# per-slot shape contract for ``defzero``.


def deflinear(prim: Callable) -> None:
    """Register ``prim`` as a linear op (collapsed mode).

    Collapsed ``_apply_linear`` vmaps over the leading ``R`` dim for batched
    coefficients ``c_1..c_{K-1}`` and applies ``prim`` directly to the primal
    and the collapsed slot ``c_K``.
    """

    def rule(self: CollapsedJetTuple, *args, **kwargs) -> CollapsedJetTuple:
        return _apply_linear(self, lambda c: prim(c, *args, **kwargs))

    COLLAPSED_MAPPING[prim] = rule


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
    ops.aten.view.default,
    ops.aten._unsafe_view.default,
    ops.aten.unsqueeze.default,
    ops.aten.squeeze.dim,
    ops.aten.squeeze.dims,
    ops.aten.sum.default,
    ops.aten.sum.dim_IntList,
    ops.aten._adaptive_avg_pool2d.default,
    ops.aten.avg_pool2d.default,
    ops.aten.mean.default,
    ops.aten.mean.dim,
):
    deflinear(_prim)

# Constant-output ops.
for _prim in (ops.aten.zeros_like.default,):
    defzero(_prim)
