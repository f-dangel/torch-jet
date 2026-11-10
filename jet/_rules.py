"""Single registry mapping each ATen op to its jet rule.

Every rule is mode-agnostic: it reads ``collapsed`` off its ``JetTuple`` input,
so one body serves standard and collapsed Taylor mode.
"""

from functools import partial
from typing import Callable

from torch import Tensor, ops

from jet import compositions, primitives
from jet.primitives import JetTuple, _deflinear, _deflinear_first, _defzero

#: A jet rule: maps a ``JetTuple`` (and any structural args) to its image.
Rule = Callable[..., JetTuple]


def _defelementwise(deriv_fn: Callable[[Tensor, int], dict[int, Tensor]]) -> Rule:
    """Build the elementwise-unary rule from ``deriv_fn`` (e.g. ``_sin_derivatives``)."""
    return partial(primitives._elementwise, deriv_fn=deriv_fn)


#: Maps an ``aten`` op overload to its jet rule.
RULES: dict[Callable, Rule] = {
    # Elementwise unary (both modes from the shared derivative table)
    ops.aten.sin.default: _defelementwise(primitives._sin_derivatives),
    ops.aten.cos.default: _defelementwise(primitives._cos_derivatives),
    ops.aten.tanh.default: _defelementwise(primitives._tanh_derivatives),
    ops.aten.sigmoid.default: _defelementwise(primitives._sigmoid_derivatives),
    ops.aten.erf.default: _defelementwise(primitives._erf_derivatives),
    ops.aten.relu.default: _defelementwise(primitives._relu_derivatives),
    ops.aten.exp.default: _defelementwise(primitives._exp_derivatives),
    ops.aten.log.default: _defelementwise(primitives._log_derivatives),
    ops.aten.rsqrt.default: _defelementwise(primitives._rsqrt_derivatives),
    ops.aten.hardswish.default: _defelementwise(primitives._hardswish_derivatives),
    ops.aten.silu.default: _defelementwise(primitives._silu_derivatives),
    # Power (also elementwise, but the exponent is a call-time arg, so it carries
    # its own rule rather than registering through ``_defelementwise``).
    ops.aten.pow.Tensor_Scalar: primitives.jet_pow,
    # Hardtanh likewise takes its clamp bounds at call time (``relu6`` lowers
    # to ``hardtanh(x, 0, 6)``).
    ops.aten.hardtanh.default: primitives.jet_hardtanh,
    ops.aten.threshold.default: primitives.jet_threshold,
    # Hardsigmoid's bounds are fixed, so it is an ordinary elementwise table.
    ops.aten.hardsigmoid.default: _defelementwise(primitives._hardsigmoid_derivatives),
    # Mode-agnostic primitives (one body reading ``self.collapsed``)
    ops.aten.add.Tensor: primitives.jet_add,
    ops.aten.add.Scalar: primitives.jet_add,
    ops.aten.sub.Tensor: primitives.jet_sub,
    ops.aten.rsub.Scalar: primitives.jet_rsub,
    ops.aten.mul.Tensor: primitives.jet_mul,
    # ``x * 2.0`` traces to ``mul.Tensor``; decompositions (e.g. the sdpa math
    # path's ``q * sqrt(scale)``) call the scalar overload directly.
    ops.aten.mul.Scalar: primitives.jet_mul,
    ops.aten.mm.default: primitives.jet_mm,
    # Batched matmul: the same bilinear rule (``matmul`` handles the batch dims).
    ops.aten.bmm.default: primitives.jet_mm,
    ops.aten.cat.default: primitives.jet_cat,
    ops.aten.where.self: primitives.jet_where,
    ops.aten.masked_fill.Scalar: primitives.jet_masked_fill,
    ops.aten.masked_fill.Tensor: primitives.jet_masked_fill,
    # ``F.pad(mode="constant")``: affine (the fill value only pads the primal).
    ops.aten.constant_pad_nd.default: primitives.jet_constant_pad_nd,
    ops.aten.max_pool2d_with_indices.default: primitives.jet_max_pool2d_with_indices,
    # The fused, indices-free pooling op some backends emit (e.g. MPS).
    ops.aten.max_pool2d.default: primitives.jet_max_pool2d,
    ops.aten.max_pool2d_backward.default: primitives.jet_max_pool2d_backward,
    ops.aten.nll_loss_forward.default: primitives.jet_nll_loss_forward,
    ops.aten.nll_loss_backward.default: primitives.jet_nll_loss_backward,
    # Token embedding lookup: linear in the weight, integer indices stay constant.
    ops.aten.embedding.default: primitives.jet_embedding,
    ops.aten.index_put.default: primitives.jet_index_put,
    ops.aten.copy.default: primitives.jet_copy,
    ops.aten.as_strided.default: primitives.jet_as_strided,
    ops.aten.as_strided_scatter.default: primitives.jet_as_strided_scatter,
    # Current torch lowers all scalar division (``x / 2.0``) to this overload;
    # a Taylor-expanded divisor uses the division recurrence.
    ops.aten.div.Tensor: primitives.jet_div,
    # Composites -- one mode-agnostic body that pulls its sub-rules from this
    # registry (see :mod:`jet.compositions`).
    ops.aten.addmm.default: compositions.addmm,
    ops.aten.addcmul.default: compositions.addcmul,
    # ``stack(tensors, dim) == cat([unsqueeze(t, dim) for t in tensors], dim)``;
    # decomposed into the registered ``unsqueeze`` and ``cat`` rules.
    ops.aten.stack.default: compositions.stack,
    ops.aten.convolution.default: compositions.convolution,
    ops.aten.mse_loss.default: compositions.mse_loss,
    ops.aten._log_softmax.default: compositions.log_softmax,
    ops.aten._softmax.default: compositions.softmax,
    ops.aten._safe_softmax.default: compositions.safe_softmax,
    ops.aten.gelu.default: compositions.gelu,
    ops.aten.tanh_backward.default: compositions.tanh_backward,
    ops.aten.convolution_backward.default: compositions.convolution_backward,
    ops.aten.hardswish_backward.default: compositions.elementwise_backward(
        primitives._hardswish_derivatives
    ),
    ops.aten.hardsigmoid_backward.default: compositions.elementwise_backward(
        primitives._hardsigmoid_derivatives
    ),
    ops.aten.hardtanh_backward.default: compositions.elementwise_backward(
        primitives._hardtanh_derivatives
    ),
    # The backward never sees ``value`` and the shifted table drops order 0, so
    # any ``value`` works; pass 0.
    ops.aten.threshold_backward.default: compositions.elementwise_backward(
        lambda x0, threshold, K: primitives._threshold_derivatives(
            x0, threshold, 0.0, K
        )
    ),
    ops.aten.gelu_backward.default: compositions.elementwise_backward(
        primitives._gelu_derivatives
    ),
    ops.aten.native_batch_norm.default: compositions.native_batch_norm,
    ops.aten.native_layer_norm.default: compositions.native_layer_norm,
    # CUDA / MPS lower ``nn.RMSNorm`` to this fused op (CPU traces its primitives).
    ops.aten._fused_rms_norm.default: compositions._fused_rms_norm,
    # CUDA lowers eval-mode batch norm to the cuDNN overload (same map, one
    # extra output).
    ops.aten.cudnn_batch_norm.default: compositions.cudnn_batch_norm,
    # On MPS the math attention backend is one kernel (CPU and CUDA trace the
    # math-path ops).
    ops.aten._scaled_dot_product_attention_math_for_mps.default: compositions._scaled_dot_product_attention_math_for_mps,
}

# Linear ops (pointwise-linear, shape-only, reductions): one mode-agnostic
# `_deflinear` rule each (its `_apply_linear` follows the jet's collapsed flag).
# Composite rules (:mod:`jet.compositions`) reuse several of these -- e.g.
# ``sum.dim_IntList`` in ``log_softmax``, ``view.default`` in
# ``native_batch_norm`` -- by pulling them straight from this registry.
for _prim in (
    ops.aten.neg.default,
    ops.aten.div.Scalar,
    ops.aten.t.default,
    ops.aten.transpose.int,
    ops.aten.permute.default,
    ops.aten.expand.default,
    ops.aten.clone.default,
    # Transformer views: ``x[i]`` (``select``), ``x[a:b]`` (``slice``), fancy
    # ``x[idx]`` (``index``; its ``indices: Tensor?[]`` list is forwarded as-is),
    # and ``.to(dtype)`` (``_to_copy``; still linear with a dtype change).
    ops.aten.select.int,
    ops.aten.slice.Tensor,
    ops.aten.index.Tensor,
    ops.aten._to_copy.default,
    # ``x[:, 0:]`` (e.g. a language model head keeping all logits): a no-op view.
    ops.aten.alias.default,
    ops.aten._unsafe_view.default,
    ops.aten.unsqueeze.default,
    ops.aten.squeeze.dim,
    ops.aten.squeeze.dims,
    ops.aten.sum.default,
    ops.aten.sum.dim_IntList,
    ops.aten.view.default,
    ops.aten.reshape.default,  # ``capture_graph`` rewrites ``view`` to this
    ops.aten._adaptive_avg_pool2d.default,
    ops.aten.avg_pool2d.default,
    # Bilinear upsampling (``F.interpolate(mode="bilinear")``, e.g. torchvision's
    # segmentation heads); both ``size=`` and ``scale_factor=`` lower to it.
    ops.aten.upsample_bilinear2d.default,
    ops.aten.mean.default,
    ops.aten.mean.dim,
    ops.aten.split.Tensor,  # ``Tensor[]`` output: one jet per piece
    ops.aten.select_backward.default,
    ops.aten.slice_backward.default,
    ops.aten.upsample_bilinear2d_backward.default,
    ops.aten.embedding_dense_backward.default,
):
    RULES[_prim] = _deflinear(_prim)

for _prim in (
    ops.aten.avg_pool2d_backward.default,
    ops.aten._adaptive_avg_pool2d_backward.default,
    ops.aten.max_pool2d_with_indices_backward.default,
):
    RULES[_prim] = _deflinear_first(_prim)

# Constant-output ops: primal carries the value, coefficients are zero.
for op in (
    ops.aten.zeros_like.default,
    ops.aten.ones_like.default,
    ops.aten.empty_like.default,
    ops.aten.fill.Scalar,
    ops.aten.new_zeros.default,
):
    RULES[op] = _defzero(op)
