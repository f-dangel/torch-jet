"""Composite jet rules: ATen ops that decompose into other registered ops.

Each rule here is **mode-agnostic** -- a single function that pulls its
sub-rules from the shared ``RULES`` registry (:mod:`jet._rules`). The
primitives a composition wires together (``add``, ``mm``, ``pow``, ...) read
the ``collapsed`` flag off their jet inputs and carry the per-mode arithmetic;
the composition only encodes the algebraic identity (e.g.
``addmm == add(self, mm(mat1, mat2))``).
"""

from math import pi, sqrt
from operator import neg, sub
from typing import Callable

from torch import Tensor, contiguous_format, finfo, ones, ops, uint8
from torch import bool as bool_dtype
from torch import dtype as torch_dtype

import jet._rules
from jet.primitives import (
    JetTuple,
    _align_conv_bias,
    _apply_bilinear,
    _apply_linear,
    _bn_channel_view,
    _deflinear,
    _elementwise,
    _gelu_exact_derivatives,
    _has_jet_arg,
    _pointwise,
    _sum_exp,
    jet_mm,
)


def _rule(op: Callable) -> Callable:
    """Fetch ``op``'s rule from the shared registry.

    ``jet._rules`` imports this module to assemble ``RULES``, so the registry
    is reached through the module (``jet._rules.RULES``) and read at call time;
    binding the ``RULES`` name at import time would be circular.

    Falls through to the native op when no argument is a jet, like the
    interpreter, so compositions may feed all-constant intermediates to
    sub-rules.
    """

    def dispatch(*args: object, **kwargs: object) -> object:
        if _has_jet_arg((*args, *kwargs.values())):
            return jet._rules.RULES[op](*args, **kwargs)
        return op(*args, **kwargs)

    return dispatch


def addmm(
    self: Tensor | JetTuple, mat1: Tensor | JetTuple, mat2: Tensor | JetTuple
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.addmm(self, mat1, mat2)``.

    ``addmm(self, mat1, mat2) == self + mat1 @ mat2``, so the rule composes the
    matrix-product rule with the affine bias addition. Any operand may be
    Taylor-expanded, including the bias; ``add`` broadcasts a lower-rank bias
    over the product's rows, and ``mm`` returns a plain tensor when both
    matrices are constant.

    Args:
        self: The bias; a jet or a constant ``Tensor``.
        mat1: The first matrix; a jet or a constant ``Tensor``.
        mat2: The second matrix; a jet or a constant ``Tensor``.

    Returns:
        The value and its Taylor coefficients.
    """
    add = _rule(ops.aten.add.Tensor)
    mm = _rule(ops.aten.mm.default)
    return add(self, mm(mat1, mat2))


def addcmul(
    self: Tensor | JetTuple,
    tensor1: Tensor | JetTuple,
    tensor2: Tensor | JetTuple,
    value: float | int = 1,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.addcmul(self, tensor1, tensor2, value)``.

    ``addcmul(self, tensor1, tensor2, value) == self + value * tensor1 * tensor2``,
    so the rule composes the bilinear ``mul`` rule with the affine addition. Any
    operand may be Taylor-expanded. A nested jet meets this op in the fused
    joins of the Faà di Bruno terms (:func:`jet.primitives._partition_term`).

    Args:
        self: The summand; a jet or a constant ``Tensor``.
        tensor1: The first factor; a jet or a constant ``Tensor``.
        tensor2: The second factor; a jet or a constant ``Tensor``.
        value: The scalar that scales the product. Default: ``1``.

    Returns:
        The value and its Taylor coefficients.
    """
    add, mul = _rule(ops.aten.add.Tensor), _rule(ops.aten.mul.Tensor)
    product = mul(tensor1, tensor2)
    return add(self, product if value == 1 else mul(product, value))


def stack(tensors: list[Tensor | JetTuple], dim: int = 0) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.stack(tensors, dim)``.

    ``stack(tensors, dim) == cat([unsqueeze(t, dim) for t in tensors], dim)``, so
    the rule inserts a fresh axis at ``dim`` in each operand (via the linear
    ``unsqueeze`` rule) and concatenates along it (via the ``cat`` rule). Any
    operand may be Taylor-expanded: ``unsqueeze`` passes a constant tensor
    through and lifts each coefficient of a jet, and ``cat`` already handles a
    mixed list of jets and constants plus the collapsed-mode direction-dim shift.

    Args:
        tensors: The operands to stack; each a jet or a constant ``Tensor``, all
            of equal shape.
        dim: The axis at which to insert the new dimension.

    Returns:
        The value and its Taylor coefficients.
    """
    cat = _rule(ops.aten.cat.default)
    unsqueeze = _rule(ops.aten.unsqueeze.default)
    return cat([unsqueeze(t, dim) for t in tensors], dim)


def convolution(
    input: Tensor | JetTuple,
    weight: Tensor | JetTuple,
    bias: Tensor | JetTuple | None,
    *conv_args: object,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.convolution(input, weight, bias, ...)``.

    ``convolution(input, weight, bias) == convolution(input, weight, None) +
    bias``, so the rule composes the bilinear bias-free convolution (the Leibniz
    rule when both are jets, else coefficient-wise) with the affine bias
    addition, mirroring :func:`addmm`. Any operand may be Taylor-expanded,
    including the bias.

    Args:
        input: The convolution input; a jet or a constant ``Tensor``.
        weight: The convolution kernel; a jet or a constant ``Tensor``.
        bias: The bias; a jet, a constant ``Tensor``, or ``None``.
        *conv_args: The remaining ``aten.convolution`` structural arguments
            (``stride``, ``padding``, ``dilation``, ``transposed``,
            ``output_padding``, ``groups``), forwarded unchanged.

    Returns:
        The value and its Taylor coefficients.
    """
    add = _rule(ops.aten.add.Tensor)

    def cv(a: Tensor, b: Tensor) -> Tensor:
        """Bias-free convolution -- the bilinear core of ``aten.convolution``."""
        return ops.aten.convolution.default(a, b, None, *conv_args)

    product = _apply_bilinear(cv, input, weight)
    if bias is None:
        return product
    # Reshape the 1-D bias to broadcast over the output's batch and spatial dims
    # (channel is dim 1) and defer the add to ``add``. Conv preserves rank, so
    # the output ndim is the input ndim.
    ndim = (input[0] if isinstance(input, JetTuple) else input).ndim
    return add(_align_conv_bias(bias, ndim), product)


def _reduce_loss(loss: JetTuple, reduction: int) -> JetTuple:
    """Apply a loss reduction coefficient-wise.

    ``reduction`` follows ATen's enum -- ``0`` (none, identity), ``1`` (mean),
    ``2`` (sum). Reductions are linear, so they pull the registered ``mean`` /
    ``sum`` rule (which, in collapsed mode, vmaps over the direction dim ``R``).

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
        return _rule(ops.aten.mean.default)(loss)
    if reduction == 2:  # 'sum'
        return _rule(ops.aten.sum.default)(loss)
    raise ValueError(f"Unsupported reduction {reduction}; expected 0, 1, or 2.")


def mse_loss(
    self: Tensor | JetTuple, target: Tensor | JetTuple, reduction: int = 1
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.mse_loss(self, target, reduction)``.

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
    sub = _rule(ops.aten.sub.Tensor)
    pow = _rule(ops.aten.pow.Tensor_Scalar)
    squared_error = pow(sub(self, target), 2)
    return _reduce_loss(squared_error, reduction)


def log_softmax(self: JetTuple, dim: int, half_to_float: bool = False) -> JetTuple:
    """Taylor-mode arithmetic for ``aten._log_softmax(self, dim, half_to_float)``.

    Uses the shift-invariant identity
    ``log_softmax(x) = (x - m) - log(sum(exp(x - m), dim))`` with
    ``m = max(x, dim)`` a constant taken from the primal. This is the
    log-sum-exp trick, and it stabilizes the whole jet -- not just the forward
    pass. The sum of the exponentials is streamed by :func:`_sum_exp` to reduce
    memory. That matters for the cross entropy of a language model, whose
    logits are the largest tensor of the whole graph.

    Args:
        self: The logits and their Taylor coefficients.
        dim: The dimension along which to normalize.
        half_to_float: Whether to promote half-precision inputs to float32
            before computing, like the ATen op (then the output is float32).

    Returns:
        The value and its Taylor coefficients.
    """
    if half_to_float:
        self = _apply_linear(self, lambda c: c.float())
    log = _rule(ops.aten.log.default)
    dim = dim % self[0].ndim
    shift = self[0].amax(dim, keepdim=True)
    log_sum = log(_sum_exp(self, shift, dim))
    primal = (self[0] - shift) - log_sum[0]
    return _pointwise(self, log_sum, sub, neg, primal=primal)


def softmax(self: JetTuple, dim: int, half_to_float: bool = False) -> JetTuple:
    """Taylor-mode arithmetic for ``aten._softmax(self, dim, half_to_float)``.

    Uses the shift-invariant identity
    ``softmax(x) = exp(x - m) / sum(exp(x - m), dim)`` with ``m = max(x, dim)``
    a constant taken from the primal. ``exp`` reuses the elementwise machinery,
    the sum over ``dim`` is linear, and the final division broadcasts. The
    shift keeps every exponential at most ``1``, which stabilizes the whole
    jet -- not just the forward pass.

    Args:
        self: The logits and their Taylor coefficients.
        dim: The dimension along which to normalize.
        half_to_float: Whether to promote half-precision inputs to float32
            before computing, like the ATen op (then the output is float32).

    Returns:
        The value and its Taylor coefficients.
    """
    if half_to_float:
        self = _apply_linear(self, lambda c: c.float())
    sub = _rule(ops.aten.sub.Tensor)
    exp = _rule(ops.aten.exp.default)
    sum = _rule(ops.aten.sum.dim_IntList)
    div = _rule(ops.aten.div.Tensor)
    shift = self[0].amax(dim, keepdim=True)
    exp_jet = exp(sub(self, shift))
    return div(exp_jet, sum(exp_jet, dim, keepdim=True))


def safe_softmax(
    self: JetTuple, dim: int, dtype: torch_dtype | None = None
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten._safe_softmax(self, dim, dtype)``.

    The ``softmax`` of sdpa's math fallback (e.g. float64 on CUDA): a fully
    masked row (all ``-inf``) gives ``0`` instead of ``NaN``, so every
    coefficient of such a row is zeroed.

    Args:
        self: The logits and their Taylor coefficients.
        dim: The dimension along which to normalize.
        dtype: An optional dtype to cast the input to first.

    Returns:
        The value and its Taylor coefficients.
    """
    if dtype is not None:
        self = _apply_linear(self, lambda c: c.to(dtype))
    masked = self[0].isneginf().all(dim, keepdim=True)
    return _apply_linear(softmax(self, dim), lambda c: c.masked_fill(masked, 0.0))


def gelu(self: JetTuple, approximate: str = "none") -> JetTuple:
    """Taylor-mode arithmetic for ``aten.gelu(self, approximate=...)``.

    The exact GELU is one elementwise rule over its closed-form derivative
    table (one Faà di Bruno pass, no products of jets). The ``"tanh"``
    approximation ``0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`` stays
    composed from the ``tanh`` / ``pow`` / ``mul`` / ``add`` rules: its table
    composes ``tanh`` over the cubic at every order, which costs more than
    the composition on large inputs.

    Args:
        self: The input and its Taylor coefficients.
        approximate: ``"none"`` (exact) or ``"tanh"``.

    Returns:
        The value and its Taylor coefficients.

    Raises:
        NotImplementedError: If ``approximate`` is neither ``"none"`` nor ``"tanh"``.
    """
    if approximate == "none":
        return _elementwise(self, _gelu_exact_derivatives)
    mul = _rule(ops.aten.mul.Tensor)
    add = _rule(ops.aten.add.Tensor)
    if approximate == "tanh":
        pow = _rule(ops.aten.pow.Tensor_Scalar)
        cubic = add(self, mul(pow(self, 3), 0.044715))
        inner = _rule(ops.aten.tanh.default)(mul(cubic, sqrt(2 / pi)))
    else:
        raise NotImplementedError(f"gelu: unsupported approximate={approximate!r}")
    return mul(mul(self, 0.5), add(inner, 1.0))


def _standardize(input: Tensor | JetTuple, dims: list[int], eps: float) -> tuple:
    """Standardize ``input`` over ``dims`` with its own statistics.

    ``(input - mean) * (var + eps)**-0.5`` with biased ``var``, composed from
    the registered ``mean`` / ``sub`` / ``mul`` / ``pow`` / ``add`` rules. The
    shared core of layer / group / instance norm; ``input`` may be a jet or a
    constant.

    Returns:
        ``(out, mean, rstd)``; ``mean`` / ``rstd`` keep ``dims`` as size 1.
    """
    sub = _rule(ops.aten.sub.Tensor)
    mul = _rule(ops.aten.mul.Tensor)
    add = _rule(ops.aten.add.Tensor)
    pow = _rule(ops.aten.pow.Tensor_Scalar)
    mean = _rule(ops.aten.mean.dim)
    mu = mean(input, dims, keepdim=True)
    centered = sub(input, mu)
    rstd = pow(add(mean(pow(centered, 2), dims, keepdim=True), eps), -0.5)
    return mul(centered, rstd), mu, rstd


def _affine(
    out: Tensor | JetTuple,
    weight: Tensor | JetTuple | None,
    bias: Tensor | JetTuple | None,
) -> Tensor | JetTuple:
    """``out * weight + bias`` for optional (``None`` = absent) broadcastable params."""
    if weight is not None:
        out = _rule(ops.aten.mul.Tensor)(out, weight)
    if bias is not None:
        out = _rule(ops.aten.add.Tensor)(out, bias)
    return out


def native_batch_norm(
    input: Tensor | JetTuple,
    weight: Tensor | JetTuple | None,
    bias: Tensor | JetTuple | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    training: bool,
    momentum: float,
    eps: float,
) -> tuple:
    """Taylor-mode arithmetic for ``aten.native_batch_norm`` (eval mode).

    Eval-mode batch norm normalizes each channel with the frozen running
    statistics, evaluated in this order:

        ``((input - running_mean) * (weight / sqrt(running_var + eps))) + bias``

    The scale ``weight / sqrt(running_var + eps)`` is formed on the
    channel-sized tensors, so the activation jet is scaled once. The constant
    ``running_mean`` is subtracted first, from the primal only (coefficients
    pass through), which keeps the digits of inputs close to a large running
    mean; ``input * scale - running_mean * scale`` would cancel them. Any of
    ``input`` / ``weight`` / ``bias`` may be a jet, a constant, or (``weight``
    / ``bias``) ``None`` -- any input rank
    (1d/2d/3d batch norm) is supported. Training mode is deferred until PyTorch
    fixes its fused ``native_batch_norm``'s incorrect higher-order autograd in
    training (pytorch/pytorch#186256), without which a training rule cannot be
    validated.

    Returns:
        The ATen op's ``(output, save_mean, save_invstd)`` triple. ``save_mean``
        / ``save_invstd`` are empty (only the forward output is consumed in a
        Taylor-mode pass).

    Raises:
        NotImplementedError: In training mode, or in eval mode without running
            statistics (the batch-statistic path, not the affine eval map).
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

    sub = _rule(ops.aten.sub.Tensor)
    mul = _rule(ops.aten.mul.Tensor)
    add = _rule(ops.aten.add.Tensor)
    view = _rule(ops.aten.view.default)

    primal = input[0] if isinstance(input, JetTuple) else input
    shape = _bn_channel_view(primal)
    rstd = view((running_var + eps).rsqrt(), shape)
    scale = rstd if weight is None else mul(view(weight, shape), rstd)
    out = mul(sub(input, view(running_mean, shape)), scale)
    out = out if bias is None else add(out, view(bias, shape))
    empty = primal.new_empty(0)
    return out, empty, empty


def native_layer_norm(
    input: Tensor | JetTuple,
    normalized_shape: list[int],
    weight: Tensor | JetTuple | None,
    bias: Tensor | JetTuple | None,
    eps: float,
) -> tuple:
    """Taylor-mode arithmetic for ``aten.native_layer_norm``.

    Layer norm standardizes each sample over its trailing ``normalized_shape``
    dims with its own statistics (:func:`_standardize`), then applies the
    optional affine ``weight`` / ``bias`` (:func:`_affine`); ``input`` /
    ``weight`` / ``bias`` may each be a jet or a constant.

    Returns:
        The ATen op's ``(output, mean, rstd)`` triple as a plain tuple of jets
        (``mean`` / ``rstd`` keep the normalized dims as size 1, like the op),
        so a downstream ``getitem`` falls through to the native op.
    """
    dims = list(range(-len(normalized_shape), 0))
    out, mu, rstd = _standardize(input, dims, eps)
    out = _affine(out, weight, bias)
    # The ATen op's output is contiguous (ConvNeXt ``view``s it after a permute).
    if not out[0].is_contiguous():
        clone = _rule(ops.aten.clone.default)
        out = clone(out, memory_format=contiguous_format)
    return out, mu, rstd


def _fused_rms_norm(
    input: Tensor | JetTuple,
    normalized_shape: list[int],
    weight: Tensor | JetTuple | None,
    eps: float | None,
) -> tuple:
    """Taylor-mode arithmetic for ``aten._fused_rms_norm``.

    On CUDA / MPS, ``nn.RMSNorm`` lowers to this fused op; on CPU, to ``pow`` /
    ``mean`` / ``add`` / ``rsqrt`` / ``mul``, which have rules of their own and
    which this rule mirrors. It rescales each sample by its root mean square
    over the trailing ``normalized_shape`` dims,
    ``input * rsqrt(mean(input**2) + eps)``, then applies the optional
    ``weight`` (:func:`_affine`); ``input`` / ``weight`` may each be a jet or a
    constant. ``eps=None`` is the op's default, ``finfo(input.dtype).eps``.

    Returns:
        The ATen op's ``(output, rstd)`` pair as a plain tuple of jets (``rstd``
        keeps the normalized dims as size 1, like the op on CPU / CUDA; on MPS
        the op returns ``None`` there), so a downstream ``getitem`` falls
        through to the native op.
    """
    mul = _rule(ops.aten.mul.Tensor)
    add = _rule(ops.aten.add.Tensor)
    pow = _rule(ops.aten.pow.Tensor_Scalar)
    mean = _rule(ops.aten.mean.dim)
    rsqrt = _rule(ops.aten.rsqrt.default)
    if eps is None:
        eps = finfo((input[0] if isinstance(input, JetTuple) else input).dtype).eps
    dims = list(range(-len(normalized_shape), 0))
    rstd = rsqrt(add(mean(pow(input, 2), dims, keepdim=True), eps))
    return _affine(mul(input, rstd), weight, None), rstd


def cudnn_batch_norm(
    input: Tensor | JetTuple,
    weight: Tensor | JetTuple,
    bias: Tensor | JetTuple | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    training: bool,
    exponential_average_factor: float,
    epsilon: float,
) -> tuple:
    """Taylor-mode arithmetic for ``aten.cudnn_batch_norm`` (eval mode).

    On CUDA, eval-mode batch norm lowers to this cuDNN overload. It computes
    the same map as :func:`native_batch_norm` (which does the work), with a
    required ``weight`` and a fourth, cuDNN-internal ``reserve`` output.

    Returns:
        The ATen op's ``(output, save_mean, save_var, reserve)`` quadruple; the
        last three are empty (only the forward output is consumed), with the
        byte-typed ``reserve`` of the ATen op.
    """
    out, empty, _ = native_batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
    )
    return out, empty, empty, empty.new_empty(0, dtype=uint8)


def _scaled_dot_product_attention_math_for_mps(
    query: Tensor | JetTuple,
    key: Tensor | JetTuple,
    value: Tensor | JetTuple,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    dropout_mask: Tensor | None = None,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> tuple[JetTuple, None]:
    """Taylor-mode arithmetic for the MPS attention kernel (eval mode).

    On MPS ``F.scaled_dot_product_attention`` lowers to this one math kernel
    (CPU and CUDA trace its math-path ops instead). ``softmax(scale * query @
    key^T + mask, -1) @ value`` (``scale = 1/sqrt(E)`` by default), composed
    from the ``mul`` / ``transpose`` / ``masked_fill`` / ``add`` / ``_softmax``
    rules and :func:`jet_mm` (``matmul`` broadcasts over the leading batch
    dims). A causal or bool ``attn_mask`` (``True`` = attend) fills the
    blocked scores with ``-inf``, a float one is added. The ``_softmax``
    rule's max-shift keeps the whole jet stable; a fully masked row gives
    ``NaN`` like the ATen math path. Any of ``query`` ``(..., L, E)`` / ``key``
    ``(..., S, E)`` / ``value`` ``(..., S, Ev)`` may be a jet or a constant;
    ``attn_mask`` is a constant broadcastable to ``(..., L, S)``. Grouped-query
    attention (fewer key/value heads than query heads along dim ``-3``) repeats
    each key/value head for its query heads; ``enable_gqa`` is implied by the
    head counts.

    Returns:
        The value and its Taylor coefficients, shape ``(..., L, Ev)``, and
        ``None`` for the kernel's backward-only second output, so that any
        consumer (e.g. a future backward rule for nesting) fails loudly.

    Raises:
        NotImplementedError: If ``dropout_p != 0`` or ``dropout_mask`` is given
            (jets require eval mode, ``model.eval()``).
    """
    if dropout_p != 0 or dropout_mask is not None:
        raise NotImplementedError(
            "Taylor-mode scaled_dot_product_attention requires dropout_p == 0 and "
            "no dropout_mask; jets require eval mode (call model.eval())."
        )
    mul = _rule(ops.aten.mul.Tensor)
    transpose = _rule(ops.aten.transpose.int)
    q, k = (x[0] if isinstance(x, JetTuple) else x for x in (query, key))
    if q.shape[-3] != k.shape[-3]:
        repeat = _deflinear(ops.aten.repeat_interleave.self_int)
        groups = q.shape[-3] // k.shape[-3]
        key, value = (repeat(x, groups, -3) for x in (key, value))
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    scores = jet_mm(mul(query, scale), transpose(key, -2, -1))
    if is_causal:
        attn_mask = ones(
            q.shape[-2], k.shape[-2], dtype=bool_dtype, device=q.device
        ).tril()
    if attn_mask is not None:
        if attn_mask.dtype == bool_dtype:
            masked_fill = _rule(ops.aten.masked_fill.Scalar)
            scores = masked_fill(scores, ~attn_mask, -float("inf"))
        else:
            scores = _rule(ops.aten.add.Tensor)(scores, attn_mask)
    attn = _rule(ops.aten._softmax.default)(scores, dim=-1, half_to_float=False)
    return jet_mm(attn, value), None


def tanh_backward(
    grad_output: Tensor | JetTuple, output: Tensor | JetTuple
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.tanh_backward(grad_output, output)``.

    Autograd's backward of ``tanh`` computes ``grad_output * (1 - output^2)``,
    composed in that order. Either operand may be a jet (a jet of a gradient
    expands both); a constant ``output`` makes ``1 - output^2`` a plain tensor
    and the final product a linear rule.

    Args:
        grad_output: The incoming gradient; a jet or a constant ``Tensor``.
        output: ``tanh``'s output; a jet or a constant ``Tensor``.

    Returns:
        The value and its Taylor coefficients.
    """
    mul = _rule(ops.aten.mul.Tensor)
    pow = _rule(ops.aten.pow.Tensor_Scalar)
    sub = _rule(ops.aten.sub.Tensor)
    return mul(grad_output, sub(1, pow(output, 2)))


def convolution_backward(
    grad_output: Tensor | JetTuple,
    input: Tensor | JetTuple,
    weight: Tensor | JetTuple,
    bias_sizes: list[int] | None,
    *conv_args: object,
) -> tuple[Tensor | JetTuple | None, ...]:
    """Taylor-mode arithmetic for ``aten.convolution_backward``.

    The op returns the triple ``(grad_input, grad_weight, grad_bias)`` and
    takes a list of three booleans, ``output_mask``, that says which entries
    to compute; the others are returned as ``None``. Autograd masks out the
    gradients it does not need (e.g. ``grad_input`` when ``input`` does not
    require grad, ``grad_bias`` for a bias-free convolution).

    Each entry has its own Taylor-mode structure: ``grad_input`` is bilinear
    in ``(grad_output, weight)``, ``grad_weight`` is bilinear in
    ``(grad_output, input)``, and ``grad_bias`` sums ``grad_output`` over the
    batch and spatial dims, so it is linear.

    The remaining tensor of each call (``input`` for ``grad_input``,
    ``weight`` for ``grad_weight``, both for ``grad_bias``) only provides a
    shape, so the primal goes there.

    Args:
        grad_output: The incoming gradient; a jet or a constant ``Tensor``.
        input: The convolution input; a jet or a constant ``Tensor``.
        weight: The convolution kernel; a jet or a constant ``Tensor``.
        bias_sizes: The bias shape, or ``None`` for a bias-free convolution.
        *conv_args: The remaining structural arguments (``stride``,
            ``padding``, ``dilation``, ``transposed``, ``output_padding``,
            ``groups``), forwarded unchanged, followed by the ``output_mask``.

    Returns:
        The op's ``(grad_input, grad_weight, grad_bias)`` triple, each a jet
        (or a constant when none of its operands is a jet), ``None`` where
        ``output_mask`` is ``False``.
    """
    *conv_args, output_mask = conv_args
    x, w = (t[0] if isinstance(t, JetTuple) else t for t in (input, weight))

    def piece(index: int) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
        """``convolution_backward`` restricted to output ``index``."""
        mask = [i == index for i in range(3)]

        def op(g: Tensor, a: Tensor, b: Tensor) -> Tensor:
            return ops.aten.convolution_backward.default(
                g, a, b, bias_sizes, *conv_args, mask
            )[index]

        return op

    input_piece, weight_piece, bias_piece = (piece(i) for i in range(3))

    grad_input = grad_weight = grad_bias = None
    if output_mask[0]:
        grad_input = _apply_bilinear(
            lambda g, b: input_piece(g, x, b), grad_output, weight
        )
    if output_mask[1]:
        grad_weight = _apply_bilinear(
            lambda g, a: weight_piece(g, a, w), grad_output, input
        )
    if output_mask[2]:
        grad_bias = _deflinear(bias_piece)(grad_output, x, w)
    return grad_input, grad_weight, grad_bias


def elementwise_backward(
    deriv_fn: Callable[..., dict[int, Tensor | None]],
) -> Callable[..., JetTuple]:
    """Build the rule for the backward of an elementwise activation ``f``.

    Autograd's backward of ``f`` is ``grad_output * f'(self)`` (plus any
    call-time args, e.g. ``hardtanh``'s bounds, or keyword args, e.g.
    ``gelu``'s ``approximate``). The jet of ``f'`` reuses ``f``'s derivative
    table shifted up by one order, and the product is the bilinear ``mul``
    rule, so either operand may be a jet or a constant.

    Args:
        deriv_fn: ``f``'s derivative table,
            ``(x0, *args, K, **kwargs) -> {k: f^(k)(x0)}``.

    Returns:
        The rule for ``aten.f_backward(grad_output, self, *args, **kwargs)``.
    """

    def rule(
        grad_output: Tensor | JetTuple,
        self: Tensor | JetTuple,
        *args: object,
        **kwargs: object,
    ) -> JetTuple:
        def shifted(x0: Tensor, K: int) -> dict[int, Tensor | None]:
            d = deriv_fn(x0, *args, K + 1, **kwargs)
            return {k: d[k + 1] for k in range(K + 1)}

        mul = _rule(ops.aten.mul.Tensor)
        return mul(grad_output, _elementwise(self, shifted))

    return rule
