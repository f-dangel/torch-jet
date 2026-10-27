"""Composite jet rules: ATen ops that decompose into other registered ops.

Each rule here is **mode-agnostic** -- a single function that reads the
``collapsed`` flag off its jet input and pulls its sub-rules from the shared
``RULES`` registry (:mod:`jet._rules`), so standard and collapsed Taylor mode
share one body. The primitives a composition wires together (``add``, ``mm``,
``pow``, ...) carry the per-mode arithmetic; the composition only encodes the
algebraic identity (e.g. ``addmm == add(self, mm(mat1, mat2))``).
"""

from math import pi, sqrt
from typing import Callable

from torch import contiguous_format, finfo, ops, uint8

import jet._rules
from jet.primitives import (
    JetTuple,
    _align_conv_bias,
    _apply_bilinear,
    _bn_channel_view,
)


def _collapsed_of(*args: object) -> bool:
    """Return the shared ``collapsed`` flag of the ``JetTuple`` args.

    The interpreter only dispatches to a rule when at least one argument is a
    jet, so a ``JetTuple`` is always present. All jet arguments must agree on
    the mode -- a whole run is either standard or collapsed -- so a mix is a
    bug (mirrors :func:`jet.primitives._jet_order`'s single-``K`` check).

    Raises:
        TypeError: If no argument is a ``JetTuple`` (should be unreachable).
        ValueError: If two ``JetTuple`` args disagree on ``collapsed``.
    """
    flags = {arg.collapsed for arg in args if isinstance(arg, JetTuple)}
    if not flags:
        raise TypeError("composition rule called without a JetTuple argument")
    if len(flags) > 1:
        raise ValueError(
            "composition rule received JetTuple arguments with mixed collapsed "
            "flags; all jets in a run must share the same mode"
        )
    return flags.pop()


def _rule(op: Callable, collapsed: bool) -> Callable:
    """Fetch ``op``'s rule for ``collapsed`` mode from the shared registry.

    ``jet._rules`` imports this module to assemble ``RULES``, so the registry
    is reached through the module (``jet._rules.RULES``) and read at call time;
    binding the ``RULES`` name at import time would be circular.
    """
    return jet._rules.RULES[op][collapsed]


def addmm(self: object, mat1: object, mat2: object) -> JetTuple:
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
    collapsed = _collapsed_of(self, mat1, mat2)
    add = _rule(ops.aten.add.Tensor, collapsed)
    mm = _rule(ops.aten.mm.default, collapsed)
    return add(self, mm(mat1, mat2))


def div(self: object, other: object) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.div.Tensor(self, other)``.

    ``a / b == a * b**(-1)``, so the rule composes the reciprocal (``pow`` with
    exponent ``-1``) with the product rule. Any operand may be Taylor-expanded,
    including the divisor: the ``pow`` rule reciprocates ``other`` whether it is
    a jet (the same reciprocal series ``log`` already relies on) or a constant.
    Because ``mul`` broadcasts a lower-rank operand, division broadcasts too.

    Args:
        self: The numerator; a jet or a constant ``Tensor``.
        other: The divisor; a jet, a constant ``Tensor``, or a Python scalar.

    Returns:
        The value and its Taylor coefficients.
    """
    collapsed = _collapsed_of(self, other)
    mul = _rule(ops.aten.mul.Tensor, collapsed)
    reciprocal = _rule(ops.aten.pow.Tensor_Scalar, collapsed)
    return mul(self, reciprocal(other, -1))


def stack(tensors: list[object], dim: int = 0) -> JetTuple:
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
    collapsed = _collapsed_of(*tensors)
    cat = _rule(ops.aten.cat.default, collapsed)
    unsqueeze = _rule(ops.aten.unsqueeze.default, collapsed)
    return cat([unsqueeze(t, dim) for t in tensors], dim)


def convolution(
    input: object, weight: object, bias: object, *conv_args: object
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
    collapsed = _collapsed_of(input, weight, bias)
    add = _rule(ops.aten.add.Tensor, collapsed)

    def cv(a: object, b: object) -> object:
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
        return _rule(ops.aten.mean.default, loss.collapsed)(loss)
    if reduction == 2:  # 'sum'
        return _rule(ops.aten.sum.default, loss.collapsed)(loss)
    raise ValueError(f"Unsupported reduction {reduction}; expected 0, 1, or 2.")


def mse_loss(self: object, target: object, reduction: int = 1) -> JetTuple:
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
    collapsed = _collapsed_of(self, target)
    sub = _rule(ops.aten.sub.Tensor, collapsed)
    pow = _rule(ops.aten.pow.Tensor_Scalar, collapsed)
    squared_error = pow(sub(self, target), 2)
    return _reduce_loss(squared_error, reduction)


def log_softmax(self: JetTuple, dim: int, half_to_float: bool = False) -> JetTuple:
    """Taylor-mode arithmetic for ``aten._log_softmax(self, dim, half_to_float)``.

    Uses the shift-invariant identity
    ``log_softmax(x) = (x - m) - log(sum(exp(x - m), dim))`` with
    ``m = max(x, dim)`` a constant taken from the primal. ``exp`` and ``log``
    reuse the elementwise machinery, the sum over ``dim`` is linear, and the
    final subtraction broadcasts. This is the log-sum-exp trick, and it
    stabilizes the whole jet -- not just the forward pass.

    Args:
        self: The logits and their Taylor coefficients.
        dim: The dimension along which to normalize.
        half_to_float: Whether inputs were promoted from half precision.
            Accepted for ATen-signature compatibility; does not affect the
            float32/float64 paths.

    Returns:
        The value and its Taylor coefficients.
    """
    collapsed = _collapsed_of(self)
    sub = _rule(ops.aten.sub.Tensor, collapsed)
    exp = _rule(ops.aten.exp.default, collapsed)
    log = _rule(ops.aten.log.default, collapsed)
    sum = _rule(ops.aten.sum.dim_IntList, collapsed)
    shift = self[0].amax(dim, keepdim=True)
    shifted = sub(self, shift)
    exp_jet = exp(shifted)
    sum_exp = sum(exp_jet, dim, keepdim=True)
    log_sum_exp = log(sum_exp)
    return sub(shifted, log_sum_exp)


def gelu(self: JetTuple, approximate: str = "none") -> JetTuple:
    """Taylor-mode arithmetic for ``aten.gelu(self, approximate=...)``.

    Composed from registered rules: the exact GELU ``0.5 x (1 + erf(x / sqrt(2)))``
    from ``erf`` / ``mul`` / ``add``, the ``"tanh"`` approximation
    ``0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`` from ``tanh`` / ``pow`` as well.

    Args:
        self: The input and its Taylor coefficients.
        approximate: ``"none"`` (exact) or ``"tanh"``.

    Returns:
        The value and its Taylor coefficients.

    Raises:
        NotImplementedError: If ``approximate`` is neither ``"none"`` nor ``"tanh"``.
    """
    collapsed = _collapsed_of(self)
    mul = _rule(ops.aten.mul.Tensor, collapsed)
    add = _rule(ops.aten.add.Tensor, collapsed)
    if approximate == "none":
        inner = _rule(ops.aten.erf.default, collapsed)(mul(self, 1 / sqrt(2)))
    elif approximate == "tanh":
        pow = _rule(ops.aten.pow.Tensor_Scalar, collapsed)
        cubic = add(self, mul(pow(self, 3), 0.044715))
        inner = _rule(ops.aten.tanh.default, collapsed)(mul(cubic, sqrt(2 / pi)))
    else:
        raise NotImplementedError(f"gelu: unsupported approximate={approximate!r}")
    return mul(mul(self, 0.5), add(inner, 1.0))


def _standardize(input: object, dims: list[int], eps: float, collapsed: bool) -> tuple:
    """Standardize ``input`` over ``dims`` with its own statistics.

    ``(input - mean) * (var + eps)**-0.5`` with biased ``var``, composed from
    the registered ``mean`` / ``sub`` / ``mul`` / ``pow`` / ``add`` rules. The
    shared core of layer / group / instance norm; ``input`` may be a jet or a
    constant.

    Returns:
        ``(out, mean, rstd)``; ``mean`` / ``rstd`` keep ``dims`` as size 1.
    """
    sub = _rule(ops.aten.sub.Tensor, collapsed)
    mul = _rule(ops.aten.mul.Tensor, collapsed)
    add = _rule(ops.aten.add.Tensor, collapsed)
    pow = _rule(ops.aten.pow.Tensor_Scalar, collapsed)
    mean = _rule(ops.aten.mean.dim, collapsed)
    mu = mean(input, dims, keepdim=True)
    centered = sub(input, mu)
    rstd = pow(add(mean(pow(centered, 2), dims, keepdim=True), eps), -0.5)
    return mul(centered, rstd), mu, rstd


def _affine(out: object, weight: object, bias: object, collapsed: bool) -> object:
    """``out * weight + bias`` for optional (``None`` = absent) broadcastable params."""
    if weight is not None:
        out = _rule(ops.aten.mul.Tensor, collapsed)(out, weight)
    if bias is not None:
        out = _rule(ops.aten.add.Tensor, collapsed)(out, bias)
    return out


def native_batch_norm(
    input: object,
    weight: object,
    bias: object,
    running_mean: object,
    running_var: object,
    training: bool,
    momentum: float,
    eps: float,
) -> tuple:
    """Taylor-mode arithmetic for ``aten.native_batch_norm`` (eval mode).

    Eval-mode batch norm normalizes each channel with the frozen running
    statistics: ``(input - running_mean) / sqrt(running_var + eps) * weight +
    bias``. Any of ``input`` / ``weight`` / ``bias`` may be a jet, a constant, or
    (``weight`` / ``bias``) ``None`` -- any input rank (1d/2d/3d batch norm) is
    supported. Training mode is deferred until PyTorch fixes its fused
    ``native_batch_norm``'s incorrect higher-order autograd in training
    (pytorch/pytorch#186256), without which a training rule cannot be validated.

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

    collapsed = _collapsed_of(input, weight, bias)
    sub = _rule(ops.aten.sub.Tensor, collapsed)
    mul = _rule(ops.aten.mul.Tensor, collapsed)
    view = _rule(ops.aten.view.default, collapsed)

    primal = input[0] if isinstance(input, JetTuple) else input
    shape = _bn_channel_view(primal)
    rstd = (running_var + eps).rsqrt()
    out = sub(input, view(running_mean, shape))
    out = mul(out, view(rstd, shape))
    weight = None if weight is None else view(weight, shape)
    bias = None if bias is None else view(bias, shape)
    out = _affine(out, weight, bias, collapsed)
    empty = primal.new_empty(0)
    return out, empty, empty


def native_layer_norm(
    input: object, normalized_shape: list[int], weight: object, bias: object, eps: float
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
    collapsed = _collapsed_of(input, weight, bias)
    dims = list(range(-len(normalized_shape), 0))
    out, mu, rstd = _standardize(input, dims, eps, collapsed)
    out = _affine(out, weight, bias, collapsed)
    # The ATen op's output is contiguous (ConvNeXt ``view``s it after a permute).
    if not out[0].is_contiguous():
        clone = _rule(ops.aten.clone.default, collapsed)
        out = clone(out, memory_format=contiguous_format)
    return out, mu, rstd


def _fused_rms_norm(
    input: object, normalized_shape: list[int], weight: object, eps: float | None
) -> tuple:
    """Taylor-mode arithmetic for ``aten._fused_rms_norm``.

    On CUDA / MPS, ``nn.RMSNorm`` lowers to this fused op (on CPU, to ``pow`` /
    ``mean`` / ``add`` / ``rsqrt`` / ``mul``, which have rules of their own). It
    rescales each sample by its root mean square over the trailing
    ``normalized_shape`` dims, ``input * (mean(input**2) + eps)**-0.5``, then
    applies the optional ``weight`` (:func:`_affine`); ``input`` / ``weight``
    may each be a jet or a constant. ``eps=None`` is the op's default,
    ``finfo(input.dtype).eps``.

    Returns:
        The ATen op's ``(output, rstd)`` pair as a plain tuple of jets (``rstd``
        keeps the normalized dims as size 1, like the op on CPU / CUDA; on MPS
        the op returns ``None`` there), so a downstream ``getitem`` falls
        through to the native op.
    """
    collapsed = _collapsed_of(input, weight)
    mul = _rule(ops.aten.mul.Tensor, collapsed)
    add = _rule(ops.aten.add.Tensor, collapsed)
    pow = _rule(ops.aten.pow.Tensor_Scalar, collapsed)
    mean = _rule(ops.aten.mean.dim, collapsed)
    if eps is None:
        eps = finfo((input[0] if isinstance(input, JetTuple) else input).dtype).eps
    dims = list(range(-len(normalized_shape), 0))
    rstd = pow(add(mean(pow(input, 2), dims, keepdim=True), eps), -0.5)
    return _affine(mul(input, rstd), weight, None, collapsed), rstd


def cudnn_batch_norm(
    input: object,
    weight: object,
    bias: object,
    running_mean: object,
    running_var: object,
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
