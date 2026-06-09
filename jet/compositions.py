"""Composite jet rules: ATen ops that decompose into other registered ops.

Each rule here is **mode-agnostic** -- a single function that reads the
``collapsed`` flag off its jet input and pulls its sub-rules from the shared
``RULES`` registry (:mod:`jet._rules`), so standard and collapsed Taylor mode
share one body. The primitives a composition wires together (``add``, ``mm``,
``pow``, ...) carry the per-mode arithmetic; the composition only encodes the
algebraic identity (e.g. ``addmm == add(self, mm(mat1, mat2))``).
"""

from typing import Callable

from torch import ops

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
    add = _rule(ops.aten.add.Tensor, collapsed)
    view = _rule(ops.aten.view.default, collapsed)

    primal = input[0] if isinstance(input, JetTuple) else input
    shape = _bn_channel_view(primal)
    rstd = (running_var + eps).rsqrt()
    out = sub(input, view(running_mean, shape))
    out = mul(out, view(rstd, shape))
    if weight is not None:
        out = mul(out, view(weight, shape))
    if bias is not None:
        out = add(out, view(bias, shape))
    empty = primal.new_empty(0)
    return out, empty, empty
