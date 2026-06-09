"""Single registry mapping each ATen op to its standard / collapsed jet rules.

``RULES[op]`` is a ``{collapsed_flag: rule}`` dict: ``RULES[op][False]`` is the
standard rule, ``RULES[op][True]`` the collapsed one. The ``True`` key may be
absent (an op supported in standard mode only).
:class:`jet.jet_interpreter.JetInterpreter` indexes ``RULES[target][collapsed]``
per call.

Adding an op is a one-site edit in ``RULES`` below: elementwise unary ops are a
single ``_defelementwise`` entry (both modes from the shared derivative table);
linear and constant-output ops use the per-mode ``_deflinear`` / ``_defzero``
builders; everything else lists its two hand-written rule bodies (``jet_*`` from
:mod:`jet.operations`, ``cjet_*`` from :mod:`jet.collapsed_operations`).
"""

from typing import Callable

from torch import ops

from jet import collapsed_operations as collapsed
from jet import operations as standard


def _defelementwise(deriv_fn: Callable) -> dict:
    """Build the ``{standard, collapsed}`` rules for an elementwise unary op.

    Both modes reuse ``deriv_fn`` (e.g. ``_sin_derivatives``); the standard and
    collapsed combinators differ only in how they propagate the coefficients.
    """
    return {
        False: lambda self: standard._jet_elementwise(self, deriv_fn),
        True: lambda self: collapsed._cjet_elementwise(self, deriv_fn),
    }


def _drop_index_output(rule: Callable) -> Callable:
    """Wrap a rule whose op returns ``(value, indices)`` to keep only the value."""
    return lambda *args, **kwargs: rule(*args, **kwargs)[0]


#: Maps an ``aten`` op overload to a ``{collapsed_flag: rule}`` dict.
RULES: dict = {
    # Elementwise unary (both modes from the shared derivative table)
    ops.aten.sin.default: _defelementwise(standard._sin_derivatives),
    ops.aten.cos.default: _defelementwise(standard._cos_derivatives),
    ops.aten.tanh.default: _defelementwise(standard._tanh_derivatives),
    ops.aten.sigmoid.default: _defelementwise(standard._sigmoid_derivatives),
    ops.aten.relu.default: _defelementwise(standard._relu_derivatives),
    ops.aten.exp.default: _defelementwise(standard._exp_derivatives),
    ops.aten.log.default: _defelementwise(standard._log_derivatives),
    # Structured (one hand-written rule body per mode)
    ops.aten.pow.Tensor_Scalar: {False: standard.jet_pow, True: collapsed.cjet_pow},
    ops.aten.add.Tensor: {False: standard.jet_add, True: collapsed.cjet_add},
    ops.aten.sub.Tensor: {False: standard.jet_sub, True: collapsed.cjet_sub},
    ops.aten.mul.Tensor: {False: standard.jet_mul, True: collapsed.cjet_mul},
    ops.aten.mm.default: {False: standard.jet_mm, True: collapsed.cjet_mm},
    ops.aten.addmm.default: {False: standard.jet_addmm, True: collapsed.cjet_addmm},
    ops.aten.convolution.default: {
        False: standard.jet_convolution,
        True: collapsed.cjet_convolution,
    },
    ops.aten.max_pool2d_with_indices.default: {
        False: standard.jet_max_pool2d_with_indices,
        True: collapsed.cjet_max_pool2d_with_indices,
    },
    ops.aten.cat.default: {False: standard.jet_cat, True: collapsed.cjet_cat},
    ops.aten.mse_loss.default: {
        False: standard.jet_mse_loss,
        True: collapsed.cjet_mse_loss,
    },
    ops.aten.nll_loss_forward.default: {
        False: standard.jet_nll_loss_forward,
        True: collapsed.cjet_nll_loss_forward,
    },
    ops.aten._log_softmax.default: {
        False: standard.jet_log_softmax,
        True: collapsed.cjet_log_softmax,
    },
    ops.aten.native_batch_norm.default: {
        False: standard.jet_native_batch_norm,
        True: collapsed.cjet_native_batch_norm,
    },
}

# ``max_pool2d`` is ``max_pool2d_with_indices`` without the index output (some
# backends emit the fused, indices-free op, e.g. MPS) -- derive it in both modes.
RULES[ops.aten.max_pool2d.default] = {
    flag: _drop_index_output(rule)
    for flag, rule in RULES[ops.aten.max_pool2d_with_indices.default].items()
}

# Linear ops (pointwise-linear, shape-only, reductions): per-mode `_deflinear`.
# ``sum.dim_IntList`` / ``view.default`` are *also* bound to names in
# ``operations`` / ``collapsed_operations`` (``jet_sum`` / ``jet_view`` etc.) for
# reuse inside composite rules, but they register here like any other linear op.
for _prim in (
    ops.aten.neg.default,
    ops.aten.div.Scalar,
    ops.aten.t.default,
    ops.aten._unsafe_view.default,
    ops.aten.unsqueeze.default,
    ops.aten.squeeze.dim,
    ops.aten.squeeze.dims,
    ops.aten.sum.default,
    ops.aten.sum.dim_IntList,
    ops.aten.view.default,
    ops.aten._adaptive_avg_pool2d.default,
    ops.aten.avg_pool2d.default,
    ops.aten.mean.default,
    ops.aten.mean.dim,
):
    RULES[_prim] = {
        False: standard._deflinear(_prim),
        True: collapsed._deflinear(_prim),
    }

# Constant-output ops: per-mode `_defzero`.
RULES[ops.aten.zeros_like.default] = {
    False: standard._defzero(ops.aten.zeros_like.default),
    True: collapsed._defzero(ops.aten.zeros_like.default),
}
