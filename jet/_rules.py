"""Single registry mapping each ATen op to its standard / collapsed jet rules.

``RULES[op]`` is a ``{collapsed_flag: rule}`` dict: ``RULES[op][False]`` is the
standard rule, ``RULES[op][True]`` the collapsed one.
"""

from typing import Callable

from torch import Tensor, ops

from jet import collapsed_operations as collapsed
from jet import operations as standard
from jet.operations import JetTuple

#: A jet rule: maps a ``JetTuple`` (and any structural args) to its image.
Rule = Callable[..., JetTuple]


def _defelementwise(
    deriv_fn: Callable[[Tensor, int], dict[int, Tensor]],
) -> dict[bool, Rule]:
    """Build the ``{standard, collapsed}`` rules for an elementwise unary op.

    The rule is mode-agnostic -- :func:`jet.operations._elementwise` reads the
    standard/collapsed mode off the jet's ``.collapsed`` flag -- so both keys
    share one callable that reuses ``deriv_fn`` (e.g. ``_sin_derivatives``).
    """

    def rule(self: JetTuple) -> JetTuple:
        return standard._elementwise(self, deriv_fn)

    return {False: rule, True: rule}


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
    # The fused, indices-free pooling op some backends emit (e.g. MPS).
    ops.aten.max_pool2d.default: {
        False: standard.jet_max_pool2d,
        True: collapsed.cjet_max_pool2d,
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
