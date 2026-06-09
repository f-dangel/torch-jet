"""Single registry mapping each ATen op to its standard / collapsed jet rules.

``RULES[op]`` is a ``{collapsed_flag: rule}`` dict: ``RULES[op][False]`` is the
standard rule, ``RULES[op][True]`` the collapsed one.
"""

from typing import Callable

from torch import Tensor, ops

from jet import compositions, primitives
from jet.primitives import JetTuple

#: A jet rule: maps a ``JetTuple`` (and any structural args) to its image.
Rule = Callable[..., JetTuple]


def _defshared(rule: Rule) -> dict[bool, Rule]:
    """Register one mode-agnostic ``rule`` (it reads ``self.collapsed``) under both keys."""
    return {False: rule, True: rule}


def _defelementwise(
    deriv_fn: Callable[[Tensor, int], dict[int, Tensor]],
) -> dict[bool, Rule]:
    """Build the elementwise-unary rule from ``deriv_fn`` (e.g. ``_sin_derivatives``)."""

    def rule(self: JetTuple) -> JetTuple:
        return primitives._elementwise(self, deriv_fn)

    return _defshared(rule)


def _deflinear(prim: Callable) -> dict[bool, Rule]:
    """Build the ``{standard, collapsed}`` rules for a linear op."""
    return {False: primitives._deflinear(prim), True: primitives._cdeflinear(prim)}


def _defzero(prim: Callable) -> dict[bool, Rule]:
    """Build the constant-output rule for a ``prim``.

    The rule is mode-agnostic -- :func:`jet.primitives._defzero` zeros each
    coefficient to its input slot's shape and follows ``self.collapsed``.
    """
    return _defshared(primitives._defzero(prim))


#: Maps an ``aten`` op overload to a ``{collapsed_flag: rule}`` dict.
RULES: dict = {
    # Elementwise unary (both modes from the shared derivative table)
    ops.aten.sin.default: _defelementwise(primitives._sin_derivatives),
    ops.aten.cos.default: _defelementwise(primitives._cos_derivatives),
    ops.aten.tanh.default: _defelementwise(primitives._tanh_derivatives),
    ops.aten.sigmoid.default: _defelementwise(primitives._sigmoid_derivatives),
    ops.aten.relu.default: _defelementwise(primitives._relu_derivatives),
    ops.aten.exp.default: _defelementwise(primitives._exp_derivatives),
    ops.aten.log.default: _defelementwise(primitives._log_derivatives),
    # Power (also elementwise, but the exponent is a call-time arg, so it carries
    # its own rule rather than registering through ``_defelementwise``).
    ops.aten.pow.Tensor_Scalar: _defshared(primitives.jet_pow),
    # Structured (one hand-written rule body per mode)
    ops.aten.add.Tensor: {False: primitives.jet_add, True: primitives.cjet_add},
    ops.aten.sub.Tensor: {False: primitives.jet_sub, True: primitives.cjet_sub},
    ops.aten.mul.Tensor: {False: primitives.jet_mul, True: primitives.cjet_mul},
    ops.aten.mm.default: {False: primitives.jet_mm, True: primitives.cjet_mm},
    ops.aten.max_pool2d_with_indices.default: {
        False: primitives.jet_max_pool2d_with_indices,
        True: primitives.cjet_max_pool2d_with_indices,
    },
    # The fused, indices-free pooling op some backends emit (e.g. MPS).
    ops.aten.max_pool2d.default: {
        False: primitives.jet_max_pool2d,
        True: primitives.cjet_max_pool2d,
    },
    ops.aten.cat.default: {False: primitives.jet_cat, True: primitives.cjet_cat},
    ops.aten.nll_loss_forward.default: {
        False: primitives.jet_nll_loss_forward,
        True: primitives.cjet_nll_loss_forward,
    },
    # Composites -- one mode-agnostic body that pulls its sub-rules from this
    # registry (see :mod:`jet.compositions`).
    ops.aten.addmm.default: _defshared(compositions.addmm),
    ops.aten.convolution.default: _defshared(compositions.convolution),
    ops.aten.mse_loss.default: _defshared(compositions.mse_loss),
    ops.aten._log_softmax.default: _defshared(compositions.log_softmax),
    ops.aten.native_batch_norm.default: _defshared(compositions.native_batch_norm),
}

# Linear ops (pointwise-linear, shape-only, reductions): per-mode `_deflinear`.
# Composite rules (:mod:`jet.compositions`) reuse several of these -- e.g.
# ``sum.dim_IntList`` in ``log_softmax``, ``view.default`` in
# ``native_batch_norm`` -- by pulling them straight from this registry.
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
    RULES[_prim] = _deflinear(_prim)

# Constant-output ops: primal carries the value, coefficients are zero.
RULES[ops.aten.zeros_like.default] = _defzero(ops.aten.zeros_like.default)
