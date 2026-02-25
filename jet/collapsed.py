"""Collapsed Taylor mode via interpreter-level collapsing.

Instead of propagating R full K-jets then using PullSum graph rewrites,
propagates a single "collapsed jet" with mixed shapes:

  - Coefficient 0 (primal): shape (...)
  - Coefficients 1..K-1: shape (R, ...) -- batched over R directions
  - Coefficient K: shape (...) -- already collapsed (summed over directions)

At each nonlinear operation, the K-th output coefficient is computed as:
  out_K = LINEAR_TERM(in_K_collapsed) + NONLINEAR_TERMS(in_1..K-1).sum(0)
"""

from typing import Callable

from scipy.special import comb
from torch import Tensor, addmm, matmul, mm, ops, relu, zeros_like
from torch.func import vmap
from torch.fx import GraphModule, Interpreter
from torch.utils._pytree import register_pytree_node, tree_flatten, tree_unflatten

from jet.operations import (
    _cos_derivatives,
    _faa_di_bruno,
    _pow_derivatives,
    _sigmoid_derivatives,
    _sin_derivatives,
    _tanh_derivatives,
)
from jet.tracing import capture_graph
from jet.utils import Value, ValueAndCoefficients

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


# ---------------------------------------------------------------------------
# Helpers: apply a linear op with vmap for batched coefficients
# ---------------------------------------------------------------------------


def _apply_linear(jet: CollapsedJetTuple, K: int, op) -> CollapsedJetTuple:
    """Apply a linear *op* to every entry of *jet*, vmapping batched ones."""
    results = [op(jet[0])]
    vop = vmap(op)
    for k in range(1, K + 1):
        results.append(vop(jet[k]) if k < K else op(jet[k]))
    return CollapsedJetTuple(results)


def _apply_linear_coeffs(jet: CollapsedJetTuple, K: int, op) -> tuple:
    """Apply *op* to coefficients 1..K only, vmapping batched ones."""
    vop = vmap(op)
    return tuple(vop(jet[k]) if k < K else op(jet[k]) for k in range(1, K + 1))


# ---------------------------------------------------------------------------
# Collapsed Leibniz rule (for products: mul, mm)
# ---------------------------------------------------------------------------


def _collapsed_leibniz(self, other, K, binary_op):
    """Leibniz product rule with collapsed K-th coefficient.

    For orders 0..K-1: standard Leibniz.
    For order K: linear terms (using collapsed coefficients) +
                 nonlinear terms (using batched coefficients, summed over R).
    """
    s_out = ()
    for k in range(K + 1):
        if k < K:
            term = None
            for j in range(k + 1):
                term_j = comb(k, j, exact=True) * binary_op(self[j], other[k - j])
                term = term_j if term is None else term + term_j
            s_out += (term,)
        else:
            linear = binary_op(self[0], other[K]) + binary_op(self[K], other[0])
            if K >= 2:
                nonlinear = None
                for j in range(1, K):
                    term_j = comb(K, j, exact=True) * binary_op(self[j], other[K - j])
                    nonlinear = term_j if nonlinear is None else nonlinear + term_j
                s_out += (linear + nonlinear.sum(0),)
            else:
                s_out += (linear,)
    return CollapsedJetTuple(s_out)


# ---------------------------------------------------------------------------
# Elementwise nonlinear (shared derivative helpers + collapsed Faà di Bruno)
# ---------------------------------------------------------------------------


def _cjet_elementwise(self, derivative_order, deriv_fn):
    """Generic collapsed elementwise using shared helpers."""
    self0, vs = self[0], self[1:]
    primal, dn = deriv_fn(self0, derivative_order)
    vs_out = _faa_di_bruno(vs, derivative_order, dn, collapsed=True)
    return CollapsedJetTuple((primal, *vs_out))


def cjet_sin(self, *, derivative_order):
    return _cjet_elementwise(self, derivative_order, _sin_derivatives)


def cjet_cos(self, *, derivative_order):
    return _cjet_elementwise(self, derivative_order, _cos_derivatives)


def cjet_tanh(self, *, derivative_order):
    return _cjet_elementwise(self, derivative_order, _tanh_derivatives)


def cjet_sigmoid(self, *, derivative_order):
    return _cjet_elementwise(self, derivative_order, _sigmoid_derivatives)


def cjet_pow(self, exponent, *, derivative_order):
    assert isinstance(exponent, (float, int))
    self0, vs = self[0], self[1:]
    primal, dpow = _pow_derivatives(self0, exponent, derivative_order)
    vs_out = _faa_di_bruno(vs, derivative_order, dpow, collapsed=True)
    return CollapsedJetTuple((primal, *vs_out))


# ---------------------------------------------------------------------------
# Arithmetic (add, sub are linear; mul uses collapsed Leibniz)
# ---------------------------------------------------------------------------


def cjet_add(self, other, *, derivative_order):
    K = derivative_order
    self_is = isinstance(self, CollapsedJetTuple)
    other_is = isinstance(other, CollapsedJetTuple)
    if self_is and other_is:
        return CollapsedJetTuple(self[k] + other[k] for k in range(K + 1))
    elif self_is:
        return CollapsedJetTuple(
            (self[0] + other,) + tuple(self[k] for k in range(1, K + 1))
        )
    else:
        return CollapsedJetTuple(
            (other[0] + self,) + tuple(other[k] for k in range(1, K + 1))
        )


def cjet_sub(self, other, *, derivative_order):
    K = derivative_order
    self_is = isinstance(self, CollapsedJetTuple)
    other_is = isinstance(other, CollapsedJetTuple)
    if self_is and other_is:
        return CollapsedJetTuple(self[k] - other[k] for k in range(K + 1))
    elif self_is:
        return CollapsedJetTuple(
            (self[0] - other,) + tuple(self[k] for k in range(1, K + 1))
        )
    else:
        return CollapsedJetTuple(
            (self - other[0],) + tuple(-other[k] for k in range(1, K + 1))
        )


def cjet_mul(self, other, *, derivative_order):
    K = derivative_order
    self_is = isinstance(self, CollapsedJetTuple)
    other_is = isinstance(other, CollapsedJetTuple)
    if self_is and other_is:
        return _collapsed_leibniz(self, other, K, lambda a, b: a * b)
    elif self_is:
        return CollapsedJetTuple(other * self[k] for k in range(K + 1))
    else:
        return CollapsedJetTuple(self * other[k] for k in range(K + 1))


# ---------------------------------------------------------------------------
# Matrix operations (vmap for one-sided, collapsed Leibniz for both-sided)
# ---------------------------------------------------------------------------


def cjet_mm(self, mat2, *, derivative_order):
    K = derivative_order
    self_is = isinstance(self, CollapsedJetTuple)
    mat2_is = isinstance(mat2, CollapsedJetTuple)
    if self_is and mat2_is:
        return _collapsed_leibniz(self, mat2, K, matmul)
    elif self_is:
        return _apply_linear(self, K, lambda x: mm(x, mat2))
    else:
        return _apply_linear(mat2, K, lambda x: mm(self, x))


def cjet_addmm(bias, mat1, mat2, *, derivative_order):
    K = derivative_order
    mat1_is = isinstance(mat1, CollapsedJetTuple)
    mat2_is = isinstance(mat2, CollapsedJetTuple)
    if mat1_is and mat2_is:
        mm_jet = _collapsed_leibniz(mat1, mat2, K, matmul)
        primal = addmm(bias, mat1[0], mat2[0])
        return CollapsedJetTuple((primal,) + mm_jet[1:])
    elif mat1_is:
        primal = addmm(bias, mat1[0], mat2)
        coeffs = _apply_linear_coeffs(mat1, K, lambda x: mm(x, mat2))
        return CollapsedJetTuple((primal, *coeffs))
    else:
        primal = addmm(bias, mat1, mat2[0])
        coeffs = _apply_linear_coeffs(mat2, K, lambda x: mm(mat1, x))
        return CollapsedJetTuple((primal, *coeffs))


# ---------------------------------------------------------------------------
# Shape / reduction operations (vmap handles batch dim automatically)
# ---------------------------------------------------------------------------


def cjet_view(self, size, *, derivative_order):
    return _apply_linear(
        self, derivative_order, lambda x: ops.aten.view.default(x, size)
    )


def cjet_unsqueeze(self, dim, *, derivative_order):
    return _apply_linear(
        self, derivative_order, lambda x: ops.aten.unsqueeze.default(x, dim)
    )


def cjet_squeeze(self, dim, *, derivative_order):
    return _apply_linear(self, derivative_order, lambda x: ops.aten.squeeze.dim(x, dim))


def cjet_sum(self, dim, keepdim=False, *, derivative_order):
    if keepdim:
        raise NotImplementedError("keepdim=True is not supported.")
    pos = dim[0] if isinstance(dim, list) else dim
    return _apply_linear(self, derivative_order, lambda x: x.sum(pos))


def cjet_mean(self, dim, keepdim=False, *, derivative_order):
    return _apply_linear(
        self, derivative_order, lambda x: ops.aten.mean.dim(x, dim, keepdim)
    )


# ---------------------------------------------------------------------------
# Piecewise linear (relu) -- mask from primal broadcasts to all shapes
# ---------------------------------------------------------------------------


def cjet_relu(self, *, derivative_order):
    K = derivative_order
    primal = relu(self[0])
    mask = (self[0] > 0).to(self[0].dtype)
    coeffs = tuple(self[k] * mask for k in range(1, K + 1))
    return CollapsedJetTuple((primal, *coeffs))


# ---------------------------------------------------------------------------
# Convolution (linear in input; vmap for batched coefficients)
# ---------------------------------------------------------------------------


def cjet_convolution(
    self,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    *,
    derivative_order,
):
    K = derivative_order
    primal = ops.aten.convolution.default(
        self[0],
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    op = lambda x: ops.aten.convolution.default(
        x,
        weight,
        None,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    coeffs = _apply_linear_coeffs(self, K, op)
    return CollapsedJetTuple((primal, *coeffs))


# ---------------------------------------------------------------------------
# Batch normalization (eval mode -- scale broadcasts correctly)
# ---------------------------------------------------------------------------


def cjet_native_batch_norm(
    self,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum,
    eps,
    *,
    derivative_order,
):
    if training:
        raise NotImplementedError("Only eval-mode BatchNorm is supported.")
    K = derivative_order
    bn_result = ops.aten.native_batch_norm.default(
        self[0],
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
    )
    primal_out = bn_result[0]
    invstd = 1.0 / (running_var + eps).sqrt()
    scale = weight * invstd if weight is not None else invstd
    shape = [1] * self[0].ndim
    shape[1] = -1
    scale = scale.reshape(shape)
    coeffs = tuple(self[k] * scale for k in range(1, K + 1))
    return (CollapsedJetTuple((primal_out, *coeffs)), bn_result[1], bn_result[2])


# ---------------------------------------------------------------------------
# MaxPool2d (index-based gather; vmap for batched coefficients)
# ---------------------------------------------------------------------------


def cjet_max_pool2d_with_indices(
    self,
    kernel_size,
    stride=(),
    padding=(0, 0),
    dilation=(1, 1),
    ceil_mode=False,
    *,
    derivative_order,
):
    K = derivative_order
    values, indices = ops.aten.max_pool2d_with_indices.default(
        self[0],
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )
    flat_indices = indices.flatten(2)
    op = lambda x: x.flatten(2).gather(2, flat_indices).view_as(values)
    coeffs = _apply_linear_coeffs(self, K, op)
    return (CollapsedJetTuple((values, *coeffs)), indices)


# ---------------------------------------------------------------------------
# COLLAPSED_MAPPING
# ---------------------------------------------------------------------------

COLLAPSED_MAPPING = {
    # Elementwise nonlinear
    ops.aten.sin.default: cjet_sin,
    ops.aten.cos.default: cjet_cos,
    ops.aten.tanh.default: cjet_tanh,
    ops.aten.sigmoid.default: cjet_sigmoid,
    # Power
    ops.aten.pow.Tensor_Scalar: cjet_pow,
    # Arithmetic
    ops.aten.add.Tensor: cjet_add,
    ops.aten.sub.Tensor: cjet_sub,
    ops.aten.mul.Tensor: cjet_mul,
    # Matrix ops
    ops.aten.mm.default: cjet_mm,
    ops.aten.addmm.default: cjet_addmm,
    # Shape ops
    ops.aten.view.default: cjet_view,
    ops.aten.unsqueeze.default: cjet_unsqueeze,
    ops.aten.squeeze.dim: cjet_squeeze,
    # Reductions
    ops.aten.sum.dim_IntList: cjet_sum,
    ops.aten.mean.dim: cjet_mean,
    # Piecewise linear
    ops.aten.relu.default: cjet_relu,
    # Convolution
    ops.aten.convolution.default: cjet_convolution,
    # Batch normalization
    ops.aten.native_batch_norm.default: cjet_native_batch_norm,
    # MaxPool2d
    ops.aten.max_pool2d_with_indices.default: cjet_max_pool2d_with_indices,
}


# ---------------------------------------------------------------------------
# CollapsedJetInterpreter
# ---------------------------------------------------------------------------


class CollapsedJetInterpreter(Interpreter):
    """Interpreter that propagates CollapsedJetTuples through a traced graph."""

    def __init__(self, module: GraphModule, derivative_order: int):
        super().__init__(module)
        self.derivative_order = derivative_order

    def placeholder(self, target, args, kwargs):
        value = super().placeholder(target, args, kwargs)
        return CollapsedJetTuple(value)

    def call_function(self, target, args, kwargs):
        has_jet_arg = any(isinstance(a, CollapsedJetTuple) for a in args)
        if has_jet_arg:
            if target not in COLLAPSED_MAPPING:
                raise NotImplementedError(f"No collapsed jet rule for {target}.")
            return COLLAPSED_MAPPING[target](
                *args, derivative_order=self.derivative_order
            )
        return super().call_function(target, args, kwargs)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _is_collapsed_or_tensor(x):
    return isinstance(x, (CollapsedJetTuple, Tensor))


def _transpose_collapsed_output(result, derivative_order):
    """Transpose pytree-of-CollapsedJetTuples into tuple-of-pytrees."""
    flat, out_spec = tree_flatten(result, is_leaf=_is_collapsed_or_tensor)
    k = derivative_order + 1
    outputs = []
    for order in range(k):
        flat_order = [
            jt[order]
            if isinstance(jt, CollapsedJetTuple)
            else (jt if order == 0 else zeros_like(jt))
            for jt in flat
        ]
        outputs.append(tree_unflatten(flat_order, out_spec))
    return tuple(outputs)


def collapsed_jet(
    f: Callable[..., Value],
    derivative_order: int,
    mock_args: tuple,
    verbose: bool = False,
) -> Callable[..., ValueAndCoefficients]:
    """Overload f with collapsed Taylor-mode equivalent.

    Same API as ``jet()``, but expects mixed-shape series:
      - series[0..K-2]: tensors with leading batch dim R
      - series[K-1]: tensors without batch dim (collapsed)

    The K-th output coefficient is automatically collapsed (summed over
    directions), so no ``.sum(0)`` or PullSum graph rewrites are needed.
    """
    flat_mocks, in_spec = tree_flatten(mock_args)
    num_leaves = len(flat_mocks)

    def flat_f(*flat_tensors):
        args = tree_unflatten(list(flat_tensors), in_spec)
        return f(*args)

    mod = capture_graph(flat_f, *flat_mocks)
    if verbose:
        print(f"Traced graph:\n{mod.graph}")

    interp = CollapsedJetInterpreter(mod, derivative_order)

    def cjet_f(primals, series):
        flat_primals = tree_flatten(primals)[0]
        flat_series = [tree_flatten(s)[0] for s in series]
        input_tuples = [
            (flat_primals[i], *(fs[i] for fs in flat_series)) for i in range(num_leaves)
        ]
        result = interp.run(*input_tuples)
        all_orders = _transpose_collapsed_output(result, derivative_order)
        return all_orders[0], all_orders[1:]

    return cjet_f


