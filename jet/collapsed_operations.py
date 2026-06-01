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
from torch import Tensor, addmm, matmul, mm, ops, zeros_like
from torch.func import vmap
from torch.utils._pytree import register_pytree_node

from jet.operations import (
    _cos_derivatives,
    _faa_di_bruno,
    _order,
    _pow_derivatives,
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


# ---------------------------------------------------------------------------
# Reduction operations (vmap handles batch dim automatically)
# ---------------------------------------------------------------------------


def cjet_sum(
    self: CollapsedJetTuple,
    dim: list[int] | int,
    keepdim: bool = False,
) -> CollapsedJetTuple:
    """Collapsed jet rule for ``aten.sum``.

    ``dim`` must be an ``int`` or a 1-element ``list[int]`` (multi-dim
    reductions are not supported); a longer list raises ``ValueError``.
    """
    if keepdim:
        raise NotImplementedError("keepdim=True is not supported.")
    (pos,) = (dim,) if isinstance(dim, int) else dim
    return _apply_linear(self, lambda x: x.sum(pos))


# ---------------------------------------------------------------------------
# COLLAPSED_MAPPING
# ---------------------------------------------------------------------------

COLLAPSED_MAPPING: dict = {
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
    # Reductions
    ops.aten.sum.dim_IntList: cjet_sum,
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

    ``prim`` is applied to the primal; coefficients are filled with
    ``zeros_like`` of each input coefficient slot to preserve the per-slot
    shape contract (``(R, *S)`` for ``c_1..c_{K-1}``, ``S`` for ``c_K``).
    """

    def rule(self: CollapsedJetTuple, *args, **kwargs) -> CollapsedJetTuple:
        primal_out = prim(self[0], *args, **kwargs)
        coeffs = [zeros_like(c) for c in self[1:]]
        return CollapsedJetTuple([primal_out, *coeffs])

    COLLAPSED_MAPPING[prim] = rule


# Linear / shape-only ops.
for _prim in (
    ops.aten.view.default,
    ops.aten._unsafe_view.default,
    ops.aten.unsqueeze.default,
    ops.aten.squeeze.dim,
    ops.aten.squeeze.dims,
):
    deflinear(_prim)

# Constant-output ops.
for _prim in (ops.aten.zeros_like.default,):
    defzero(_prim)

del _prim
