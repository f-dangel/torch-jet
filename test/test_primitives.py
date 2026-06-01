"""Per-primitive correctness tests for Taylor mode.

Each row in ``PRIMITIVE_CASES`` exercises **one dispatch branch** of a
primitive registered with :class:`JetInterpreter`. Pointwise primitives
collapse to a single row; non-commutative or operand-order-dependent
primitives (binary ops, matmul, addmm) split into ``_JJ`` (both operands are
jets), ``_JC`` (left is jet), and ``_CJ`` (right is jet) rows. "Constant"
operands are closed over by the test ``f`` so they enter the captured graph
as frozen constants.

Oracles: ``rev_jet`` (standard) and ``rev_collapsed_jet`` (collapsed). Both
compute Taylor coefficients via nested reverse-mode autograd over the Taylor
path and are independent of the FX-trace + interpreter machinery.

``K`` is sampled at the collapsed-mode floor (``K=2``) and a high order ``K=5``;
intermediate orders exercise the same code paths and don't need their own
coverage at the primitive layer.
"""

from typing import Any

from pytest import mark
from torch import addmm, cos, float64, manual_seed, rand, sigmoid, sin, tanh, tensor

from test.utils import K_AND_MODE, assert_jet_matches_oracle, shape, shapes

# Deterministic constants for ``_JC`` / ``_CJ`` branches. Closed over by
# the test ``f`` and become frozen constants in the captured graph.
manual_seed(0)
_L = rand(3, 4, dtype=float64)  # left operand of mm/addmm when jet is right
_R = rand(4, 5, dtype=float64)  # right operand of mm/addmm when jet is left
_B = rand(3, 5, dtype=float64)  # addmm bias (must be const)
# Tensor constant for ``_CJ`` rows on subtract: ``scalar - jet`` lowers to
# ``aten.rsub.Scalar`` (unregistered), but ``tensor - jet`` stays on
# ``aten.sub.Tensor`` which has the CJ branch we want to exercise.
_SUB = tensor(2.0, dtype=float64)


_UNARY_POINTWISE = {"sin": sin, "cos": cos, "tanh": tanh, "sigmoid": sigmoid}
_UNARY_SHAPES = {"1d": (4,), "2d": (3, 4)}

PRIMITIVE_CASES = [
    # ---- Unary pointwise (cross-product over shapes) ---------------------
    *(
        {"id": f"{name}-{sid}", "f": fn, "args_fn": shape(*dims)}
        for name, fn in _UNARY_POINTWISE.items()
        for sid, dims in _UNARY_SHAPES.items()
    ),
    # ---- Unary with scalar exponent (float + low/high integer) -----------
    # ``pow_int_5`` at ``K=5`` hits the order-equals-exponent edge where the
    # K-th derivative of ``x**5`` vanishes.
    {"id": "pow_float", "f": lambda x: x**2.5, "args_fn": shape(4)},
    {"id": "pow_int_5", "f": lambda x: x**5, "args_fn": shape(4)},
    {"id": "pow_int_10", "f": lambda x: x**10, "args_fn": shape(4)},
    # ---- Binary add (commutative; 3 dispatch branches + aliased) ---------
    {"id": "add_JJ", "f": lambda x, y: x + y, "args_fn": shapes((4,), (4,))},
    {"id": "add_JJ_aliased", "f": lambda x: x + x, "args_fn": shape(4)},
    {"id": "add_JC", "f": lambda x: x + 2.0, "args_fn": shape(4)},
    {"id": "add_CJ", "f": lambda x: 2.0 + x, "args_fn": shape(4)},
    # ---- Binary sub (non-commutative) ------------------------------------
    {"id": "sub_JJ", "f": lambda x, y: x - y, "args_fn": shapes((4,), (4,))},
    {"id": "sub_JC", "f": lambda x: x - 2.0, "args_fn": shape(4)},
    {"id": "sub_CJ", "f": lambda x: _SUB - x, "args_fn": shape(4)},
    # ---- Binary mul (3 dispatch branches + aliased ``x*x``) --------------
    {"id": "mul_JJ", "f": lambda x, y: x * y, "args_fn": shapes((4,), (4,))},
    {"id": "mul_JJ_aliased", "f": lambda x: x * x, "args_fn": shape(4)},
    {"id": "mul_JC", "f": lambda x: x * 3.0, "args_fn": shape(4)},
    {"id": "mul_CJ", "f": lambda x: 3.0 * x, "args_fn": shape(4)},
    # ---- Matrix multiply (non-commutative) -------------------------------
    {"id": "mm_JJ", "f": lambda A, B: A @ B, "args_fn": shapes((3, 4), (4, 5))},
    {"id": "mm_JC", "f": lambda A: A @ _R, "args_fn": shape(3, 4)},
    {"id": "mm_CJ", "f": lambda B: _L @ B, "args_fn": shape(4, 5)},
    # ---- addmm (3 dispatch branches over mat1/mat2; bias must be const) ---
    {
        "id": "addmm_mat1_mat2_jet",
        "f": lambda A, B: addmm(_B, A, B),
        "args_fn": shapes((3, 4), (4, 5)),
    },
    {"id": "addmm_mat1_jet", "f": lambda A: addmm(_B, A, _R), "args_fn": shape(3, 4)},
    {"id": "addmm_mat2_jet", "f": lambda B: addmm(_B, _L, B), "args_fn": shape(4, 5)},
    # ---- Reduction -------------------------------------------------------
    {"id": "sum_dim_0", "f": lambda x: x.sum(0), "args_fn": shape(3, 4)},
    # ---- Shape-only ops --------------------------------------------------
    {"id": "view", "f": lambda x: x.view(-1), "args_fn": shape(3, 4)},
    {"id": "unsqueeze", "f": lambda x: x.unsqueeze(0), "args_fn": shape(4)},
    {"id": "squeeze", "f": lambda x: x.squeeze(0), "args_fn": shape(1, 4)},
]


@mark.parametrize("K, collapsed", K_AND_MODE)
@mark.parametrize("config", PRIMITIVE_CASES, ids=lambda c: c["id"])
def test_primitive(config: dict[str, Any], K: int, collapsed: bool):
    """``jet(primitive)`` matches its mode-specific oracle."""
    assert_jet_matches_oracle(config, K, collapsed)
