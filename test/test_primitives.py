"""Per-primitive correctness tests for Taylor mode.

Each row in ``PRIMITIVE_CASES`` exercises **one dispatch branch** of a
primitive registered with :class:`JetInterpreter`. Pointwise primitives
collapse to a single row; non-commutative or operand-order-dependent
primitives (binary ops, matmul, addmm) split into ``_JJ`` (both operands are
jets), ``_JC`` (left is jet), and ``_CJ`` (right is jet) rows. "Constant"
operands are closed over by the test ``f`` so they enter the captured graph
as frozen constants.

Oracles:

- **Standard mode**: ``jet(f, mock_args)`` vs ``rev_jet(f)``, which computes
  Taylor coefficients via nested reverse-mode autograd over the Taylor path
  and is independent of the FX-trace machinery.
- **Collapsed mode**: ``jet(f, mock_args, collapsed=True)`` vs
  ``jet._uncollapsed_via_vmap(f, mock_args, randomization=None)``, which runs
  standard ``jet`` per direction and sums at order ``K``.

``K`` is sampled at the collapsed-mode floor (``K=2``) and a high order ``K=5``;
intermediate orders exercise the same code paths and don't need their own
coverage at the primitive layer.
"""

from typing import Any

from pytest import mark
from torch import addmm, cos, float64, manual_seed, rand, sigmoid, sin, tanh, tensor

from test.utils import K_AND_MODE, assert_jet_matches_oracle, shape, shapes

# Deterministic constants for ``_JC`` / ``_CJ`` branches. These are closed
# over by the test ``f`` and become frozen constants in the captured graph.
manual_seed(0)
_MM_LEFT_CONST = rand(3, 4, dtype=float64)  # for ``const @ jet`` rows
_MM_RIGHT_CONST = rand(4, 5, dtype=float64)  # for ``jet @ const`` rows
_ADDMM_BIAS = rand(3, 5, dtype=float64)
_ADDMM_LEFT_CONST = rand(3, 4, dtype=float64)
_ADDMM_RIGHT_CONST = rand(4, 5, dtype=float64)
# Tensor constant for ``_CJ`` rows on subtract: ``scalar - jet`` lowers to
# ``aten.rsub.Scalar`` (unregistered), but ``tensor - jet`` stays on
# ``aten.sub.Tensor`` which has the CJ branch we want to exercise.
_SUB_CONST = tensor(2.0, dtype=float64)


PRIMITIVE_CASES = [
    # ---- Unary pointwise -------------------------------------------------
    {"id": "sin", "f": sin, "args_fn": shape(4)},
    {"id": "cos", "f": cos, "args_fn": shape(4)},
    {"id": "tanh", "f": tanh, "args_fn": shape(4)},
    {"id": "sigmoid", "f": sigmoid, "args_fn": shape(4)},
    # ---- Unary with scalar exponent --------------------------------------
    {"id": "pow", "f": lambda x: x**2.5, "args_fn": shape(4)},
    # ---- Binary add (commutative; still 3 dispatch branches) -------------
    {"id": "add_JJ", "f": lambda x, y: x + y, "args_fn": shapes((4,), (4,))},
    {"id": "add_JC", "f": lambda x: x + 2.0, "args_fn": shape(4)},
    {"id": "add_CJ", "f": lambda x: 2.0 + x, "args_fn": shape(4)},
    # ---- Binary sub (non-commutative) ------------------------------------
    {"id": "sub_JJ", "f": lambda x, y: x - y, "args_fn": shapes((4,), (4,))},
    {"id": "sub_JC", "f": lambda x: x - 2.0, "args_fn": shape(4)},
    {"id": "sub_CJ", "f": lambda x: _SUB_CONST - x, "args_fn": shape(4)},
    # ---- Binary mul ------------------------------------------------------
    {"id": "mul_JJ", "f": lambda x, y: x * y, "args_fn": shapes((4,), (4,))},
    {"id": "mul_JC", "f": lambda x: x * 3.0, "args_fn": shape(4)},
    {"id": "mul_CJ", "f": lambda x: 3.0 * x, "args_fn": shape(4)},
    # ---- Matrix multiply (non-commutative) -------------------------------
    {"id": "mm_JJ", "f": lambda A, B: A @ B, "args_fn": shapes((3, 4), (4, 5))},
    {"id": "mm_JC", "f": lambda A: A @ _MM_RIGHT_CONST, "args_fn": shape(3, 4)},
    {"id": "mm_CJ", "f": lambda B: _MM_LEFT_CONST @ B, "args_fn": shape(4, 5)},
    # ---- addmm (3 dispatch branches over mat1/mat2; bias must be const) ---
    {
        "id": "addmm_mat1_mat2_jet",
        "f": lambda A, B: addmm(_ADDMM_BIAS, A, B),
        "args_fn": shapes((3, 4), (4, 5)),
    },
    {
        "id": "addmm_mat1_jet",
        "f": lambda A: addmm(_ADDMM_BIAS, A, _ADDMM_RIGHT_CONST),
        "args_fn": shape(3, 4),
    },
    {
        "id": "addmm_mat2_jet",
        "f": lambda B: addmm(_ADDMM_BIAS, _ADDMM_LEFT_CONST, B),
        "args_fn": shape(4, 5),
    },
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
