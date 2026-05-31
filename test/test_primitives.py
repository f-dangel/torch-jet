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

``K`` is sampled at the collapsed-mode floor (``K=2``) and ``K_MAX=5``;
intermediate orders exercise the same code paths and don't need their own
coverage at the primitive layer.
"""

from typing import Any

from pytest import mark
from torch import addmm, float64, manual_seed, rand, sigmoid, sin, tanh, tensor
from torch import cos as torch_cos
from torch.testing import assert_close

import jet
from jet import rev_jet
from test.utils import make_collapsed_jet_args, make_standard_jet_args

K_MAX = 5
K_VALUES = [2, K_MAX]
K_IDS = [f"K={k}" for k in K_VALUES]

# Deterministic constants for ``_JC`` / ``_CJ`` branches. These are closed
# over by the test ``f`` and become frozen constants in the captured graph.
manual_seed(0)
_MM_LEFT_CONST = rand(3, 4).double()  # for ``const @ jet`` rows
_MM_RIGHT_CONST = rand(4, 5).double()  # for ``jet @ const`` rows
_ADDMM_BIAS = rand(3, 5).double()
_ADDMM_LEFT_CONST = rand(3, 4).double()
_ADDMM_RIGHT_CONST = rand(4, 5).double()
# Tensor constant for ``_CJ`` rows on subtract: ``scalar - jet`` lowers to
# ``aten.rsub.Scalar`` (unregistered), but ``tensor - jet`` stays on
# ``aten.sub.Tensor`` which has the CJ branch we want to exercise.
_SUB_CONST = tensor(2.0, dtype=float64)


def _shape(*dims: int):
    """Mock-args factory: one ``rand`` tensor of the given shape in double."""
    return lambda: (rand(*dims).double(),)


def _shapes(*shape_pairs):
    """Mock-args factory: one ``rand`` tensor per shape, all in double."""
    return lambda: tuple(rand(*s).double() for s in shape_pairs)


PRIMITIVE_CASES = [
    # ---- Unary pointwise -------------------------------------------------
    {"id": "sin", "f": sin, "mock_args_fn": _shape(4)},
    {"id": "cos", "f": torch_cos, "mock_args_fn": _shape(4)},
    {"id": "tanh", "f": tanh, "mock_args_fn": _shape(4)},
    {"id": "sigmoid", "f": sigmoid, "mock_args_fn": _shape(4)},
    # ---- Unary with scalar exponent --------------------------------------
    {"id": "pow", "f": lambda x: x**2.5, "mock_args_fn": _shape(4)},
    # ---- Binary add (commutative; still 3 dispatch branches) -------------
    {"id": "add_JJ", "f": lambda x, y: x + y, "mock_args_fn": _shapes((4,), (4,))},
    {"id": "add_JC", "f": lambda x: x + 2.0, "mock_args_fn": _shape(4)},
    {"id": "add_CJ", "f": lambda x: 2.0 + x, "mock_args_fn": _shape(4)},
    # ---- Binary sub (non-commutative) ------------------------------------
    {"id": "sub_JJ", "f": lambda x, y: x - y, "mock_args_fn": _shapes((4,), (4,))},
    {"id": "sub_JC", "f": lambda x: x - 2.0, "mock_args_fn": _shape(4)},
    {"id": "sub_CJ", "f": lambda x: _SUB_CONST - x, "mock_args_fn": _shape(4)},
    # ---- Binary mul ------------------------------------------------------
    {"id": "mul_JJ", "f": lambda x, y: x * y, "mock_args_fn": _shapes((4,), (4,))},
    {"id": "mul_JC", "f": lambda x: x * 3.0, "mock_args_fn": _shape(4)},
    {"id": "mul_CJ", "f": lambda x: 3.0 * x, "mock_args_fn": _shape(4)},
    # ---- Matrix multiply (non-commutative) -------------------------------
    {"id": "mm_JJ", "f": lambda A, B: A @ B, "mock_args_fn": _shapes((3, 4), (4, 5))},
    {"id": "mm_JC", "f": lambda A: A @ _MM_RIGHT_CONST, "mock_args_fn": _shape(3, 4)},
    {"id": "mm_CJ", "f": lambda B: _MM_LEFT_CONST @ B, "mock_args_fn": _shape(4, 5)},
    # ---- addmm (3 dispatch branches over mat1/mat2; bias must be const) ---
    {
        "id": "addmm_mat1_mat2_jet",
        "f": lambda A, B: addmm(_ADDMM_BIAS, A, B),
        "mock_args_fn": _shapes((3, 4), (4, 5)),
    },
    {
        "id": "addmm_mat1_jet",
        "f": lambda A: addmm(_ADDMM_BIAS, A, _ADDMM_RIGHT_CONST),
        "mock_args_fn": _shape(3, 4),
    },
    {
        "id": "addmm_mat2_jet",
        "f": lambda B: addmm(_ADDMM_BIAS, _ADDMM_LEFT_CONST, B),
        "mock_args_fn": _shape(4, 5),
    },
    # ---- Reduction -------------------------------------------------------
    {"id": "sum_dim_0", "f": lambda x: x.sum(0), "mock_args_fn": _shape(3, 4)},
    # ---- Shape-only ops --------------------------------------------------
    {"id": "view", "f": lambda x: x.view(-1), "mock_args_fn": _shape(3, 4)},
    {"id": "unsqueeze", "f": lambda x: x.unsqueeze(0), "mock_args_fn": _shape(4)},
    {"id": "squeeze", "f": lambda x: x.squeeze(0), "mock_args_fn": _shape(1, 4)},
]

PRIMITIVE_IDS = [c["id"] for c in PRIMITIVE_CASES]


@mark.parametrize("K", K_VALUES, ids=K_IDS)
@mark.parametrize("config", PRIMITIVE_CASES, ids=PRIMITIVE_IDS)
def test_primitive_standard(config: dict[str, Any], K: int):
    """jet(primitive) matches rev_jet(primitive) on random inputs."""
    manual_seed(0)
    f = config["f"]
    mock_args = config["mock_args_fn"]()
    args = make_standard_jet_args(mock_args, K)

    jet_out = jet.jet(f, mock_args)(*args)
    rev_out = rev_jet(f)(*args)
    assert_close(jet_out, rev_out)


@mark.parametrize("K", K_VALUES, ids=K_IDS)
@mark.parametrize("config", PRIMITIVE_CASES, ids=PRIMITIVE_IDS)
def test_primitive_collapsed(config: dict[str, Any], K: int):
    """jet(primitive, collapsed=True) matches _uncollapsed_via_vmap oracle."""
    manual_seed(0)
    f = config["f"]
    mock_args = config["mock_args_fn"]()
    args = make_collapsed_jet_args(mock_args, K)

    cjet_out = jet.jet(f, mock_args, collapsed=True)(*args)
    oracle_out = jet._uncollapsed_via_vmap(f, mock_args, randomization=None)(*args)
    assert_close(cjet_out, oracle_out)
