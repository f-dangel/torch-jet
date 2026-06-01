"""Per-primitive correctness tests for Taylor mode.

Each row in ``PRIMITIVE_CASES`` exercises one dispatch branch of a primitive
registered with :class:`JetInterpreter`.
"""

from typing import Any

from pytest import mark
from torch import addmm, cos, manual_seed, ops, rand, sigmoid, sin, tanh, tensor, zeros_like

from test.utils import (
    DEVICES,
    K_AND_MODE,
    _stateless,
    assert_jet_matches_oracle,
    device_kw,
)


def _consts(device: str) -> dict[str, Any]:
    """Build frozen constants on ``device``.

    Keys: ``L`` (3, 4), ``R`` (4, 5) -- mm/addmm operands; ``B`` (3, 5) --
    addmm bias; ``SUB`` -- scalar tensor for ``tensor - jet`` (which lowers
    to ``aten.sub.Tensor`` rather than the unregistered ``aten.rsub.Scalar``).
    """
    manual_seed(0)
    kw = device_kw(device)
    return {
        "L": rand(3, 4, **kw),
        "R": rand(4, 5, **kw),
        "B": rand(3, 5, **kw),
        "SUB": tensor(2.0, **kw),
    }


def _sub_cj(device):
    SUB = _consts(device)["SUB"]
    return lambda x: SUB - x


def _mm_jc(device):
    R = _consts(device)["R"]
    return lambda A: A @ R


def _mm_cj(device):
    L = _consts(device)["L"]
    return lambda B: L @ B


def _addmm_jj(device):
    B = _consts(device)["B"]
    return lambda mat1, mat2: addmm(B, mat1, mat2)


def _addmm_mat1_jet(device):
    cs = _consts(device)
    return lambda mat1: addmm(cs["B"], mat1, cs["R"])


def _addmm_mat2_jet(device):
    cs = _consts(device)
    return lambda mat2: addmm(cs["B"], cs["L"], mat2)


_UNARY_POINTWISE = {"sin": sin, "cos": cos, "tanh": tanh, "sigmoid": sigmoid}
_UNARY_SHAPES = {"1d": (4,), "2d": (3, 4)}

PRIMITIVE_CASES = [
    # ---- Unary pointwise (cross-product over shapes) ---------------------
    *(
        {
            "id": f"{name}-{sid}",
            "f": _stateless(fn),
            "args_fn": lambda dims=dims: (rand(*dims),),
        }
        for name, fn in _UNARY_POINTWISE.items()
        for sid, dims in _UNARY_SHAPES.items()
    ),
    # ---- Unary with scalar exponent (float + low/high integer) -----------
    # ``pow_int_5`` at ``K=5`` hits the order-equals-exponent edge where the
    # K-th derivative of ``x**5`` vanishes.
    {
        "id": "pow_float",
        "f": _stateless(lambda x: x**2.5),
        "args_fn": lambda: (rand(4),),
    },
    {"id": "pow_int_5", "f": _stateless(lambda x: x**5), "args_fn": lambda: (rand(4),)},
    {
        "id": "pow_int_10",
        "f": _stateless(lambda x: x**10),
        "args_fn": lambda: (rand(4),),
    },
    # ---- Binary add (commutative; 3 dispatch branches + aliased) ---------
    {
        "id": "add_JJ",
        "f": _stateless(lambda x, y: x + y),
        "args_fn": lambda: (rand(4), rand(4)),
    },
    {
        "id": "add_JJ_aliased",
        "f": _stateless(lambda x: x + x),
        "args_fn": lambda: (rand(4),),
    },
    {"id": "add_JC", "f": _stateless(lambda x: x + 2.0), "args_fn": lambda: (rand(4),)},
    {"id": "add_CJ", "f": _stateless(lambda x: 2.0 + x), "args_fn": lambda: (rand(4),)},
    # ---- Binary sub (non-commutative) ------------------------------------
    {
        "id": "sub_JJ",
        "f": _stateless(lambda x, y: x - y),
        "args_fn": lambda: (rand(4), rand(4)),
    },
    {"id": "sub_JC", "f": _stateless(lambda x: x - 2.0), "args_fn": lambda: (rand(4),)},
    {"id": "sub_CJ", "f": _sub_cj, "args_fn": lambda: (rand(4),)},
    # ---- Binary mul (3 dispatch branches + aliased ``x*x``) --------------
    {
        "id": "mul_JJ",
        "f": _stateless(lambda x, y: x * y),
        "args_fn": lambda: (rand(4), rand(4)),
    },
    {
        "id": "mul_JJ_aliased",
        "f": _stateless(lambda x: x * x),
        "args_fn": lambda: (rand(4),),
    },
    # ``mul_JJ_diff_rank`` operands have different primal ranks. In collapsed
    # mode this exposes whether the Leibniz cross term aligns the leading
    # direction dim ``R`` of both operands instead of relying on PyTorch
    # broadcasting (which right-aligns and would put one operand's ``R``
    # against a middle primal dim of the other).
    {
        "id": "mul_JJ_diff_rank",
        "f": _stateless(lambda x, y: x * y),
        "args_fn": lambda: (rand(3), rand(2, 3)),
    },
    {"id": "mul_JC", "f": _stateless(lambda x: x * 3.0), "args_fn": lambda: (rand(4),)},
    {"id": "mul_CJ", "f": _stateless(lambda x: 3.0 * x), "args_fn": lambda: (rand(4),)},
    # ---- Matrix multiply (non-commutative) -------------------------------
    {
        "id": "mm_JJ",
        "f": _stateless(lambda A, B: A @ B),
        "args_fn": lambda: (rand(3, 4), rand(4, 5)),
    },
    {"id": "mm_JC", "f": _mm_jc, "args_fn": lambda: (rand(3, 4),)},
    {"id": "mm_CJ", "f": _mm_cj, "args_fn": lambda: (rand(4, 5),)},
    # ---- addmm (3 dispatch branches over mat1/mat2; bias must be const) ---
    {
        "id": "addmm_mat1_mat2_jet",
        "f": _addmm_jj,
        "args_fn": lambda: (rand(3, 4), rand(4, 5)),
    },
    {"id": "addmm_mat1_jet", "f": _addmm_mat1_jet, "args_fn": lambda: (rand(3, 4),)},
    {"id": "addmm_mat2_jet", "f": _addmm_mat2_jet, "args_fn": lambda: (rand(4, 5),)},
    # ---- Reduction -------------------------------------------------------
    {
        "id": "sum_dim_0",
        "f": _stateless(lambda x: x.sum(0)),
        "args_fn": lambda: (rand(3, 4),),
    },
    # ---- Shape-only ops --------------------------------------------------
    {
        "id": "view",
        "f": _stateless(lambda x: x.view(-1)),
        "args_fn": lambda: (rand(3, 4),),
    },
    # ``_unsafe_view`` is an internal aten op emitted by Linear/addmm
    # decompositions; reached only via the explicit overload.
    {
        "id": "_unsafe_view",
        "f": _stateless(lambda x: ops.aten._unsafe_view.default(x, [-1])),
        "args_fn": lambda: (rand(3, 4),),
    },
    {
        "id": "unsqueeze",
        "f": _stateless(lambda x: x.unsqueeze(0)),
        "args_fn": lambda: (rand(4),),
    },
    {
        "id": "squeeze",
        "f": _stateless(lambda x: x.squeeze(0)),
        "args_fn": lambda: (rand(1, 4),),
    },
    # ``squeeze.dims`` is the multi-axis overload (``.squeeze([0, 1])``).
    {
        "id": "squeeze_dims",
        "f": _stateless(lambda x: x.squeeze([0, 1])),
        "args_fn": lambda: (rand(1, 1, 4),),
    },
    # ---- Constant-output ops (zero derivatives at every order) -----------
    {"id": "zeros_like", "f": _stateless(zeros_like), "args_fn": lambda: (rand(3, 4),)},
]


@mark.parametrize("device", DEVICES)
@mark.parametrize("K, collapsed", K_AND_MODE)
@mark.parametrize("config", PRIMITIVE_CASES, ids=lambda c: c["id"])
def test_primitive(config: dict[str, Any], K: int, collapsed: bool, device: str):
    """``jet(primitive)`` matches its mode-specific oracle."""
    assert_jet_matches_oracle(config, K, collapsed, device)
