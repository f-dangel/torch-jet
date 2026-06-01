"""Per-primitive correctness tests for Taylor mode.

Each row in ``PRIMITIVE_CASES`` exercises one dispatch branch of a primitive
registered with :class:`JetInterpreter`.
"""

from typing import Any

from pytest import mark
from torch import addmm, cos, manual_seed, rand, sigmoid, sin, tanh, tensor

from test.utils import (
    DEVICES,
    K_AND_MODE,
    assert_jet_matches_oracle,
    dtype_for_device,
)

#: Per-device cache for the deterministic constants closed over by the
#: ``_JC`` / ``_CJ`` dispatch-branch cases.
_CONST_CACHE: dict[str, dict[str, Any]] = {}


def _consts(device: str) -> dict[str, Any]:
    """Build (or retrieve cached) frozen constants on ``device``.

    Keys: ``L`` (3, 4), ``R`` (4, 5) -- mm/addmm operands; ``B`` (3, 5) --
    addmm bias; ``SUB`` -- scalar tensor for ``tensor - jet`` (which lowers
    to ``aten.sub.Tensor`` rather than the unregistered ``aten.rsub.Scalar``).
    """
    if device not in _CONST_CACHE:
        manual_seed(0)
        kw = {"dtype": dtype_for_device(device), "device": device}
        _CONST_CACHE[device] = {
            "L": rand(3, 4, **kw),
            "R": rand(4, 5, **kw),
            "B": rand(3, 5, **kw),
            "SUB": tensor(2.0, **kw),
        }
    return _CONST_CACHE[device]


def _stateless(f):
    """Wrap a device-independent test function as a device-aware builder."""
    return lambda device: f


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
        {"id": f"{name}-{sid}", "f": _stateless(fn), "input_shapes": [dims]}
        for name, fn in _UNARY_POINTWISE.items()
        for sid, dims in _UNARY_SHAPES.items()
    ),
    # ---- Unary with scalar exponent (float + low/high integer) -----------
    # ``pow_int_5`` at ``K=5`` hits the order-equals-exponent edge where the
    # K-th derivative of ``x**5`` vanishes.
    {"id": "pow_float", "f": _stateless(lambda x: x**2.5), "input_shapes": [(4,)]},
    {"id": "pow_int_5", "f": _stateless(lambda x: x**5), "input_shapes": [(4,)]},
    {"id": "pow_int_10", "f": _stateless(lambda x: x**10), "input_shapes": [(4,)]},
    # ---- Binary add (commutative; 3 dispatch branches + aliased) ---------
    {
        "id": "add_JJ",
        "f": _stateless(lambda x, y: x + y),
        "input_shapes": [(4,), (4,)],
    },
    {"id": "add_JJ_aliased", "f": _stateless(lambda x: x + x), "input_shapes": [(4,)]},
    {"id": "add_JC", "f": _stateless(lambda x: x + 2.0), "input_shapes": [(4,)]},
    {"id": "add_CJ", "f": _stateless(lambda x: 2.0 + x), "input_shapes": [(4,)]},
    # ---- Binary sub (non-commutative) ------------------------------------
    {
        "id": "sub_JJ",
        "f": _stateless(lambda x, y: x - y),
        "input_shapes": [(4,), (4,)],
    },
    {"id": "sub_JC", "f": _stateless(lambda x: x - 2.0), "input_shapes": [(4,)]},
    {"id": "sub_CJ", "f": _sub_cj, "input_shapes": [(4,)]},
    # ---- Binary mul (3 dispatch branches + aliased ``x*x``) --------------
    {
        "id": "mul_JJ",
        "f": _stateless(lambda x, y: x * y),
        "input_shapes": [(4,), (4,)],
    },
    {"id": "mul_JJ_aliased", "f": _stateless(lambda x: x * x), "input_shapes": [(4,)]},
    {"id": "mul_JC", "f": _stateless(lambda x: x * 3.0), "input_shapes": [(4,)]},
    {"id": "mul_CJ", "f": _stateless(lambda x: 3.0 * x), "input_shapes": [(4,)]},
    # ---- Matrix multiply (non-commutative) -------------------------------
    {
        "id": "mm_JJ",
        "f": _stateless(lambda A, B: A @ B),
        "input_shapes": [(3, 4), (4, 5)],
    },
    {"id": "mm_JC", "f": _mm_jc, "input_shapes": [(3, 4)]},
    {"id": "mm_CJ", "f": _mm_cj, "input_shapes": [(4, 5)]},
    # ---- addmm (3 dispatch branches over mat1/mat2; bias must be const) ---
    {"id": "addmm_mat1_mat2_jet", "f": _addmm_jj, "input_shapes": [(3, 4), (4, 5)]},
    {"id": "addmm_mat1_jet", "f": _addmm_mat1_jet, "input_shapes": [(3, 4)]},
    {"id": "addmm_mat2_jet", "f": _addmm_mat2_jet, "input_shapes": [(4, 5)]},
    # ---- Reduction -------------------------------------------------------
    {"id": "sum_dim_0", "f": _stateless(lambda x: x.sum(0)), "input_shapes": [(3, 4)]},
    # ---- Shape-only ops --------------------------------------------------
    {"id": "view", "f": _stateless(lambda x: x.view(-1)), "input_shapes": [(3, 4)]},
    {
        "id": "unsqueeze",
        "f": _stateless(lambda x: x.unsqueeze(0)),
        "input_shapes": [(4,)],
    },
    {
        "id": "squeeze",
        "f": _stateless(lambda x: x.squeeze(0)),
        "input_shapes": [(1, 4)],
    },
]


@mark.parametrize("device", DEVICES)
@mark.parametrize("K, collapsed", K_AND_MODE)
@mark.parametrize("config", PRIMITIVE_CASES, ids=lambda c: c["id"])
def test_primitive(config: dict[str, Any], K: int, collapsed: bool, device: str):
    """``jet(primitive)`` matches its mode-specific oracle."""
    assert_jet_matches_oracle(config, K, collapsed, device)
