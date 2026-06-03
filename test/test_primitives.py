"""Per-primitive correctness tests for Taylor mode.

Each row in ``PRIMITIVE_CASES`` exercises one dispatch branch of a primitive
registered with :class:`JetInterpreter`.
"""

from typing import Any, Callable

from pytest import mark, raises
from torch import (
    addmm,
    cat,
    cos,
    exp,
    float32,
    log,
    manual_seed,
    ops,
    rand,
    randn,
    relu,
    sigmoid,
    sin,
    tanh,
    tensor,
    zeros_like,
)
from torch.nn.functional import (
    adaptive_avg_pool2d,
    avg_pool2d,
    conv2d,
    log_softmax,
    max_pool2d,
    mse_loss,
    nll_loss,
)

from jet import jet
from jet.collapsed_operations import CollapsedJetTuple, cjet_nll_loss_forward
from jet.operations import JetTuple, jet_nll_loss_forward
from test.utils import (
    K_AND_MODE,
    _stateless,
    assert_jet_matches_oracle,
    class_index_loss,
    device_kw,
    make_jet_args,
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
        # conv2d kernels: CONV_W/CONV_B for a plain conv (in=2, out=4, 3x3);
        # DW_W for a grouped conv (groups=2, in=2, out=4).
        "CONV_W": rand(4, 2, 3, 3, **kw),
        "CONV_B": rand(4, **kw),
        "DW_W": rand(4, 1, 3, 3, **kw),
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


def _mse_loss(reduction: str, shape: tuple[int, ...]) -> Callable[[str], Callable]:
    """Build ``mse_loss(x, target, reduction)`` against a frozen target.

    The target is drawn under ``manual_seed(1)`` so it differs from the
    ``setup_case`` input (seeded with ``0``); otherwise ``x - target`` would be
    zero and the order-1 coefficient ``2 * (x - target) * x'`` would vanish,
    masking bugs in that term.
    """

    def build(device: str) -> Callable:
        manual_seed(1)
        target = rand(*shape, **device_kw(device))
        return lambda x: mse_loss(x, target, reduction=reduction)

    return build


def _conv2d_jc(device):
    cs = _consts(device)
    return lambda x: conv2d(x, cs["CONV_W"], cs["CONV_B"], stride=1, padding=1)


def _conv2d_grouped_jc(device):
    # Depthwise/grouped conv (groups=2, no bias) -- the mobile/ResNet pattern.
    cs = _consts(device)
    return lambda x: conv2d(x, cs["DW_W"], None, stride=2, padding=1, groups=2)


def _conv2d_weight_jet(device):
    # Weight is the Taylor-expanded operand (constant input) -- conv is linear
    # in the kernel just as it is in the input.
    cs = _consts(device)
    X = rand(1, 2, 5, 5, **device_kw(device))
    return lambda w: conv2d(X, w, cs["CONV_B"], stride=1, padding=1)


def _conv2d_input_weight_jet(device):
    # Both operands are jets -> exercises the bilinear Leibniz branch.
    cs = _consts(device)
    return lambda x, w: conv2d(x, w, cs["CONV_B"], stride=1, padding=1)


def _cat_jet_const(device):
    # Concatenate a jet with a constant tensor (along the channel dim). Seed
    # first so the captured constant is deterministic: ``setup_case`` re-seeds
    # only after building ``f`` (mirrors ``_consts``).
    manual_seed(0)
    const = rand(1, 2, 6, 6, **device_kw(device))
    return lambda x: cat([x, const], dim=1)


_UNARY_POINTWISE = {
    "sin": sin,
    "cos": cos,
    "tanh": tanh,
    "sigmoid": sigmoid,
    "exp": exp,
}
_UNARY_SHAPES = {"1d": (4,), "2d": (3, 4)}
_POW_EXPONENTS = {
    "pow_float": 2.5,
    "pow_int_0": 0,
    "pow_int_5": 5,
    "pow_int_10": 10,
    "pow_int_negative": -2.0,
}

# Elementwise binary ops over a broadcasting stress matrix (both operands jets).
# In collapsed mode the batched coefficients carry a leading direction dim ``R``;
# right-aligned broadcasting must not shift it. The same-rank pairs broadcast
# without moving ``R``; the different-rank pairs are the ones that previously
# collided ``R`` with a primal dim (for ``add`` / ``sub`` -- ``mul`` already
# aligned via vmap).
_BROADCAST_BINOPS = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
}
_BROADCAST_PAIRS = [
    ((1, 4), (3, 1)),  # same rank, mutual broadcast
    ((1, 1), (3, 4)),  # same rank, scalar-like broadcast
    ((4,), (3, 4)),  # different rank
    ((3, 4), (4,)),  # different rank (reversed)
    ((4,), (3, 1)),  # different rank, mutual broadcast
    ((3, 1), (4,)),  # different rank, mutual broadcast (reversed)
    ((2, 1, 4), (1, 3, 1)),  # higher rank, same rank, mutual
    ((2, 1, 4), (3, 1)),  # higher rank, different rank
]

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
    # ---- ReLU (piecewise linear; ``randn`` straddles 0 to exercise both
    # the active ``x>0`` mask and the zeroed-out ``x<0`` branch) -----------
    *(
        {
            "id": f"relu-{sid}",
            "f": _stateless(relu),
            "args_fn": lambda dims=dims: (randn(*dims),),
        }
        for sid, dims in _UNARY_SHAPES.items()
    ),
    # ---- log (positive domain; ``rand + 0.5`` keeps inputs in [0.5, 1.5)
    # so the ``1 / x0**k`` derivatives stay well-conditioned) --------------
    *(
        {
            "id": f"log-{sid}",
            "f": _stateless(log),
            "args_fn": lambda dims=dims: (rand(*dims) + 0.5,),
        }
        for sid, dims in _UNARY_SHAPES.items()
    ),
    # ---- Unary with scalar exponent (float + low/high integer) -----------
    # ``pow_int_5`` at ``K=5`` hits the order-equals-exponent edge where the
    # K-th derivative of ``x**5`` vanishes; ``pow_int_0`` is the constant case
    # (every derivative structurally zero). ``pow_int_negative`` exercises a
    # negative integer exponent (its inputs are shifted off zero).
    *(
        {
            "id": pid,
            "f": _stateless(lambda x, p=p: x**p),
            "args_fn": lambda p=p: (rand(4) + 0.5,) if p < 0 else (rand(4),),
        }
        for pid, p in _POW_EXPONENTS.items()
    ),
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
    # ---- Broadcasting stress matrix (binary ops, both operands jets) ------
    # See ``_BROADCAST_PAIRS`` / ``_BROADCAST_BINOPS``.
    *(
        {
            "id": f"{name}_bcast_{'x'.join(map(str, sa))}_{'x'.join(map(str, sb))}",
            "f": _stateless(op),
            "args_fn": lambda sa=sa, sb=sb: (rand(*sa), rand(*sb)),
        }
        for name, op in _BROADCAST_BINOPS.items()
        for sa, sb in _BROADCAST_PAIRS
    ),
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
    # ---- Convolution (bilinear in input/weight; bias is the affine term) --
    {"id": "conv2d", "f": _conv2d_jc, "args_fn": lambda: (rand(1, 2, 5, 5),)},
    {
        "id": "conv2d_grouped",
        "f": _conv2d_grouped_jc,
        "args_fn": lambda: (rand(1, 2, 6, 6),),
    },
    # Jet weight (constant input) and both-jet (bilinear Leibniz) branches.
    {
        "id": "conv2d_weight_jet",
        "f": _conv2d_weight_jet,
        "args_fn": lambda: (rand(4, 2, 3, 3),),
    },
    {
        "id": "conv2d_input_weight_jet",
        "f": _conv2d_input_weight_jet,
        "args_fn": lambda: (rand(1, 2, 5, 5), rand(4, 2, 3, 3)),
    },
    # ---- Max pooling (piecewise linear; gather at the primal's arg-max) ---
    {
        "id": "max_pool2d",
        "f": _stateless(lambda x: max_pool2d(x, kernel_size=2, stride=2)),
        "args_fn": lambda: (rand(1, 2, 6, 6),),
    },
    # Unbatched (C, H, W) input -- exercises the leading-dim gather path.
    {
        "id": "max_pool2d_unbatched",
        "f": _stateless(lambda x: max_pool2d(x, kernel_size=2, stride=2)),
        "args_fn": lambda: (rand(2, 6, 6),),
    },
    # ---- Adaptive average pooling (linear); non-(1,1) output keeps it as
    # ``aten._adaptive_avg_pool2d`` (a (1,1) target lowers to ``mean.dim``) ----
    {
        "id": "adaptive_avg_pool2d",
        "f": _stateless(lambda x: adaptive_avg_pool2d(x, (2, 2))),
        "args_fn": lambda: (rand(1, 2, 6, 6),),
    },
    # ---- mean (linear); ``mean()`` (all dims) lowers to ``aten.mean.default``,
    # ``mean(dim=...)`` to ``aten.mean.dim`` (the global-average-pool pattern) --
    {
        "id": "mean_all",
        "f": _stateless(lambda x: x.mean()),
        "args_fn": lambda: (rand(1, 2, 6, 6),),
    },
    {
        "id": "mean_dim",
        "f": _stateless(lambda x: x.mean(dim=[2, 3])),
        "args_fn": lambda: (rand(1, 2, 6, 6),),
    },
    {
        "id": "mean_dim_keepdim",
        "f": _stateless(lambda x: x.mean(dim=[2, 3], keepdim=True)),
        "args_fn": lambda: (rand(1, 2, 6, 6),),
    },
    # ---- Average pooling (linear) ----------------------------------------
    {
        "id": "avg_pool2d",
        "f": _stateless(lambda x: avg_pool2d(x, kernel_size=2, stride=2)),
        "args_fn": lambda: (rand(1, 2, 6, 6),),
    },
    # ---- Concatenation (linear; jets nested in the operand list) ---------
    {
        "id": "cat_JJ",
        "f": _stateless(lambda x, y: cat([x, y], dim=1)),
        "args_fn": lambda: (rand(1, 2, 6, 6), rand(1, 3, 6, 6)),
    },
    # ``dim=0`` exercises the collapsed-mode ``dim + 1`` shift for a
    # non-negative concat dim (cat_JJ uses dim=1).
    {
        "id": "cat_JJ_dim0",
        "f": _stateless(lambda x, y: cat([x, y], dim=0)),
        "args_fn": lambda: (rand(2, 3, 4), rand(1, 3, 4)),
    },
    # A negative ``dim`` skips the collapsed-mode shift: it already counts from
    # the end, past the leading direction dim of the batched coefficients.
    {
        "id": "cat_JJ_dim_neg",
        "f": _stateless(lambda x, y: cat([x, y], dim=-1)),
        "args_fn": lambda: (rand(1, 2, 6, 6), rand(1, 2, 6, 3)),
    },
    {
        "id": "cat_jet_const",
        "f": _cat_jet_const,
        "args_fn": lambda: (rand(1, 4, 6, 6),),
    },
    # ---- Normalization ---------------------------------------------------
    # ``log_softmax`` couples elements along ``dim`` via logsumexp; the rule
    # composes elementwise exp/log with a linear sum and a broadcast sub.
    # Cover a positive dim, a negative dim, and a 1d (dim-0) case.
    *(
        {
            "id": sid,
            "f": _stateless(lambda x, d=dim: log_softmax(x, dim=d)),
            "args_fn": lambda shape=shape: (rand(*shape),),
        }
        for sid, dim, shape in (
            ("log_softmax_dim1", 1, (3, 4)),
            ("log_softmax_dim_neg1", -1, (3, 4)),
            ("log_softmax_dim0_1d", 0, (4,)),
        )
    ),
    # ---- Loss functions --------------------------------------------------
    # ``nll_loss`` is linear in the (log-prob) input given a frozen integer
    # target, so the same op applies to every coefficient. It returns
    # ``(output, total_weight)``; cover all three reductions and the
    # ``getitem`` that selects ``output``, both with and without per-class
    # weights (the weighted ``mean`` exercises the ``total_weight`` divisor).
    *(
        {
            "id": f"nll_loss_{reduction}" + ("_weighted" if weighted else ""),
            "f": class_index_loss(nll_loss, reduction, 8, 5, weighted=weighted),
            "args_fn": lambda: (rand(8, 5),),
        }
        for reduction in ("none", "mean", "sum")
        for weighted in (False, True)
    ),
    # ---- Reduction -------------------------------------------------------
    # ``sum()`` (no-dim) lowers to ``aten.sum.default``; the dim/keepdim
    # variants all lower to ``aten.sum.dim_IntList``.
    {
        "id": "sum_all",
        "f": _stateless(lambda x: x.sum()),
        "args_fn": lambda: (rand(3, 4),),
    },
    {
        "id": "sum_dim_0",
        "f": _stateless(lambda x: x.sum(0)),
        "args_fn": lambda: (rand(3, 4),),
    },
    {
        "id": "sum_dim_list",
        "f": _stateless(lambda x: x.sum([0, 1])),
        "args_fn": lambda: (rand(3, 4),),
    },
    {
        "id": "sum_keepdim",
        "f": _stateless(lambda x: x.sum(0, keepdim=True)),
        "args_fn": lambda: (rand(3, 4),),
    },
    {
        "id": "sum_dim_list_keepdim",
        "f": _stateless(lambda x: x.sum([0, 1], keepdim=True)),
        "args_fn": lambda: (rand(3, 4),),
    },
    # ---- Pointwise-linear ops --------------------------------------------
    {"id": "neg", "f": _stateless(lambda x: -x), "args_fn": lambda: (rand(3, 4),)},
    # ``div.Scalar`` (division by a constant scalar) is reached only via the
    # explicit overload; ``x / 2.0`` lowers to ``aten.div.Tensor``.
    {
        "id": "div_scalar",
        "f": _stateless(lambda x: ops.aten.div.Scalar(x, 2.0)),
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
    # ---- Loss functions --------------------------------------------------
    # ``mse_loss`` composes ``sub`` + ``pow`` then a linear reduction, against
    # a frozen target (the common loss usage). Cover all three reduction
    # enums: none (elementwise), mean, sum.
    *(
        {
            "id": f"mse_loss_{reduction}",
            "f": _mse_loss(reduction, (3, 4)),
            "args_fn": lambda: (rand(3, 4),),
        }
        for reduction in ("none", "mean", "sum")
    ),
    # Both prediction and target are jets -> the ``sub`` operand-pair branch.
    # The two ``rand`` draws differ, so ``x - target`` (and its order-1
    # coefficient) stays nonzero.
    *(
        {
            "id": f"mse_loss_jet_target_{reduction}",
            "f": _stateless(
                lambda x, target, reduction=reduction: mse_loss(
                    x, target, reduction=reduction
                )
            ),
            "args_fn": lambda: (rand(3, 4), rand(3, 4)),
        }
        for reduction in ("none", "mean", "sum")
    ),
    # ---- Constant-output ops (zero derivatives at every order) -----------
    {"id": "zeros_like", "f": _stateless(zeros_like), "args_fn": lambda: (rand(3, 4),)},
    # ``zeros_like_dtype_cast`` guards ``defzero`` against dropping the
    # ``dtype`` kwarg when allocating coefficient slots: the per-coefficient
    # zero tensors must inherit the *output* dtype (``float32``), not the
    # input coefficient's dtype (``float64`` on CPU/CUDA).
    {
        "id": "zeros_like_dtype_cast",
        "f": _stateless(lambda x: zeros_like(x, dtype=float32)),
        "args_fn": lambda: (rand(3, 4),),
    },
]


@mark.parametrize("K, collapsed", K_AND_MODE)
@mark.parametrize("config", PRIMITIVE_CASES, ids=lambda c: c["id"])
def test_primitive(config: dict[str, Any], K: int, collapsed: bool, device: str):
    """``jet(primitive)`` matches its mode-specific oracle."""
    assert_jet_matches_oracle(config, K, collapsed, device)


@mark.parametrize("collapsed", [False, True], ids=["standard", "collapsed"])
def test_conv_taylor_expanded_bias_raises(collapsed: bool, device: str):
    """A Taylor-expanded convolution bias is rejected (bias must be constant)."""
    cs = _consts(device)
    kw = device_kw(device)
    X, W = rand(1, 2, 5, 5, **kw), cs["CONV_W"]

    def f(b):
        return conv2d(X, W, b, stride=1, padding=1)

    bias = rand(4, **kw)
    jet_args = make_jet_args((bias,), K=2, collapsed=collapsed)
    with raises(NotImplementedError, match="Taylor-expanded bias"):
        jet(f, (bias,), collapsed=collapsed)(*jet_args)


@mark.parametrize("collapsed", [False, True], ids=["standard", "collapsed"])
def test_nll_loss_taylor_expanded_target_raises(collapsed: bool, device: str):
    """A Taylor-expanded nll_loss target (label) is rejected (must be constant).

    Labels are class indices, not differentiable; the rule must reject a
    Taylor-expanded target with a clear error rather than a cryptic ATen one.
    """
    kw = device_kw(device)
    rule = cjet_nll_loss_forward if collapsed else jet_nll_loss_forward
    tup = CollapsedJetTuple if collapsed else JetTuple
    logits = tup((rand(8, 5, **kw), rand(8, 5, **kw)))
    target = tup((rand(8, **kw), rand(8, **kw)))  # a Taylor-expanded label
    with raises(NotImplementedError, match="Taylor-expanded target"):
        rule(logits, target, None, 1, -100)
