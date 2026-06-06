"""Per-primitive correctness tests for Taylor mode.

Each row in ``PRIMITIVE_CASES`` exercises one dispatch branch of a primitive
registered with :class:`JetInterpreter`.
"""

from itertools import product
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
    batch_norm,
    conv2d,
    log_softmax,
    max_pool2d,
    mse_loss,
    nll_loss,
)
from torch.testing import assert_close

from jet import _rev_jet, jet
from jet.collapsed_operations import (
    CollapsedJetTuple,
    cjet_native_batch_norm,
    cjet_nll_loss_forward,
)
from jet.operations import JetTuple, jet_native_batch_norm, jet_nll_loss_forward
from test.utils import (
    K_AND_MODE,
    _stateless,
    assert_jet_matches_oracle,
    class_index_loss,
    device_kw,
    make_jet_args,
    rev_collapsed_jet,
    setup_case,
    tolerances_for,
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


#: Bias-shape variants for a jet bias: full ``(3, 5)`` and the row-broadcast
#: 1D ``(5,)`` that must broadcast over the product's rows in every coefficient.
_BIAS_SHAPES = {"full": (3, 5), "bcast": (5,)}


def _matmul_case(bias: str, mat1: str, mat2: str, bias_shape: str) -> dict[str, Any]:
    """Build one ``{id, f, args_fn}`` case for a ``(bias, mat1, mat2)`` combo.

    Each operand code is ``"C"`` (constant) or ``"J"`` (jet); ``bias`` may also
    be ``"N"`` (absent -> plain ``mm``). Jet operands are consumed positionally
    in the order ``bias, mat1, mat2`` so the harness attaches Taylor
    coefficients to them; constants are frozen via :func:`_consts`. A jet bias
    uses ``bias_shape`` -- ``"full"`` ``(3, 5)`` or row-broadcast ``"bcast"``
    ``(5,)``. The id signature is ``<op>_<bias?><mat1><mat2>`` (the absent-bias
    letter is dropped, so plain ``mm`` reads ``mm_JC`` etc.).
    """
    mat1_jet, mat2_jet = mat1 == "J", mat2 == "J"

    def builder(device):
        cs = _consts(device)

        def f(*args: Any):
            it = iter(args)
            b = next(it) if bias == "J" else (cs["B"] if bias == "C" else None)
            m1 = next(it) if mat1_jet else cs["L"]
            m2 = next(it) if mat2_jet else cs["R"]
            return m1 @ m2 if b is None else addmm(b, m1, m2)

        return f

    def args_fn():
        out = []
        if bias == "J":
            out.append(rand(*_BIAS_SHAPES[bias_shape]))
        if mat1_jet:
            out.append(rand(3, 4))
        if mat2_jet:
            out.append(rand(4, 5))
        return tuple(out)

    op = "mm" if bias == "N" else "addmm"
    sig = f"{mat1}{mat2}" if bias == "N" else f"{bias}{mat1}{mat2}"
    suffix = "_bcast" if bias_shape == "bcast" else ""
    return {"id": f"{op}_{sig}{suffix}", "f": builder, "args_fn": args_fn}


def _matmul_cases() -> list[dict[str, Any]]:
    """Exhaustive ``(bias, mat1, mat2)`` dispatch matrix for ``mm`` / ``addmm``.

    ``addmm(bias, mat1, mat2) == bias + mat1 @ mat2``. Each of ``mat1`` /
    ``mat2`` is a constant (``C``) or a jet (``J``); ``bias`` is absent
    (``N`` -> plain ``mm``), constant (``C``), or a jet (``J``). Combos with no
    jet operand are skipped (nothing to differentiate). A jet bias is tested
    both full-shape and row-broadcast 1D -- it is the only operand whose own
    coefficients must broadcast over the product's rows.
    """
    cases = []
    for bias, mat1, mat2 in product("NCJ", "CJ", "CJ"):
        if "J" not in (bias, mat1, mat2):
            continue  # no jet operand -> nothing to differentiate
        for shape in ["full", "bcast"] if bias == "J" else ["full"]:
            cases.append(_matmul_case(bias, mat1, mat2, shape))
    return cases


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


# Convolution shapes -- single source of truth. A plain
# conv maps C_IN -> C_OUT channels with a K x K kernel over an H x W image; the
# grouped variant uses GROUPS (depthwise-style weight (C_OUT, C_IN/GROUPS, K, K)).
# Every conv shape derives from these four numbers, so changing the channel
# count (or kernel/groups) touches exactly one place.
_C_IN, _C_OUT, _K, _GROUPS = 2, 4, 3, 2
_CONV_SHAPES = {
    "x": (1, _C_IN, 5, 5),  # input (N, C_in, H, W)
    "x_grouped": (1, _C_IN, 6, 6),  # input for the strided grouped conv
    "w": (_C_OUT, _C_IN, _K, _K),  # weight (C_out, C_in, kH, kW)
    "dw": (_C_OUT, _C_IN // _GROUPS, _K, _K),  # grouped weight
    "b": (_C_OUT,),  # bias (C_out,)
}


def _conv_consts(device: str) -> dict[str, Any]:
    """Frozen conv operands on ``device`` (one tensor per ``_CONV_SHAPES`` entry)."""
    manual_seed(0)
    kw = device_kw(device)
    return {name: rand(*shape, **kw) for name, shape in _CONV_SHAPES.items()}


def _conv_case(input: str, weight: str, bias: str) -> dict[str, Any]:
    """Build one ``{id, f, args_fn}`` case for an ``(input, weight, bias)`` combo.

    Each operand code is ``"C"`` (constant) or ``"J"`` (jet); ``bias`` may also be
    ``"N"`` (absent). Jet operands are consumed positionally in the order
    ``input, weight, bias`` so the harness attaches Taylor coefficients to them;
    constants are frozen via :func:`_conv_consts`. All combos use a plain
    stride-1, padding-1 conv. The id signature is ``conv2d_<input><weight><bias>``
    (the absent-bias letter is dropped, so a bias-free conv reads ``conv2d_JC``).
    """
    input_jet, weight_jet, bias_jet = input == "J", weight == "J", bias == "J"

    def builder(device):
        c = _conv_consts(device)

        def f(*args: Any):
            it = iter(args)
            x = next(it) if input_jet else c["x"]
            w = next(it) if weight_jet else c["w"]
            b = next(it) if bias_jet else (c["b"] if bias == "C" else None)
            return conv2d(x, w, b, stride=1, padding=1)

        return f

    def args_fn():
        out = []
        if input_jet:
            out.append(rand(*_CONV_SHAPES["x"]))
        if weight_jet:
            out.append(rand(*_CONV_SHAPES["w"]))
        if bias_jet:
            out.append(rand(*_CONV_SHAPES["b"]))
        return tuple(out)

    sig = f"{input}{weight}" + ("" if bias == "N" else bias)
    return {"id": f"conv2d_{sig}", "f": builder, "args_fn": args_fn}


def _conv2d_grouped_jet(device):
    """Depthwise/grouped conv (groups=2, no bias) -- the mobile/ResNet pattern."""
    c = _conv_consts(device)
    return lambda x: conv2d(x, c["dw"], None, stride=2, padding=1, groups=_GROUPS)


def _conv_cases() -> list[dict[str, Any]]:
    """Exhaustive ``(input, weight, bias)`` dispatch matrix for ``convolution``.

    Conv is bilinear in ``input`` / ``weight`` with an affine ``bias``. Each of
    ``input`` / ``weight`` is a constant (``C``) or a jet (``J``); ``bias`` is
    absent (``N`` -> bias-free conv), constant (``C``), or a jet (``J``). Combos
    with no jet operand are skipped (nothing to differentiate). Plus one grouped
    (depthwise) conv -- a distinct weight shape, stride, and ``groups`` -- to
    exercise the grouped dispatch path.
    """
    cases = [
        _conv_case(input, weight, bias)
        for input, weight, bias in product("CJ", "CJ", "NCJ")
        if "J" in (input, weight, bias)
    ]
    cases.append(
        {
            "id": "conv2d_grouped",
            "f": _conv2d_grouped_jet,
            "args_fn": lambda: (rand(*_CONV_SHAPES["x_grouped"]),),
        }
    )
    return cases


def _cat_jet_const(device):
    # Concatenate a jet with a constant tensor (along the channel dim). Seed
    # first so the captured constant is deterministic: ``setup_case`` re-seeds
    # only after building ``f`` (mirrors ``_consts``).
    manual_seed(0)
    const = rand(1, 2, 6, 6, **device_kw(device))
    return lambda x: cat([x, const], dim=1)


def _bn_pick(args, inp: str, weight: str, bias: str, x_const, w_const, b_const):
    # Resolve input/weight/bias from the positional jet args (order: input,
    # weight, bias). ``inp`` is ``"J"`` (a jet arg) or ``"C"`` (captured
    # constant); ``weight`` / ``bias`` add ``"N"`` (absent). Jets are consumed
    # from ``args`` in order; constants/None come from the captured closures.
    it = iter(args)
    x = next(it) if inp == "J" else x_const
    w = next(it) if weight == "J" else (w_const if weight == "C" else None)
    b = next(it) if bias == "J" else (b_const if bias == "C" else None)
    return x, w, b


def _bn_primals(shape, inp: str, weight: str, bias: str):
    # Jet primals (CPU) for a batch-norm case, in the order input, weight, bias --
    # only the entries flagged ``"J"`` (Taylor-expanded). Seeded apart from the
    # captured constants so the values differ; ``setup_case`` migrates these to
    # the device/dtype.
    manual_seed(1)
    out = []
    if inp == "J":
        out.append(rand(*shape))
    if weight == "J":
        out.append(rand(shape[1]))
    if bias == "J":
        out.append(rand(shape[1]))
    return tuple(out)


def _bn_case(
    inp: str, weight: str, bias: str, shape, eps: float, training: bool = False
) -> dict[str, Any]:
    """Build one ``{id, f, args_fn}`` batch-norm case (eval or training).

    Each operand code is ``"J"`` (jet) or ``"C"`` (constant); ``weight`` /
    ``bias`` may also be ``"N"`` (absent, ``affine=False``). Jet operands are
    consumed positionally in the order input, weight, bias; constants and the
    running statistics are captured.
    """
    C = shape[1]

    def builder(device):
        manual_seed(0)
        kw = device_kw(device)
        x_const = randn(*shape, **kw)
        w_const, b_const = randn(C, **kw), randn(C, **kw)
        rm, rv = randn(C, **kw), rand(C, **kw) + 0.5  # running stats (constant)

        def f(*args):
            x, w, b = _bn_pick(args, inp, weight, bias, x_const, w_const, b_const)
            return batch_norm(x, rm, rv, w, b, training=training, eps=eps)

        return f

    mode = "train" if training else "eval"
    suffix = "_eps" if eps != 1e-5 else ""
    cid = f"batch_norm_{mode}_in{inp}_w{weight}_b{bias}_{len(shape)}d{suffix}"
    return {
        "id": cid,
        "f": builder,
        "args_fn": lambda: _bn_primals(shape, inp, weight, bias),
    }


def _bn_cases(training: bool = False) -> list[dict[str, Any]]:
    """Batch-norm dispatch matrix (see :func:`_bn_case`).

    ``input`` is a jet ``J`` or constant ``C``; weight and bias are coupled by
    BatchNorm's ``affine`` flag -- both learnable (each ``J`` or ``C``) or both
    absent (``N``). ``nn.BatchNorm`` never produces weight-only / bias-only.
    Configs keep at least one jet, at 4D, plus a non-default ``eps`` and the
    non-4D ranks (1d/3d/... batch norm) swept with constant and jet weight/bias
    -- reshaping a *jet* weight/bias to the per-channel view is the
    rank-sensitive path (vmapped over the direction dim ``R`` in collapsed mode).
    """
    affine = (("N", "N"), ("J", "J"), ("J", "C"), ("C", "J"), ("C", "C"))
    ranks = ((4, 3, 7), (4, 3, 2, 3, 3), (4, 3))  # 3d / 5d / 2d
    cases = [
        _bn_case(i, w, b, (4, 3, 5, 5), 1e-5, training)
        for i in ("J", "C")
        for w, b in affine
        if "J" in (i, w, b)
    ]
    cases.append(_bn_case("J", "C", "C", (4, 3, 5, 5), 1e-2, training))  # non-def eps
    cases += [_bn_case("J", "C", "C", s, 1e-5, training) for s in ranks]
    cases += [_bn_case("J", "J", "J", s, 1e-5, training) for s in ranks]
    return cases


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

# Elementwise binary ops over a broadcasting stress matrix, covering every
# dispatch branch: ``JJ`` (both operands jets), ``JC`` (jet + constant), ``CJ``
# (constant + jet). In collapsed mode the batched coefficients carry a leading
# direction dim ``R`` that right-aligned broadcasting must not shift; the JC/CJ
# branches also stress a jet broadcasting *up* to a larger constant.
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


def _binop_case(name, op, sa, sb, variant: str) -> dict[str, Any]:
    """One broadcasting case for ``op`` over operand shapes ``sa``/``sb``.

    ``variant`` selects the dispatch branch: ``"JJ"`` (both jets), ``"JC"``
    (first operand ``sa`` a jet, second ``sb`` a frozen constant), ``"CJ"``
    (first ``sa`` a frozen constant, second ``sb`` a jet). Constants are drawn
    under ``manual_seed(3)`` on the device.
    """
    cid = f"{name}_{'x'.join(map(str, sa))}_{'x'.join(map(str, sb))}_{variant}"
    if variant == "JJ":
        return {
            "id": cid,
            "f": _stateless(op),
            "args_fn": lambda sa=sa, sb=sb: (rand(*sa), rand(*sb)),
        }
    const_shape, jet_shape = (sb, sa) if variant == "JC" else (sa, sb)

    def build(device, op=op, cs=const_shape, variant=variant):
        manual_seed(3)
        c = rand(*cs, **device_kw(device))
        return (lambda x: op(x, c)) if variant == "JC" else (lambda x: op(c, x))

    return {"id": cid, "f": build, "args_fn": lambda js=jet_shape: (rand(*js),)}


def _binop_cases() -> list[dict[str, Any]]:
    """Every (op, shape-pair, dispatch-branch) broadcasting case for add/sub/mul."""
    return [
        _binop_case(name, op, sa, sb, variant)
        for name, op in _BROADCAST_BINOPS.items()
        for sa, sb in _BROADCAST_PAIRS
        for variant in ("JJ", "JC", "CJ")
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
    # ---- Broadcasting stress matrix (add/sub/mul; JJ/JC/CJ branches) ------
    # See ``_binop_cases`` / ``_BROADCAST_PAIRS`` / ``_BROADCAST_BINOPS``.
    *_binop_cases(),
    # ---- Matrix multiply / addmm -----------------------------------------
    # Exhaustive (bias, mat1, mat2) dispatch matrix: bias in {absent -> mm,
    # const, jet}; mat1, mat2 in {const, jet}. Covers every Taylor-expanded-
    # bias path, including a jet bias combined with exactly one jet matrix
    # (bias coefficients added onto a single-matrix product). See
    # ``_matmul_cases``.
    *_matmul_cases(),
    # ---- Convolution (bilinear in input/weight; bias is the affine term) --
    *_conv_cases(),
    # ---- Batch norm (eval; affine per-channel map from running statistics) -
    *_bn_cases(),
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
    # ``t`` (matrix transpose) is emitted by ``Linear`` (``addmm(b, x, W.t())``);
    # it changes the coefficients' shape, so collapsed mode must vmap over R.
    {"id": "t", "f": _stateless(lambda x: x.t()), "args_fn": lambda: (rand(3, 4),)},
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


@mark.parametrize("config", _bn_cases(training=True), ids=lambda c: c["id"])
@mark.parametrize("K, collapsed", K_AND_MODE)
def test_batch_norm_train(config: dict[str, Any], K: int, collapsed: bool, device: str):
    """Training-mode batch norm matches the fused op's autograd up to 2nd order.

    Training normalizes with input-dependent batch statistics. PyTorch's fused
    ``native_batch_norm`` has correct autograd only through 2nd order in training
    (pytorch/pytorch#186256), so the rule is validated against it up to order 2;
    orders >= 3 are pinned by the xfail canary
    :func:`test_batch_norm_train_high_order_diverges`. Same fused-autograd oracle
    convention as the eval cases, just capped where PyTorch is trustworthy.
    """
    f, primals = setup_case(config, device)
    jet_args = make_jet_args(primals, K, collapsed=collapsed)
    actual = jet(f, primals, collapsed=collapsed)(*jet_args)
    oracle = rev_collapsed_jet(f) if collapsed else _rev_jet(f)
    expected = oracle(*jet_args)
    tol = tolerances_for(device)
    for k in range(min(K, 2) + 1):
        assert_close(actual[k], expected[k], **tol)


@mark.xfail(
    reason="fused native_batch_norm training autograd is wrong at order >= 3 "
    "(pytorch/pytorch#186256); xpasses once PyTorch fixes the fused op",
    strict=True,
)
@mark.parametrize("collapsed", [False, True], ids=["standard", "collapsed"])
def test_batch_norm_train_high_order_diverges(collapsed: bool, device: str):
    """Canary: the training rule and the fused op's autograd diverge at order >= 3.

    The fused double-backward saves mean / invstd without autograd history, so its
    >= 3rd-order derivatives are wrong (pytorch/pytorch#186256). Our jet rule is
    correct, so the two DISAGREE -- this test asserts agreement and is expected to
    fail; once PyTorch fixes the fused op it xpasses, flagging that the order-2 cap
    in :func:`test_batch_norm_train` can be lifted.
    """
    config = _bn_case("J", "C", "C", (4, 3, 5, 5), 1e-5, training=True)
    f, primals = setup_case(config, device)
    K = 5
    jet_args = make_jet_args(primals, K, collapsed=collapsed)
    actual = jet(f, primals, collapsed=collapsed)(*jet_args)
    oracle = rev_collapsed_jet(f) if collapsed else _rev_jet(f)
    expected = oracle(*jet_args)
    tol = tolerances_for(device)
    for k in range(3, K + 1):
        assert_close(actual[k], expected[k], **tol)


@mark.parametrize("collapsed", [False, True], ids=["standard", "collapsed"])
def test_batch_norm_eval_without_running_stats_raises(collapsed: bool, device: str):
    """Eval-mode batch norm without running statistics must raise clearly.

    With ``running_mean``/``running_var`` set to ``None``, ATen falls back to
    batch statistics even in eval mode -- the non-affine path the rule does not
    implement. It must raise a clear error rather than a cryptic ``TypeError``
    from ``None + eps``.
    """
    kw = device_kw(device)
    rule = cjet_native_batch_norm if collapsed else jet_native_batch_norm
    tup = CollapsedJetTuple if collapsed else JetTuple
    x = tup((rand(4, 3, 5, 5, **kw), rand(4, 3, 5, 5, **kw)))
    with raises(NotImplementedError, match="running statistics"):
        rule(x, None, None, None, None, False, 0.1, 1e-5)


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
