"""Per-primitive correctness tests for Taylor mode.

Each row in ``PRIMITIVE_CASES`` exercises one dispatch branch of a primitive
registered with :class:`JetInterpreter`.
"""

from functools import partial
from itertools import product
from typing import Any, Callable
from warnings import catch_warnings, filterwarnings

from pytest import mark, raises, warns
from torch import (
    addmm,
    bfloat16,
    cat,
    cos,
    erf,
    exp,
    finfo,
    float16,
    float32,
    linspace,
    log,
    manual_seed,
    ones_like,
    ops,
    rand,
    randint,
    randn,
    relu,
    rsqrt,
    sigmoid,
    sin,
    stack,
    tanh,
    tensor,
    where,
    zeros_like,
)
from torch import (
    dtype as dtype_t,
)
from torch.nn.functional import (
    adaptive_avg_pool2d,
    avg_pool2d,
    batch_norm,
    conv2d,
    embedding,
    gelu,
    hardsigmoid,
    hardswish,
    hardtanh,
    interpolate,
    layer_norm,
    log_softmax,
    max_pool2d,
    mse_loss,
    nll_loss,
    pad,
    relu6,
    rms_norm,
    scaled_dot_product_attention,
    silu,
    softmax,
)
from torch.testing import assert_close

from jet import compositions, jet, primitives
from jet.compositions import native_batch_norm
from jet.primitives import JetTuple, jet_nll_loss_forward
from test.utils import (
    _stateless,
    assert_jet_matches_oracle,
    class_index_loss,
    device_kw,
    make_jet_args,
    setup_case,
    tolerances_for,
)


def _consts(device: str) -> dict[str, Any]:
    """Build frozen constants on ``device``.

    Keys: ``L`` (3, 4), ``R`` (4, 5) -- mm/addmm operands; ``L3`` (2, 3, 4),
    ``R3`` (2, 4, 5) -- their batched (bmm) counterparts; ``B`` (3, 5) --
    addmm bias; ``SUB`` -- scalar tensor for ``tensor - jet`` (which lowers
    to ``aten.sub.Tensor`` rather than the unregistered ``aten.rsub.Scalar``).
    """
    manual_seed(0)
    kw = device_kw(device)
    return {
        "L": rand(3, 4, **kw),
        "R": rand(4, 5, **kw),
        "L3": rand(2, 3, 4, **kw),
        "R3": rand(2, 4, 5, **kw),
        "B": rand(3, 5, **kw),
        "SUB": tensor(2.0, **kw),
    }


def _sub_cj(device):
    SUB = _consts(device)["SUB"]
    return lambda x: SUB - x


def _div_tensor_const(device):
    # Divide a jet by a frozen (broadcasting) constant tensor divisor; lowers to
    # ``aten.div.Tensor`` like ``x / scalar`` does.
    manual_seed(3)
    c = rand(4, **device_kw(device)) + 0.5  # off zero -> well-conditioned divisor
    return lambda x: x / c


def _const_over_jet(device):
    # Frozen constant numerator over a jet divisor (kept off zero via ``+ 2``);
    # exercises the reciprocal-of-a-jet path with a non-jet numerator.
    manual_seed(3)
    c = rand(3, 4, **device_kw(device)) + 0.5
    return lambda x: c / (x + 2.0)


#: Bias-shape variants for a jet bias: full ``(3, 5)`` and the row-broadcast
#: 1D ``(5,)`` that must broadcast over the product's rows in every coefficient.
_BIAS_SHAPES = {"full": (3, 5), "bcast": (5,)}


def _matmul_case(
    bias: str, mat1: str, mat2: str, bias_shape: str, batched: bool = False
) -> dict[str, Any]:
    """Build one ``{id, f, args_fn}`` case for a ``(bias, mat1, mat2)`` combo.

    Each operand code is ``"C"`` (constant) or ``"J"`` (jet); ``bias`` may also
    be ``"N"`` (absent -> plain ``mm``). Jet operands are consumed positionally
    in the order ``bias, mat1, mat2`` so the harness attaches Taylor
    coefficients to them; constants are frozen via :func:`_consts`. A jet bias
    uses ``bias_shape`` -- ``"full"`` ``(3, 5)`` or row-broadcast ``"bcast"``
    ``(5,)``. The id signature is ``<op>_<bias?><mat1><mat2>`` (the absent-bias
    letter is dropped, so plain ``mm`` reads ``mm_JC`` etc.). ``batched`` uses
    the 3-D operands instead (bias must be ``"N"``); ``m1 @ m2`` then lowers to
    ``bmm`` and the id prefix is ``bmm``.
    """
    mat1_jet, mat2_jet = mat1 == "J", mat2 == "J"
    L, R = ("L3", "R3") if batched else ("L", "R")
    shapes = ((2, 3, 4), (2, 4, 5)) if batched else ((3, 4), (4, 5))

    def builder(device):
        cs = _consts(device)

        def f(*args: Any):
            it = iter(args)
            b = next(it) if bias == "J" else (cs["B"] if bias == "C" else None)
            m1 = next(it) if mat1_jet else cs[L]
            m2 = next(it) if mat2_jet else cs[R]
            return m1 @ m2 if b is None else addmm(b, m1, m2)

        return f

    def args_fn():
        out = []
        if bias == "J":
            out.append(rand(*_BIAS_SHAPES[bias_shape]))
        if mat1_jet:
            out.append(rand(*shapes[0]))
        if mat2_jet:
            out.append(rand(*shapes[1]))
        return tuple(out)

    op = ("bmm" if batched else "mm") if bias == "N" else "addmm"
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
    coefficients must broadcast over the product's rows. The three bias-free
    combos are repeated on 3-D operands, where ``@`` lowers to ``bmm``.
    """
    cases = []
    for bias, mat1, mat2 in product("NCJ", "CJ", "CJ"):
        if "J" not in (bias, mat1, mat2):
            continue  # no jet operand -> nothing to differentiate
        for shape in ["full", "bcast"] if bias == "J" else ["full"]:
            cases.append(_matmul_case(bias, mat1, mat2, shape))
    for mat1, mat2 in ["CJ", "JC", "JJ"]:
        cases.append(_matmul_case("N", mat1, mat2, "full", batched=True))
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


def _stack_jet_const(device):
    # Stack a jet with a constant tensor of equal shape (new leading axis). Seed
    # first so the captured constant is deterministic (mirrors ``_cat_jet_const``).
    manual_seed(0)
    const = rand(2, 3, 4, **device_kw(device))
    return lambda x: stack([x, const], dim=0)


def _norm_pick(args, inp: str, weight: str, bias: str, x_const, w_const, b_const):
    # Resolve input/weight/bias of a normalization case from the positional jet
    # args (order: input, weight, bias). ``inp`` is ``"J"`` (a jet arg) or
    # ``"C"`` (captured constant); ``weight`` / ``bias`` add ``"N"`` (absent).
    # Jets are consumed from ``args`` in order; constants/None come from the
    # captured closures.
    it = iter(args)
    x = next(it) if inp == "J" else x_const
    w = next(it) if weight == "J" else (w_const if weight == "C" else None)
    b = next(it) if bias == "J" else (b_const if bias == "C" else None)
    return x, w, b


def _norm_primals(shape, param_shape, inp: str, weight: str, bias: str):
    # Jet primals (CPU) for a normalization case, in the order input, weight,
    # bias -- only the entries flagged ``"J"`` (Taylor-expanded); the affine
    # parameters have ``param_shape``. Seeded apart from the captured constants
    # so the values differ; ``setup_case`` migrates these to the device/dtype.
    manual_seed(1)
    out = []
    if inp == "J":
        out.append(rand(*shape))
    if weight == "J":
        out.append(rand(*param_shape))
    if bias == "J":
        out.append(rand(*param_shape))
    return tuple(out)


def _norm_case(
    prefix: str,
    call: Callable,
    shape: tuple[int, ...],
    param_shape: tuple[int, ...],
    inp: str,
    weight: str,
    bias: str,
    consts: Callable = lambda kw: (),
    suffix: str = "",
) -> dict[str, Any]:
    """Build one ``{id, f, args_fn}`` normalization case.

    Each operand code is ``"J"`` (jet) or ``"C"`` (constant); ``weight`` /
    ``bias`` may also be ``"N"`` (absent). Jet operands are consumed
    positionally in the order input, weight, bias; constants are captured.
    ``call(x, w, b, *consts(kw))`` applies the normalization, with
    ``consts(kw)`` building further captured constants (e.g. running stats) on
    the device. The affine parameters have ``param_shape``.
    """

    def builder(device):
        manual_seed(0)
        kw = device_kw(device)
        x_const = randn(*shape, **kw)
        w_const, b_const = randn(*param_shape, **kw), randn(*param_shape, **kw)
        extra = consts(kw)

        def f(*args):
            x, w, b = _norm_pick(args, inp, weight, bias, x_const, w_const, b_const)
            return call(x, w, b, *extra)

        return f

    return {
        "id": f"{prefix}_in{inp}_w{weight}_b{bias}{suffix}",
        "f": builder,
        "args_fn": lambda: _norm_primals(shape, param_shape, inp, weight, bias),
    }


def _bn_variant(shape: tuple[int, ...], eps: float = 1e-5) -> dict[str, Any]:
    """Eval batch norm at ``shape`` (see :func:`_norm_cases`).

    Eval-mode batch norm is affine, so the generic oracle (reverse-mode AD of
    the fused op) is exact. The running statistics are closed-over plain
    tensors (never jets), like module buffers.
    """
    C = shape[1]
    return {
        "call": lambda x, w, b, rm, rv: batch_norm(
            x, rm, rv, w, b, training=False, eps=eps
        ),
        "shape": shape,
        "param_shape": (C,),
        "consts": lambda kw: (randn(C, **kw), rand(C, **kw) + 0.5),
        "suffix": f"_{len(shape)}d" + ("_eps" if eps != 1e-5 else ""),
    }


def _ln_variant(shape: tuple[int, ...], nd: int) -> dict[str, Any]:
    """Layer norm over the last ``nd`` dims of ``shape`` (see :func:`_norm_cases`).

    The oracle differentiates the substitute definition from
    ``DifferentiableSubstitutes`` because the fused ``native_layer_norm`` has
    a silently wrong third derivative (pytorch/pytorch#175754).
    """
    normalized = shape[-nd:]
    return {
        "call": lambda x, w, b: layer_norm(x, normalized, w, b),
        "shape": shape,
        "param_shape": normalized,
        "suffix": f"_{nd}d",
    }


def _rms_variant(
    shape: tuple[int, ...], nd: int, eps: float | None = 1e-5
) -> dict[str, Any]:
    """RMS norm over the last ``nd`` dims of ``shape`` (see :func:`_norm_cases`).

    On CUDA / MPS, ``rms_norm`` lowers to the fused ``_fused_rms_norm`` (its own
    rule); on CPU, to ``pow`` / ``mean`` / ``add`` / ``rsqrt`` / ``mul``.
    ``eps=None`` is the op's default (``finfo(dtype).eps``); there is no bias.
    """
    normalized = shape[-nd:]
    return {
        "call": lambda x, w, b: rms_norm(x, normalized, w, eps),
        "shape": shape,
        "param_shape": normalized,
        "suffix": f"_{nd}d" + ("_default_eps" if eps is None else ""),
    }


_BN_RANKS = ({"shape": (4, 3, 7)}, {"shape": (4, 3, 2, 3, 3)}, {"shape": (4, 3)})
# ``nn.BatchNorm`` creates weight and bias together (``affine``), so only the
# both-present and both-absent pairs occur. The non-4D ranks also run with jet
# weight/bias: the rule reshapes them to the per-channel view, and that reshape
# depends on the input rank (and on the direction dim in collapsed mode).
_BN_GRID = (
    (
        ("J", "C"),
        (("N", "N"), ("J", "J"), ("J", "C"), ("C", "J"), ("C", "C")),
        ({"shape": (4, 3, 5, 5)},),
    ),
    (("J",), (("C", "C"),), ({"shape": (4, 3, 5, 5), "eps": 1e-2},)),  # non-default eps
    (("J",), (("C", "C"), ("J", "J")), _BN_RANKS),  # 3d / 5d / 2d
)
# ``nn.LayerNorm`` has weight and bias, weight only (``bias=False``), or
# neither. Normalizing over two dims (ConvNeXt) also runs with jet weight/bias.
_LN_GRID = (
    (
        ("J", "C"),
        (
            ("N", "N"),
            ("J", "N"),
            ("C", "N"),
            ("J", "J"),
            ("J", "C"),
            ("C", "J"),
            ("C", "C"),
        ),
        ({"shape": (2, 8), "nd": 1},),
    ),
    (("J",), (("J", "J"), ("C", "C")), ({"shape": (2, 4, 3, 3), "nd": 2},)),
)
# ``nn.RMSNorm`` has a weight but no bias. Normalizing over two dims also runs
# with jet weight; one case leaves ``eps`` at the op's default.
_RMS_GRID = (
    (("J", "C"), (("N", "N"), ("J", "N"), ("C", "N")), ({"shape": (2, 8), "nd": 1},)),
    (
        ("J",),
        (("J", "N"),),
        ({"shape": (2, 4, 2, 2), "nd": 2}, {"shape": (2, 8), "nd": 1, "eps": None}),
    ),
)


def _norm_cases(
    prefix: str, variant: Callable[..., dict[str, Any]], grid: tuple
) -> list[dict[str, Any]]:
    """Expand a normalization's dispatch ``grid`` into :func:`_norm_case` dicts.

    ``grid`` holds ``(input codes, (weight, bias) pairs, variants)`` rows, e.g.
    ``(("J", "C"), (("J", "J"), ("N", "N")), ({"shape": (2, 8)},))``; their products
    keeping at least one jet are the cases, and ``variant(**v)`` maps a shape
    variant to the ``call``, ``shape``, ``param_shape``, ``consts`` and
    ``suffix`` keywords of :func:`_norm_case`.
    """
    return [
        _norm_case(prefix, inp=i, weight=w, bias=b, **variant(**v))
        for inputs, pairs, variants in grid
        for i in inputs
        for w, b in pairs
        for v in variants
        if "J" in (i, w, b)
    ]


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
    # An integer-valued ``float`` exponent (``2.0``) hits the
    # ``exponent == k`` branch of ``_pow_derivatives`` at ``K >= 2``, where the
    # k-th derivative is ``factorial(exponent)``. ``scipy``'s ``factorial(...,
    # exact=True)`` rejects a ``float`` argument, so the exponent must be coerced
    # to ``int`` first; this case regresses that coercion.
    "pow_float_int": 2.0,
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


def _embedding_case(name: str, shape: tuple[int, ...], **kw) -> dict[str, Any]:
    """``embedding(indices, weight, **kw)`` with a jet weight ``(7, 4)``.

    The constant long ``indices`` of ``shape`` are drawn under ``manual_seed(3)``.
    """

    def build(device):
        manual_seed(3)
        idx = randint(0, 7, shape, device=device)
        return lambda w: embedding(idx, w, **kw)

    return {"id": f"embedding_{name}", "f": build, "args_fn": lambda: (rand(7, 4),)}


def _index_case(cid: str, index: Callable) -> dict[str, Any]:
    """Fancy-indexing case ``x[index(idx)]`` on a ``rand(3, 4)`` input.

    ``idx`` is a constant long index tensor drawn under ``manual_seed(3)`` on
    the device; ``index`` turns it into the subscript (e.g. ``(slice(None), idx)``).
    """

    def build(device):
        manual_seed(3)
        idx = randint(0, 3, (2,), device=device)
        return lambda x: x[index(idx)]

    return {"id": cid, "f": build, "args_fn": lambda: (rand(3, 4),)}


def _binop_cases() -> list[dict[str, Any]]:
    """Every (op, shape-pair, dispatch-branch) broadcasting case for add/sub/mul."""
    return [
        _binop_case(name, op, sa, sb, variant)
        for name, op in _BROADCAST_BINOPS.items()
        for sa, sb in _BROADCAST_PAIRS
        for variant in ("JJ", "JC", "CJ")
    ]


_WHERE_SHAPE = (3, 4)


def _where_case(
    name: str, mask_shape: tuple[int, ...] = _WHERE_SHAPE
) -> dict[str, Any]:
    """``where``/``masked_fill`` over a frozen bool mask of ``mask_shape``.

    ``JJ`` takes two jets, ``JC``/``CJ`` a jet and a frozen constant (either
    side), ``scalar`` a Python-scalar branch, ``masked_fill_tensor`` a 0-d
    constant fill. Mask and constants are drawn under ``manual_seed(3)``.
    """

    def build(device):
        manual_seed(3)
        m = rand(*mask_shape, **device_kw(device)) > 0.5
        c = rand(*_WHERE_SHAPE, **device_kw(device))
        v = tensor(-1.0, **device_kw(device))
        return {
            "where_JJ": lambda x, y: where(m, x, y),
            "where_JC": lambda x: where(m, x, c),
            "where_CJ": lambda x: where(m, c, x),
            "where_scalar": lambda x: where(m, x, -1.0),
            "masked_fill_scalar": lambda x: x.masked_fill(m, -1.0),
            "masked_fill_tensor": lambda x: x.masked_fill(m, v),
        }[name]

    n = 2 if name == "where_JJ" else 1
    cid = name if mask_shape == _WHERE_SHAPE else f"{name}_bcast"
    return {
        "id": cid,
        "f": build,
        "args_fn": lambda: tuple(rand(*_WHERE_SHAPE) for _ in range(n)),
    }


def _sdpa_case(
    cid: str,
    jets: str = "JJJ",
    S: int = 5,
    mask: str | None = None,
    kv_heads: int = 2,
    **kw,
) -> dict[str, Any]:
    """``scaled_dot_product_attention(q, k, v, **kw)`` with ``q`` of shape (2, 2, 3, 4).

    ``jets`` marks which of ``q`` / ``k`` / ``v`` (batch, heads, L=3 / S=5, E=4)
    are jets, the rest are frozen constants; ``mask`` adds a ``"bool"`` (no
    fully masked row) or additive ``"float"`` mask; ``kv_heads`` sets the
    key/value heads (fewer than the two query heads for grouped-query
    attention). Drawn under ``manual_seed(3)``.
    """
    shapes = ((2, 2, 3, 4), (2, kv_heads, S, 4), (2, kv_heads, S, 4))

    def build(device):
        manual_seed(3)
        consts = [rand(*shape, **device_kw(device)) for shape in shapes]
        m = None if mask is None else rand(3, S, **device_kw(device))
        if mask == "bool":
            m = m > 0.5
            m[:, 0] = True

        def f(*args):
            it = iter(args)
            qkv = [next(it) if j == "J" else c for j, c in zip(jets, consts)]
            return scaled_dot_product_attention(*qkv, attn_mask=m, **kw)

        return f

    return {
        "id": cid,
        "f": build,
        "args_fn": lambda: tuple(rand(*s) for s, j in zip(shapes, jets) if j == "J"),
    }


def _mha_sdpa(x):
    """``nn.MultiheadAttention``'s attention block, ``(L, B, H*E)`` in, ``(L*B, H*E)`` out.

    The fused CPU kernel lays its output out like its ``(L, B, H, E)``-ordered
    query, so the ``.contiguous()`` is a no-op the fake-mode trace drops; the
    following ``view`` then relies on the rule reproducing that layout.
    """
    L, B, HE = x.shape
    q = x.view(L, B * 2, HE // 2).transpose(0, 1).view(B, 2, L, HE // 2)
    out = scaled_dot_product_attention(q, q, q)
    return out.permute(2, 0, 1, 3).contiguous().view(L * B, HE)


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
    # ---- Hardtanh (piecewise linear with two kinks; ``4 * randn`` straddles
    # the default bounds ``[-1, 1]`` and ``relu6``'s ``[0, 6]`` to exercise the
    # inner slope and both clamped branches) --------------------------------
    *(
        {
            "id": f"hardtanh-{sid}",
            "f": _stateless(hardtanh),
            "args_fn": lambda dims=dims: (4 * randn(*dims),),
        }
        for sid, dims in _UNARY_SHAPES.items()
    ),
    {
        "id": "hardtanh_bounds",
        "f": _stateless(lambda x: hardtanh(x, -2.0, 3.0)),
        "args_fn": lambda: (4 * randn(3, 4),),
    },
    {"id": "relu6", "f": _stateless(relu6), "args_fn": lambda: (4 * randn(3, 4),)},
    # ---- Hardsigmoid (``relu6(x + 3) / 6``; ``4 * randn`` straddles the kinks
    # at ``-3`` and ``3`` to exercise the inner slope and both clamped
    # branches) -----------------------------------------------------------
    *(
        {
            "id": f"hardsigmoid-{sid}",
            "f": _stateless(hardsigmoid),
            "args_fn": lambda dims=dims: (4 * randn(*dims),),
        }
        for sid, dims in _UNARY_SHAPES.items()
    ),
    # The kinks themselves: autograd's convention there (slope ``0`` at both)
    # must hold on every device.
    {
        "id": "hardsigmoid_kinks",
        "f": _stateless(hardsigmoid),
        "args_fn": lambda: (tensor([-3.0, -1.0, 0.0, 1.0, 3.0]),),
    },
    # ---- Hardswish (quadratic on (-3, 3), kinks at +-3; ``4 * randn`` covers
    # all three branches) and SiLU (smooth ``x * sigmoid(x)``; plain ``randn``
    # like ``sigmoid``, whose float32 Stirling-form derivatives lose precision
    # at high order for large ``|x|``) ---------------------------------------
    *(
        {
            "id": f"{name}-{sid}",
            "f": _stateless(f),
            "args_fn": lambda dims=dims, scale=scale: (scale * randn(*dims),),
        }
        for name, f, scale in (("hardswish", hardswish, 4), ("silu", silu, 1))
        for sid, dims in _UNARY_SHAPES.items()
    ),
    # The kinks themselves: autograd's one-sided convention there (``0`` at
    # ``-3``, ``1`` at ``3``) must hold on every device.
    {
        "id": "hardswish_kinks",
        "f": _stateless(hardswish),
        "args_fn": lambda: (tensor([-3.0, -1.0, 0.0, 1.0, 3.0]),),
    },
    # ---- erf (``2 * randn`` covers the tails) ------------------------------
    *(
        {
            "id": f"erf-{sid}",
            "f": _stateless(erf),
            "args_fn": lambda dims=dims: (2 * randn(*dims),),
        }
        for sid, dims in _UNARY_SHAPES.items()
    ),
    # ---- GELU, exact and tanh (``2 * randn`` covers the tails of the exact
    # variant; plain ``randn`` for the tanh variant, like ``sigmoid`` / ``silu``)
    *(
        {
            "id": f"{name}-{sid}",
            "f": _stateless(f),
            "args_fn": lambda dims=dims, scale=scale: (scale * randn(*dims),),
        }
        for name, f, scale in (
            ("gelu", gelu, 2),
            ("gelu_tanh", lambda x: gelu(x, approximate="tanh"), 1),
        )
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
    # ---- rsqrt (positive domain, shifted like ``log``) --------------------
    *(
        {
            "id": f"rsqrt-{sid}",
            "f": _stateless(rsqrt),
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
    # ---- add.Scalar (called directly: ``x + 0.5`` and functionalized
    # ``add_`` trace to ``add.Tensor``) ------------------------------------
    *(
        {
            "id": f"add_scalar-{sid}",
            "f": _stateless(lambda x: ops.aten.add.Scalar(x, 0.5)),
            "args_fn": lambda dims=dims: (rand(*dims),),
        }
        for sid, dims in _UNARY_SHAPES.items()
    ),
    # ---- Constant padding (``F.pad``; a nonzero fill, cropping via negative
    # padding, and padding two dims) --------------------------------------
    *(
        {
            "id": f"pad_{pid}",
            "f": _stateless(f),
            "args_fn": lambda dims=dims: (rand(*dims),),
        }
        for pid, f, dims in (
            ("value", lambda x: pad(x, (1, 2), value=1.5), (2, 3)),
            ("crop", lambda x: pad(x, (-1, 1)), (2, 3)),
            ("2d", lambda x: pad(x, (1, 0, 0, 2), value=-1.0), (2, 3, 4)),
        )
    ),
    # ---- mul.Scalar (called directly: ``x * 2.0`` traces to ``mul.Tensor``) --
    {
        "id": "mul_scalar",
        "f": _stateless(lambda x: ops.aten.mul.Scalar(x, 2.0)),
        "args_fn": lambda: (rand(3, 4),),
    },
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
    *(_where_case(n) for n in ("where_JJ", "where_JC", "where_CJ", "where_scalar")),
    *(_where_case(n) for n in ("masked_fill_scalar", "masked_fill_tensor")),
    _where_case("where_JJ", (1, 4)),
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
    *_norm_cases("batch_norm", _bn_variant, _BN_GRID),
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
    # ---- Bilinear upsampling (linear); ``size=`` and ``scale_factor=`` both
    # lower to ``aten.upsample_bilinear2d.default`` (the latter with scales) --
    *(
        {
            "id": f"upsample_bilinear_{name}",
            "f": _stateless(lambda x, kw=kw: interpolate(x, mode="bilinear", **kw)),
            "args_fn": lambda: (rand(2, 3, 4, 5),),
        }
        for name, kw in {
            "size": {"size": (7, 9), "align_corners": False},
            "size_align_corners": {"size": (7, 9), "align_corners": True},
            "scale": {"scale_factor": (2.0, 1.5), "align_corners": False},
            "scale_align_corners": {"scale_factor": 2.0, "align_corners": True},
        }.items()
    ),
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
    # ``cat`` skips a 1-D empty operand of any other rank (legacy behaviour);
    # transformers' empty key/value cache is one, ``cat([tensor([]), key], -2)``.
    {
        "id": "cat_jet_empty_const",
        "f": _stateless(lambda x: cat([tensor([]).to(x), x], dim=-2)),
        "args_fn": lambda: (rand(2, 3, 4),),
    },
    # ---- Split (linear; a tuple of jets consumed via ``getitem``) ---------
    # Uneven split (``4 = 3 + 1``) along a negative dim, both pieces returned.
    {
        "id": "split",
        "f": _stateless(lambda x: x.split(3, dim=-1)),
        "args_fn": lambda: (rand(2, 4),),
    },
    # ShuffleNetV2's pattern: ``chunk(2, dim=1)`` lowers to ``split.Tensor``;
    # one half passes through, the other goes through a nonlinearity, and
    # ``cat`` rejoins them, so both ``getitem`` outputs feed the result.
    {
        "id": "chunk_shufflenet",
        "f": _stateless(
            lambda x: cat([x.chunk(2, dim=1)[0], sin(x.chunk(2, dim=1)[1])], dim=1)
        ),
        "args_fn": lambda: (rand(2, 4, 3, 3),),
    },
    # ---- Stack (cat of unsqueeze; jets nested in the operand list) --------
    # ``stack`` inserts a new axis at ``dim`` and concatenates along it. Cover
    # the default ``dim=0`` and a positive ``dim`` (both exercise the
    # collapsed-mode direction-dim shift through ``unsqueeze`` and ``cat``), plus
    # a negative ``dim`` (which skips the shift, counting from the end past the
    # leading direction dim of the batched coefficients).
    *[
        {
            "id": f"stack_JJ_dim{dim}",
            "f": _stateless(lambda x, y, dim=dim: stack([x, y], dim=dim)),
            "args_fn": lambda: (rand(2, 3, 4), rand(2, 3, 4)),
        }
        for dim in (0, 1, -1)
    ],
    # A mixed list: one operand is a frozen constant, the other Taylor-expanded.
    {
        "id": "stack_jet_const",
        "f": _stack_jet_const,
        "args_fn": lambda: (rand(2, 3, 4),),
    },
    # ---- Normalization ---------------------------------------------------
    # ``layer_norm`` couples elements over the normalized dims via the sample
    # statistics; full jet / constant / absent matrix over input, weight, bias.
    *_norm_cases("layer_norm", _ln_variant, _LN_GRID),
    # ``rms_norm`` couples elements via the root mean square (no centering).
    *_norm_cases("rms_norm", _rms_variant, _RMS_GRID),
    # ``log_softmax``/``softmax`` couple elements along ``dim`` via the
    # normalizer; the rules compose exp with a linear sum and a broadcast
    # sub/div. Inputs are scaled so the max-shift matters.
    *(
        {
            "id": f"{name}_{sid}",
            "f": _stateless(lambda x, f=f, d=dim: f(x, dim=d)),
            "args_fn": lambda shape=shape: (4 * randn(*shape),),
        }
        for name, f in (("log_softmax", log_softmax), ("softmax", softmax))
        for dim, shape, sid in (
            (1, (3, 4), "dim1"),
            (-1, (3, 4), "dim_neg1"),
            (0, (4,), "dim0_1d"),
        )
    ),
    # ---- Token embedding (linear in the weight; constant integer indices) --
    _embedding_case("1d", (5,)),
    _embedding_case("2d", (2, 3)),
    _embedding_case("padding_idx", (2, 3), padding_idx=0),
    # ``scaled_dot_product_attention`` composes scaled ``matmul``s, masking, and
    # ``softmax``; every pattern of jet (J) / constant (C) query, key, value
    # meets every option.
    *(
        _sdpa_case(f"sdpa_{jets}{oid}", jets=jets, **kw)
        # ``CCJ``: only ``value`` is a jet, so the attention weights are constants
        for jets in ("JJJ", "JCC", "CCJ", "CJJ", "CJC", "JCJ", "JJC")
        for oid, kw in (
            ("", {}),
            ("_causal", {"is_causal": True}),
            ("_bool_mask", {"mask": "bool"}),
            ("_float_mask", {"mask": "float"}),
            ("_scale", {"scale": 0.3}),
            # Grouped-query attention: one key/value head serves both query heads.
            ("_gqa", {"kv_heads": 1, "enable_gqa": True}),
        )
    ),
    {
        "id": "sdpa_mha_layout",
        "f": _stateless(_mha_sdpa),
        "args_fn": lambda: (rand(3, 2, 8),),
    },
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
    # ``div.Tensor``: division by a constant divisor is linear in the numerator
    # (each coefficient maps ``c -> c / other``). ``x / 2.0`` lowers here, as
    # does ``x / tensor`` against a frozen tensor divisor.
    {
        "id": "div_tensor",
        "f": _stateless(lambda x: x / 2.0),
        "args_fn": lambda: (rand(3, 4),),
    },
    {
        "id": "div_tensor_const",
        "f": _div_tensor_const,
        "args_fn": lambda: (rand(3, 4),),
    },
    # ``div.Tensor`` with a Taylor-expanded divisor: ``a / b == a * b**(-1)``,
    # so the composite reciprocates ``b`` via ``pow`` and multiplies. Covers a
    # jet divisor (``div_JJ``), broadcasting of a lower-rank jet divisor through
    # ``mul`` (``div_JJ_bcast``), and a constant numerator over a jet divisor
    # (``div_const_J``). Divisors are kept off zero so the reciprocal is
    # well-conditioned.
    {
        "id": "div_JJ",
        "f": _stateless(lambda a, b: a / b),
        "args_fn": lambda: (rand(3, 4), rand(3, 4) + 1.0),
    },
    {
        "id": "div_JJ_bcast",
        "f": _stateless(lambda a, b: a / b),
        "args_fn": lambda: (rand(3, 4), rand(4) + 1.0),
    },
    {
        "id": "div_const_J",
        "f": _const_over_jet,
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
    # ``transpose.int`` / ``permute`` are the general axis swaps (``t`` covers
    # only 2d). ``expand`` broadcasts to a larger (view) shape. ``clone`` is what
    # ``.contiguous()`` lowers to after a non-contiguous view (torchvision's
    # ``channel_shuffle``) and carries a ``memory_format`` kwarg.
    {
        "id": "transpose",
        "f": _stateless(lambda x: x.transpose(0, 2)),
        "args_fn": lambda: (rand(2, 3, 4),),
    },
    {
        "id": "permute",
        "f": _stateless(lambda x: x.permute(2, 0, 1)),
        "args_fn": lambda: (rand(2, 3, 4),),
    },
    {
        "id": "expand",
        "f": _stateless(lambda x: x.expand(2, 3, 4)),
        "args_fn": lambda: (rand(1, 4),),
    },
    {
        "id": "repeat_interleave",
        "f": _stateless(lambda x: x.repeat_interleave(2, dim=1)),
        "args_fn": lambda: (rand(2, 3, 4),),
    },
    {
        "id": "clone",
        "f": _stateless(lambda x: x.clone()),
        "args_fn": lambda: (rand(3, 4),),
    },
    {
        "id": "contiguous",
        "f": _stateless(lambda x: x.t().contiguous()),
        "args_fn": lambda: (rand(3, 4),),
    },
    # Transformer views: ``x[0]``/``x[:, 1]`` -> ``select.int``, ``x[1:3]``/
    # ``x[..., ::2]`` -> ``slice.Tensor``, ``x[idx]``/``x[:, idx]`` -> ``index.Tensor``,
    # ``.to(dtype)`` -> ``_to_copy`` (a same-dtype ``.to``, as on MPS, is a no-op),
    # ``x[:, 0:]`` -> ``alias`` (a full slice behind a leading ``:``).
    *(
        {"id": sid, "f": _stateless(fn), "args_fn": lambda: (rand(3, 4),)}
        for sid, fn in (
            ("select_dim0", lambda x: x[0]),
            ("select_dim1", lambda x: x[:, 1]),
            ("slice_rows", lambda x: x[1:3]),
            ("slice_step", lambda x: x[..., ::2]),
            ("to_copy_f32", lambda x: x.to(float32)),
            ("alias", lambda x: x[:, 0:]),
        )
    ),
    _index_case("index_rows", lambda idx: idx),
    _index_case("index_cols", lambda idx: (slice(None), idx)),
    # ``as_strided`` restrides a tensor over the same storage; here into the
    # channels_last strides of a ``[N, C, 1, 1]`` tensor. ``as_strided_`` is the
    # in-place variant that adaptive average pooling emits on a channels_last
    # input (torchvision's ConvNeXt); ``capture_graph`` maps it to ``as_strided``.
    {
        "id": "as_strided",
        "f": _stateless(
            lambda x: ops.aten.as_strided.default(x, [2, 3, 1, 1], [3, 1, 3, 3])
        ),
        "args_fn": lambda: (rand(2, 3, 1, 1),),
    },
    {
        "id": "as_strided_",
        # Permute to NHWC, make contiguous, permute back: logically still NCHW, but
        # with NHWC memory strides; pooling then restrides its output (``as_strided_``).
        "f": _stateless(
            lambda x: adaptive_avg_pool2d(
                x.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2), 1
            )
        ),
        "args_fn": lambda: (rand(2, 3, 4, 4),),
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
    # Autograd seeds its backward pass with ``ones_like(loss)``.
    {"id": "ones_like", "f": _stateless(ones_like), "args_fn": lambda: (rand(3, 4),)},
]


@mark.parametrize("config", PRIMITIVE_CASES, ids=lambda c: c["id"])
def test_primitive(config: dict[str, Any], K: int, collapsed: bool, device: str):
    """``jet(primitive)`` matches its mode-specific oracle."""
    assert_jet_matches_oracle(config, K, collapsed, device)


def test_batch_norm_training_raises(collapsed: bool, device: str):
    """Training-mode batch norm is unsupported and must raise clearly.

    Only eval mode is implemented. Training is deferred until PyTorch fixes its
    fused ``native_batch_norm``'s incorrect higher-order autograd in training
    (pytorch/pytorch#186256): until then the fused op cannot serve as a
    correctness oracle, and we do not maintain a separate manual normalization
    just to validate a training rule. A ``training=True`` call must raise a clear
    error rather than silently mis-normalize.
    """
    manual_seed(0)
    kw = device_kw(device)
    (x,) = (rand(4, 3, 5, 5, **kw),)

    def f(x):
        return batch_norm(x, None, None, None, None, training=True, eps=1e-5)

    jet_args = make_jet_args((x,), 2, collapsed=collapsed)
    with raises(NotImplementedError, match="eval mode only"):
        jet(f, (x,), collapsed=collapsed)(*jet_args)


def test_batch_norm_eval_without_running_stats_raises(collapsed: bool, device: str):
    """Eval-mode batch norm without running statistics must raise clearly.

    With ``running_mean``/``running_var`` set to ``None``, ATen falls back to
    batch statistics even in eval mode -- the non-affine path the rule does not
    implement. It must raise a clear error rather than a cryptic ``TypeError``
    from ``None + eps``.
    """
    kw = device_kw(device)
    tup = partial(JetTuple, collapsed=collapsed)
    x = tup((rand(4, 3, 5, 5, **kw), rand(4, 3, 5, 5, **kw)))
    with raises(NotImplementedError, match="running statistics"):
        native_batch_norm(x, None, None, None, None, False, 0.1, 1e-5)


def test_nll_loss_taylor_expanded_target_raises(collapsed: bool, device: str):
    """A Taylor-expanded nll_loss target (label) is rejected (must be constant).

    Labels are class indices, not differentiable; the rule must reject a
    Taylor-expanded target with a clear error rather than a cryptic ATen one.
    """
    kw = device_kw(device)
    tup = partial(JetTuple, collapsed=collapsed)
    logits = tup((rand(8, 5, **kw), rand(8, 5, **kw)))
    target = tup((rand(8, **kw), rand(8, **kw)))  # a Taylor-expanded label
    with raises(NotImplementedError, match="Taylor-expanded target"):
        jet_nll_loss_forward(logits, target, None, 1, -100)


def test_embedding_padding_idx_warns(collapsed: bool, device: str):
    """``padding_idx >= 0`` on a jet weight warns (autograd would mask that row)."""

    def run(case):
        f, args = setup_case(case, device)
        jet(f, args, collapsed=collapsed)(*make_jet_args(args, 2, collapsed))

    with warns(UserWarning, match="padding_idx"):
        run(_embedding_case("padding_idx", (2, 3), padding_idx=0))
    with catch_warnings():
        filterwarnings("error", message="padding_idx")
        run(_embedding_case("1d", (5,)))


def test_max_pool2d_matches_with_indices(collapsed: bool, device: str):
    """The fused ``aten.max_pool2d`` rule equals the with-indices values jet.

    ``jet_max_pool2d`` is the indices-free variant some backends emit (e.g.
    MPS); CPU/CUDA lower to ``max_pool2d_with_indices``, so it is otherwise
    untested. It must return exactly the values jet of
    :func:`jet_max_pool2d_with_indices`, with the indices output dropped.
    """
    kw = device_kw(device)
    (x,) = make_jet_args((rand(2, 3, 8, 8, **kw),), 3, collapsed=collapsed)
    jet_in = JetTuple(x, collapsed=collapsed)
    pool_args = (2, 2)  # kernel_size, stride

    fused = primitives.jet_max_pool2d(jet_in, *pool_args)
    expected, _ = primitives.jet_max_pool2d_with_indices(jet_in, *pool_args)

    assert fused.collapsed == expected.collapsed
    assert len(fused) == len(expected)
    for got, want in zip(fused, expected):
        assert_close(got, want, **tolerances_for(device))


@mark.parametrize(
    "rule",
    [compositions.softmax, compositions.log_softmax],
    ids=["softmax", "log_softmax"],
)
def test_softmax_half_to_float(rule: Callable, collapsed: bool, device: str):
    """``half_to_float=True`` promotes a half jet to float32, like the ATen op.

    ATen rejects the flag for float input on CPU, so the rule is tested directly.
    """
    K = 3
    (x,) = make_jet_args((rand(3, 4, device=device),), K, collapsed=collapsed)
    x = JetTuple(x, collapsed=collapsed)
    x_half = JetTuple(tuple(c.to(float16) for c in x), collapsed=collapsed)

    got = rule(x_half, -1, True)
    want = rule(x, -1, False)

    for g, w in zip(got, want):
        assert g.dtype == float32
        assert_close(g, w, rtol=1e-3, atol=1e-3)


@mark.parametrize("dtype", [float16, bfloat16], ids=["float16", "bfloat16"])
def test_where_scalar_keeps_dtype(dtype: dtype_t, collapsed: bool, device: str):
    """A scalar operand of ``where``/``masked_fill`` does not upcast a half jet."""
    K = 2
    mask = tensor([[True, False, True, False]] * 3, device=device)
    (x,) = make_jet_args((rand(3, 4, device=device, dtype=dtype),), K, collapsed)
    x = JetTuple(x, collapsed=collapsed)

    for out in (
        primitives.jet_where(mask, x, 0.5),
        primitives.jet_where(mask, -float("inf"), x),
        primitives.jet_masked_fill(x, mask, -float("inf")),
    ):
        assert all(c.dtype == dtype for c in out)


@mark.parametrize(
    "op, adapter, extra",
    [
        (
            ops.aten._scaled_dot_product_efficient_attention.default,
            compositions._scaled_dot_product_efficient_attention,
            (None, False),
        ),
        (
            ops.aten._scaled_dot_product_flash_attention.default,
            compositions._scaled_dot_product_flash_attention,
            (),
        ),
    ],
    ids=["efficient", "flash"],
)
def test_cuda_sdpa_adapter_layout(
    op: Callable, adapter: Callable, extra: tuple, collapsed: bool, device: str
):
    """A CUDA adapter's output has the strides of the kernel's meta (runs anywhere).

    The query is ``nn.MultiheadAttention``'s ``(L, B, H, E)``-ordered one.
    """
    K, (L, B, H, E) = 2, (3, 2, 2, 4)
    x = rand(L, B, H * E, device=device)
    q = x.view(L, B * H, E).transpose(0, 1).view(B, H, L, E)
    want = op(*3 * (q.to("meta"),), *extra)[0].stride()
    jets = (
        JetTuple(j, collapsed=collapsed) for j in make_jet_args(3 * (q,), K, collapsed)
    )
    for c in adapter(*jets, *extra)[0]:
        assert c.stride()[-4:] == want


def test_fused_rms_norm_matches_traced(collapsed: bool, device: str):
    """The fused rule equals the jet of the CPU decomposition (CI only traces the latter)."""
    K = 3
    x, w = make_jet_args(
        (rand(2, 8, device=device), rand(8, device=device)), K, collapsed
    )
    x, w = JetTuple(x, collapsed=collapsed), JetTuple(w, collapsed=collapsed)

    out, rstd = compositions._fused_rms_norm(x, [8], w, None)
    want = jet(lambda x, w: rms_norm(x, (8,), w), (x[0], w[0]), collapsed=collapsed)(
        x, w
    )

    tol = tolerances_for(device)
    for got, exp_ in zip(out, want):
        assert_close(got, exp_, **tol)
    want_rstd = ((x[0] ** 2).mean(-1, keepdim=True) + finfo(float32).eps).rsqrt()
    assert_close(rstd[0], want_rstd, **tol)


def test_div_self_is_exact(collapsed: bool, device: str):
    """The order-1..K coefficients of ``x / x`` are exactly zero.

    The division recurrence yields ``q_1 = (x_1 - x_1) / x_0 = 0`` and so on,
    whereas the composition ``x * x**-1`` (the previous rule) accumulates
    float32 roundoff.
    """
    manual_seed(0)
    x = linspace(0.5, 1.5, 12, dtype=float32, device=device).reshape(3, 4)
    K = 5
    jet_args = make_jet_args((x,), K, collapsed=collapsed)
    coeffs = jet(lambda x: x / x, (x,), collapsed=collapsed)(*jet_args)
    for c in coeffs[1:]:
        assert_close(c, zeros_like(c), atol=0, rtol=0)
    composed = jet(lambda x: x * x**-1, (x,), collapsed=collapsed)(*jet_args)
    assert 1e-3 > composed[K].abs().max() > 1e-5
