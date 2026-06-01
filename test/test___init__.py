"""Tests for jet/__init__.py."""

from typing import Any, Callable

from pytest import mark, raises
from torch import (
    Tensor,
    cos,
    float64,
    manual_seed,
    rand,
    sigmoid,
    sin,
    tanh,
    tensor,
    zeros,
    zeros_like,
)
from torch.nn import Linear, Module, Sequential, Tanh
from torch.nn.functional import linear
from torch.utils._pytree import tree_map

import jet
from jet import rev_jet
from jet.tracing import capture_graph
from test.utils import report_pytrees_nonclose

INF = float("inf")


def f_multiply(x: Tensor) -> Tensor:
    """Test function for multiplication of two variables.

    Args:
        x: Input tensor.

    Returns:
        Tensor resulting from the multiplication of sin(x) and cos(sin(x)).
    """
    y = sin(x)
    return sin(y) * cos(y)


def _deep_pytree_f(
    x: Tensor, params: list[Tensor | list[Tensor]]
) -> tuple[Tensor, dict[str, Tensor]]:
    """Function with deeply nested input/output of different structure.

    The input uses ``tuple``/``list`` containers (``dict`` arguments are
    unsupported); the output mixes a ``tuple`` with a ``dict``.

    Args:
        x: Input tensor.
        params: Nested pytree ``[Tensor, [Tensor, Tensor]]`` (``[w, [b0, b1]]``).

    Returns:
        A pytree ``(Tensor, {"a": Tensor, "b": Tensor})`` with different structure
        from the input.
    """
    w = params[0]
    b0, b1 = params[1]
    h = sin(x) * w
    return (h + b0, {"a": cos(h) * b1, "b": tanh(h + b0 + b1)})


def _deep_pytree_mock_args_fn() -> tuple[Tensor, list[Tensor | list[Tensor]]]:
    """Create mock arguments for :func:`_deep_pytree_f`.

    Returns:
        A tuple ``(x, params)`` with random double-precision tensors.
    """
    return (
        rand(3).double(),
        [rand(3).double(), [rand(3).double(), rand(3).double()]],
    )


# make generation of test cases deterministic
manual_seed(1)

_TANH_LINEAR_W = tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype=float64)
_TANH_LINEAR_B = tensor([0.12, -0.34], dtype=float64)
_LINEAR = Linear(4, 2).double()
_MLP = Sequential(
    Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
).double()
_MLP_BATCHED = Sequential(
    Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
).double()

# ---------------------------------------------------------------------------
# JET_CASES: single-input configs used by ``setup_case`` (imported by
# test_laplacian, test_bilaplacian, test_simplify, and benchmarks).
# ---------------------------------------------------------------------------

JET_CASES = [
    {"f": sin, "mock_args_fn": lambda: (rand(1).double(),), "id": "sin-1d"},
    {"f": sin, "mock_args_fn": lambda: (rand(2).double(),), "id": "sin-2d"},
    {"f": cos, "mock_args_fn": lambda: (rand(3).double(),), "id": "cos"},
    {"f": tanh, "mock_args_fn": lambda: (rand(5).double(),), "id": "tanh"},
    {"f": sigmoid, "mock_args_fn": lambda: (rand(4).double(),), "id": "sigmoid"},
    {"f": _LINEAR, "mock_args_fn": lambda: (rand(4).double(),), "id": "linear"},
    {"f": lambda x: x**2, "mock_args_fn": lambda: (rand(5).double(),), "id": "pow-2"},
    {"f": lambda x: x**10, "mock_args_fn": lambda: (rand(5).double(),), "id": "pow-10"},
    {
        "f": lambda x: x**1.5,
        "mock_args_fn": lambda: (rand(5).double(),),
        "id": "pow-1.5",
    },
    {
        "f": lambda x: x + 2.0,
        "mock_args_fn": lambda: (rand(5).double(),),
        "id": "add-2.0",
    },
    {
        "f": lambda x: x - 2.0,
        "mock_args_fn": lambda: (rand(5).double(),),
        "id": "sub-2.0",
    },
    {
        "f": lambda x: x + x,
        "mock_args_fn": lambda: (rand(5).double(),),
        "id": "add-x-x_",
    },
    {
        "f": lambda x: x * 3.0,
        "mock_args_fn": lambda: (rand(5).double(),),
        "id": "mul-3.0",
    },
    {
        "f": lambda x: x * x,
        "mock_args_fn": lambda: (rand(5).double(),),
        "id": "mul-x-x_",
    },
    {
        "f": lambda x: sin(sin(x)),
        "mock_args_fn": lambda: (rand(2).double(),),
        "id": "sin-sin",
    },
    {
        "f": lambda x: tanh(tanh(x)),
        "mock_args_fn": lambda: (rand(2).double(),),
        "id": "tanh-tanh",
    },
    {
        "f": lambda x: linear(tanh(x), _TANH_LINEAR_W, bias=_TANH_LINEAR_B),
        "mock_args_fn": lambda: (rand(3).double(),),
        "id": "tanh-linear",
    },
    {
        "f": _MLP,
        "mock_args_fn": lambda: (rand(5).double(),),
        "id": "two-layer-tanh-mlp",
    },
    {
        "f": _MLP_BATCHED,
        "mock_args_fn": lambda: (rand(10, 5).double(),),
        "id": "batched-two-layer-tanh-mlp",
    },
    {
        "f": lambda x: sigmoid(sigmoid(x)),
        "mock_args_fn": lambda: (rand(3).double(),),
        "id": "sigmoid-sigmoid",
    },
    {
        "f": lambda x: sin(x) + x,
        "mock_args_fn": lambda: (rand(3).double(),),
        "id": "sin-residual",
    },
    {
        "f": lambda x: sin(x) - x,
        "mock_args_fn": lambda: (rand(3).double(),),
        "id": "sin-neg-residual",
    },
    {
        "f": f_multiply,
        "mock_args_fn": lambda: (rand(5).double(),),
        "id": "multiply-variables",
    },
    {
        "f": lambda x: x.sum(0),
        "mock_args_fn": lambda: (rand(3, 5).double(),),
        "id": "sum-3",
    },
]

JET_CASES_IDS = [config["id"] for config in JET_CASES]

K_MAX = 5
K = list(range(K_MAX + 1))
K_IDS = [f"derivative_order={derivative_order}" for derivative_order in K]


def setup_case(config: dict[str, Any]) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Instantiate the function and its input.

    The input is taken verbatim from the first leaf of ``config["mock_args_fn"]()``
    -- if the case wants a batched input it should encode the batch dimension
    into its ``mock_args_fn`` directly.

    Args:
        config: Configuration dictionary of the test case. Must have ``"f"`` and
            ``"mock_args_fn"`` keys. ``mock_args_fn()`` must return a tuple
            whose first leaf is the (already double-precision) input tensor.

    Returns:
        Tuple containing the function and the input tensor.
    """
    manual_seed(0)
    return config["f"], config["mock_args_fn"]()[0]


# ---------------------------------------------------------------------------
# ALL_CASES: unified test case list for ``test_jet``.
# Single-input cases derive ``mock_args_fn`` from JET_CASES; general cases
# (multi-input, pytree I/O) are added directly.
# ---------------------------------------------------------------------------

ALL_CASES = JET_CASES + [
    # multi-input: (Tensor, ..., Tensor) -> Tensor
    {
        "id": "add-xy",
        "f": lambda x, y: x + y,
        "mock_args_fn": lambda: (rand(3).double(), rand(3).double()),
    },
    {
        "id": "sin-x-cos-y",
        "f": lambda x, y: sin(x) * cos(y),
        "mock_args_fn": lambda: (rand(3).double(), rand(3).double()),
    },
    {
        "id": "sub-xy",
        "f": lambda x, y: x - y,
        "mock_args_fn": lambda: (rand(4).double(), rand(4).double()),
    },
    {
        "id": "mul-xy",
        "f": lambda x, y: x * y,
        "mock_args_fn": lambda: (rand(5).double(), rand(5).double()),
    },
    # pytree-input: PyTree -> Tensor
    {
        "id": "list-linear",
        "f": lambda x, params: x @ params[0] + params[1],
        "mock_args_fn": lambda: (
            rand(3).double(),
            [rand(3, 2).double(), rand(2).double()],
        ),
    },
    {
        "id": "list-sin-cos",
        "f": lambda x, params: sin(x) * params[0] + params[1],
        "mock_args_fn": lambda: (
            rand(4).double(),
            [rand(4).double(), rand(4).double()],
        ),
    },
    # dict inputs in supported positions (single dict arg, and dict first)
    {
        "id": "dict-in-single",
        "f": lambda d: sin(d["a"]) * d["b"],
        "mock_args_fn": lambda: ({"a": rand(4).double(), "b": rand(4).double()},),
    },
    {
        "id": "dict-first",
        "f": lambda params, x: params["scale"] * sin(x) + params["bias"],
        "mock_args_fn": lambda: (
            {"scale": rand(3).double(), "bias": rand(3).double()},
            rand(3).double(),
        ),
    },
    # pytree-output: Tensor -> PyTree
    {
        "id": "tuple-sin-cos",
        "f": lambda x: (sin(x), cos(x)),
        "mock_args_fn": lambda: (rand(3).double(),),
    },
    {
        "id": "dict-sin-cos-out",
        "f": lambda x: {"sin": sin(x), "cos": cos(x)},
        "mock_args_fn": lambda: (rand(3).double(),),
    },
    # multi-input, pytree-output: (Tensor, Tensor) -> PyTree
    {
        "id": "multi-in-tuple-out",
        "f": lambda x, y: (x + y, x * y),
        "mock_args_fn": lambda: (rand(4).double(), rand(4).double()),
    },
    {
        "id": "multi-in-dict-out",
        "f": lambda x, y: {"sum": x + y, "prod": x * y},
        "mock_args_fn": lambda: (rand(4).double(), rand(4).double()),
    },
    # deeply nested containers with different input/output structure
    {
        "id": "nested-list-in-tuple-dict-out",
        "f": _deep_pytree_f,
        "mock_args_fn": _deep_pytree_mock_args_fn,
    },
]

ALL_CASES_IDS = [c["id"] for c in ALL_CASES]


@mark.parametrize("derivative_order", K, ids=K_IDS)
@mark.parametrize("config", ALL_CASES, ids=ALL_CASES_IDS)
def test_jet(config: dict[str, Any], derivative_order: int):
    """Compare forward jet with rev_jet for all function types.

    Args:
        config: Configuration dictionary of the test case.
        derivative_order: The order of the jet to compute.
    """
    manual_seed(0)
    f = config["f"]
    mock_args = config["mock_args_fn"]()

    manual_seed(42)
    primals = config["mock_args_fn"]()
    # Build the new-convention args: one pytree per argument of f, with each
    # tensor leaf zipped into a (primal, c_1, ..., c_K) jet tuple.
    coeffs_by_order = [config["mock_args_fn"]() for _ in range(derivative_order)]
    args = tuple(
        tree_map(
            lambda *ts: tuple(ts),
            primals[arg_idx],
            *(coeffs_by_order[order][arg_idx] for order in range(derivative_order)),
        )
        for arg_idx in range(len(primals))
    )

    jet_f = jet.jet(f, mock_args)
    jet_out = jet_f(*args)

    rev_jet_f = rev_jet(f)
    rev_jet_out = rev_jet_f(*args)

    report_pytrees_nonclose(jet_out, rev_jet_out)


def _setup_collapsed_jet_args(
    config: dict[str, Any], derivative_order: int, R: int = 2
):
    """Set up mock args and jet args (new convention) for collapsed jet testing.

    Each tensor leaf of every argument is bundled into a jet tuple
    ``(primal, c_1, ..., c_K)`` with the collapsed-jet shape convention:

    - ``c_1`` carries the ``R`` directions (shape ``(R, *)``);
    - ``c_2..c_{K-1}`` are batched zeros (shape ``(R, *)``);
    - ``c_K`` is the collapsed zero (shape ``(*)``, no ``R`` dim).

    Args:
        config: Configuration dictionary with ``"f"`` and ``"mock_args_fn"`` keys.
        derivative_order: The order of the Taylor expansion (K >= 2).
        R: Number of random directions. Default: ``2``.

    Returns:
        Tuple ``(f, mock_args, args)`` ready for both ``jet(..., collapsed=True)``
        and ``_uncollapsed_via_vmap``: ``mock_args`` is the tracing template
        (zero tensors), and ``args`` is a tuple of pytrees -- one per argument
        of ``f`` -- whose tensor leaves are the jet tuples.
    """
    K = derivative_order
    f = config["f"]
    mock_args = config["mock_args_fn"]()

    if isinstance(f, Module):
        f = f.double()

    manual_seed(42)

    def make_jet_leaf(primal_meta: Tensor) -> tuple[Tensor, ...]:
        primal = rand(*primal_meta.shape, dtype=float64)
        coeffs = [rand(R, *primal.shape, dtype=float64)]  # c_1: batched directions
        coeffs += [  # c_2..c_{K-1}: batched zeros
            zeros(R, *primal.shape, dtype=float64) for _ in range(K - 2)
        ]
        coeffs.append(zeros_like(primal))  # c_K: collapsed zero
        return (primal, *coeffs)

    args = tuple(tree_map(make_jet_leaf, arg_template) for arg_template in mock_args)
    mock_args = tree_map(lambda t: zeros(*t.shape, dtype=float64), mock_args)

    return f, mock_args, args


@mark.parametrize("derivative_order", [2, 3, 4], ids=["K=2", "K=3", "K=4"])
@mark.parametrize("config", ALL_CASES, ids=ALL_CASES_IDS)
def test_collapsed_jet(config: dict[str, Any], derivative_order: int):
    """Collapsed jet matches standard jet + vmap + sum.

    Args:
        config: Configuration dictionary of the test case.
        derivative_order: The order of the jet to compute.
    """
    f, mock_args, args = _setup_collapsed_jet_args(config, derivative_order)

    std_f = jet._uncollapsed_via_vmap(f, mock_args, randomization=None)
    cjet_f = jet.jet(f, mock_args, collapsed=True)

    report_pytrees_nonclose(std_f(*args), cjet_f(*args))


def test_collapsed_jet_rejects_order_below_2():
    """jet(..., collapsed=True) raises ValueError at call time for K < 2."""
    cjet_f = jet.jet(sin, (zeros(3),), collapsed=True)
    x = zeros(3)

    # K=1: jet tuple has length 2 -> only a primal and one coefficient.
    # K=0: jet tuple has length 1 -> only a primal.
    for jet_tuple in [(x, x), (x,)]:
        with raises(ValueError, match="collapsed mode requires K >= 2"):
            cjet_f(jet_tuple)


def test_collapsed_jet_constant_output_uses_collapsed_shape():
    """F1: constant outputs in collapsed mode get c_1..c_{K-1} of shape (R, *S).

    A function with a constant output leaf (a tensor independent of the inputs)
    must still produce coefficients matching the collapsed-mode shape contract,
    or downstream consumers see broken broadcasts.
    """
    R = 3  # K=2 implicit from passing one coefficient slot to cjet_f below
    out_shape = (4,)

    def f(x: Tensor) -> tuple[Tensor, Tensor]:
        return sin(x), zeros(*out_shape, dtype=float64)  # second leaf is constant

    cjet_f = jet.jet(f, (zeros(3, dtype=float64),), collapsed=True)
    primal = rand(3, dtype=float64)
    c1 = rand(R, 3, dtype=float64)
    cK = zeros(3, dtype=float64)
    (_, _, _), (const, const_c1, const_cK) = cjet_f((primal, c1, cK))
    assert const.shape == out_shape
    assert const_c1.shape == (R, *out_shape)
    assert const_cK.shape == out_shape


def test_capture_graph_rejects_non_tuple_mock_args():
    """capture_graph requires mock_args to be a tuple (not a bare tensor)."""
    with raises(TypeError, match="must be a tuple"):
        capture_graph(sin, zeros(3))


def test_jet_rejects_unsupported_tuple_dict_signature():
    """Reject only the (tensor/tuple, dict) two-argument signature (make_fx bug).

    All other dict signatures are supported, so they must not raise.
    """
    t, d = zeros(3), {"a": zeros(3)}

    # Unsupported: two args, first tensor/tuple, second dict.
    f = lambda x, params: x * params["a"]  # noqa: E731
    match = r"pytorch/pytorch#185640"  # pin to the tracked upstream issue
    for collapsed in (False, True):
        with raises(NotImplementedError, match=match):
            jet.jet(f, (t, d), collapsed=collapsed)

    # Supported dict signatures must not raise.
    jet.jet(lambda d: d["a"] * 2, (d,))  # single dict arg
    jet.jet(lambda d, x: d["a"] + x, (d, t))  # dict first
    jet.jet(lambda x, y, d: x + y + d["a"], (t, t, d))  # three args, trailing dict
