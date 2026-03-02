"""Tests for jet/__init__.py."""

from typing import Any, Callable

from pytest import mark
from torch import Tensor, cos, float64, manual_seed, rand, sigmoid, sin, tanh, tensor
from torch.nn import Linear, Module, Sequential, Tanh
from torch.nn.functional import linear
from torch.utils._pytree import TreeSpec, tree_flatten

import jet
from jet import rev_jet

from test.utils import report_nonclose


def compare_jet_results(
    out1: tuple[Tensor, ...], out2: tuple[Tensor, ...]
) -> None:
    """Compare two jet outputs in flat-tuple format ``(f0, f1, ..., fk)``.

    Kept for backward compatibility with ``test_simplify.py`` which compares
    laplacian/bilaplacian outputs (flat tuples).
    """
    value1, series1 = out1[0], out1[1:]
    value2, series2 = out2[0], out2[1:]

    report_nonclose(value1, value2, name="Values")
    assert len(series1) == len(series2)
    for i, (s1, s2) in enumerate(zip(series1, series2)):
        report_nonclose(s1, s2, name=f"Coefficients {i + 1}")


def _assert_specs_compatible(spec1: TreeSpec, spec2: TreeSpec, name: str = "") -> None:
    """Assert two tree specs are structurally compatible.

    Tolerates ``dict`` vs ``immutable_dict`` differences introduced by
    ``make_fx`` tracing by normalizing container type names before comparing.

    Args:
        spec1: First tree spec.
        spec2: Second tree spec.
        name: Label for error messages.

    Raises:
        AssertionError: If the tree specs are not structurally compatible.
    """
    if spec1 == spec2:
        return
    # Normalize container type names (make_fx turns dict → immutable_dict)
    s1 = str(spec1).replace("immutable_dict", "dict")
    s2 = str(spec2).replace("immutable_dict", "dict")
    assert s1 == s2, f"{name} tree structure mismatch: {spec1} vs {spec2}"


def compare_primals_series(out1: tuple[Any, tuple], out2: tuple[Any, tuple]) -> None:
    """Compare two jet outputs in ``(primals_out, series_out)`` format.

    Verifies that primals and each series order have matching tree structure
    (tolerating ``dict``/``immutable_dict`` differences) and close tensor values.

    Args:
        out1: First ``(primals_out, series_out)`` pair.
        out2: Second ``(primals_out, series_out)`` pair.

    Raises:
        AssertionError: If tree structures differ or tensor values are not close.
    """
    primals1, series1 = out1
    primals2, series2 = out2

    flat_p1, spec_p1 = tree_flatten(primals1)
    flat_p2, spec_p2 = tree_flatten(primals2)
    _assert_specs_compatible(spec_p1, spec_p2, "Primals")
    for j, (t1, t2) in enumerate(zip(flat_p1, flat_p2)):
        report_nonclose(t1, t2, name=f"Primals leaf {j}")

    assert len(series1) == len(series2), (
        f"Series length mismatch: {len(series1)} vs {len(series2)}"
    )
    for i, (s1, s2) in enumerate(zip(series1, series2)):
        sf1, spec_s1 = tree_flatten(s1)
        sf2, spec_s2 = tree_flatten(s2)
        _assert_specs_compatible(spec_s1, spec_s2, f"Series order {i + 1}")
        for j, (t1, t2) in enumerate(zip(sf1, sf2)):
            report_nonclose(t1, t2, name=f"Series order {i + 1} leaf {j}")


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
    x: Tensor, params: dict[str, Tensor | list[Tensor]]
) -> tuple[Tensor, dict[str, Tensor]]:
    """Function with deeply nested dict/list input and different output structure.

    Args:
        x: Input tensor.
        params: Nested pytree ``{"w": Tensor, "bs": [Tensor, Tensor]}``.

    Returns:
        A pytree ``(Tensor, {"a": Tensor, "b": Tensor})`` with different
        structure from the input.
    """
    h = sin(x) * params["w"]
    b0, b1 = params["bs"][0], params["bs"][1]
    return (h + b0, {"a": cos(h) * b1, "b": tanh(h + b0 + b1)})


def _deep_pytree_mock_args_fn() -> tuple[Tensor, dict[str, Tensor | list[Tensor]]]:
    """Create mock arguments for :func:`_deep_pytree_f`.

    Returns:
        A tuple ``(x, params)`` with random double-precision tensors.
    """
    return (
        rand(3).double(),
        {"w": rand(3).double(), "bs": [rand(3).double(), rand(3).double()]},
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
    {"f": lambda x: x**1.5, "mock_args_fn": lambda: (rand(5).double(),), "id": "pow-1.5"},
    {"f": lambda x: x + 2.0, "mock_args_fn": lambda: (rand(5).double(),), "id": "add-2.0"},
    {"f": lambda x: x - 2.0, "mock_args_fn": lambda: (rand(5).double(),), "id": "sub-2.0"},
    {"f": lambda x: x + x, "mock_args_fn": lambda: (rand(5).double(),), "id": "add-x-x_"},
    {"f": lambda x: x * 3.0, "mock_args_fn": lambda: (rand(5).double(),), "id": "mul-3.0"},
    {"f": lambda x: x * x, "mock_args_fn": lambda: (rand(5).double(),), "id": "mul-x-x_"},
    {"f": lambda x: sin(sin(x)), "mock_args_fn": lambda: (rand(2).double(),), "id": "sin-sin"},
    {"f": lambda x: tanh(tanh(x)), "mock_args_fn": lambda: (rand(2).double(),), "id": "tanh-tanh"},
    {
        "f": lambda x: linear(tanh(x), _TANH_LINEAR_W, bias=_TANH_LINEAR_B),
        "mock_args_fn": lambda: (rand(3).double(),),
        "id": "tanh-linear",
    },
    {"f": _MLP, "mock_args_fn": lambda: (rand(5).double(),), "id": "two-layer-tanh-mlp"},
    {
        "f": _MLP_BATCHED,
        "mock_args_fn": lambda: (rand(10, 5).double(),),
        "id": "batched-two-layer-tanh-mlp",
    },
    {"f": lambda x: sigmoid(sigmoid(x)), "mock_args_fn": lambda: (rand(3).double(),), "id": "sigmoid-sigmoid"},
    {"f": lambda x: sin(x) + x, "mock_args_fn": lambda: (rand(3).double(),), "id": "sin-residual"},
    {"f": lambda x: sin(x) - x, "mock_args_fn": lambda: (rand(3).double(),), "id": "sin-neg-residual"},
    {"f": f_multiply, "mock_args_fn": lambda: (rand(5).double(),), "id": "multiply-variables"},
    {"f": lambda x: x.sum(0), "mock_args_fn": lambda: (rand(3, 5).double(),), "id": "sum-3"},
]

JET_CASES_IDS = [config["id"] for config in JET_CASES]

K_MAX = 5
K = list(range(K_MAX + 1))
K_IDS = [f"{k=}" for k in K]


def setup_case(
    config: dict[str, Any], vmapsize: int = 0, derivative_order: int | None = None
) -> tuple[Callable[[Tensor], Tensor], Tensor, tuple[Tensor, ...]]:
    """Instantiate the function, its input, and Taylor coefficients.

    Args:
        config: Configuration dictionary of the test case. Must have ``"f"`` and
            ``"mock_args_fn"`` keys.
        vmapsize: Whether to generate inputs and Taylor coefficients for a vmap-ed
            operation. ``0`` means no vmap is applied. Default: ``0``.
        derivative_order: The number of Taylor coefficients to generate. No
            coefficients are generated if ``None``. Default: ``None``.

    Returns:
        Tuple containing the function, the input tensor, and the Taylor
        coefficients. All are in double precision to avoid numerical issues.
    """
    manual_seed(0)
    f = config["f"]

    if isinstance(f, Module):
        f = f.double()

    # Extract shape from mock_args_fn (single-input cases only)
    mock_args = config["mock_args_fn"]()
    shape = mock_args[0].shape

    vmap_shape = shape if vmapsize == 0 else (vmapsize, *shape)
    x = rand(*vmap_shape).double()
    vs = (
        ()
        if derivative_order is None
        else tuple(rand(*vmap_shape).double() for _ in range(derivative_order))
    )

    return f, x, vs


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
        "id": "dict-linear",
        "f": lambda x, params: x @ params["w"] + params["b"],
        "mock_args_fn": lambda: (
            rand(3).double(),
            {"w": rand(3, 2).double(), "b": rand(2).double()},
        ),
    },
    {
        "id": "dict-sin-cos",
        "f": lambda x, params: sin(x) * params["scale"] + params["bias"],
        "mock_args_fn": lambda: (
            rand(4).double(),
            {"scale": rand(4).double(), "bias": rand(4).double()},
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
    # deeply nested mixed containers with different input/output structure
    {
        "id": "nested-dict-list-in-tuple-dict-out",
        "f": _deep_pytree_f,
        "mock_args_fn": _deep_pytree_mock_args_fn,
    },
]

ALL_CASES_IDS = [c["id"] for c in ALL_CASES]


@mark.parametrize("k", K, ids=K_IDS)
@mark.parametrize("config", ALL_CASES, ids=ALL_CASES_IDS)
def test_jet(config: dict[str, Any], k: int):
    """Compare forward jet with rev_jet for all function types.

    Args:
        config: Configuration dictionary of the test case.
        k: The order of the jet to compute.
    """
    manual_seed(0)
    f = config["f"]
    if isinstance(f, Module):
        f = f.double()
    mock_args = config["mock_args_fn"]()

    manual_seed(42)
    primals = config["mock_args_fn"]()
    num_args = len(mock_args)
    per_order = tuple(config["mock_args_fn"]() for _ in range(k))
    series = tuple(tuple(per_order[j][i] for j in range(k)) for i in range(num_args))

    jet_f = jet.jet(f, k, mock_args)
    jet_out = jet_f(primals, series)

    rev_jet_f = rev_jet(f, k)
    rev_jet_out = rev_jet_f(primals, series)

    compare_primals_series(jet_out, rev_jet_out)
