"""Tests for jet/__init__.py."""

from typing import Any, Callable

from pytest import mark
from torch import Tensor, cos, manual_seed, rand, sigmoid, sin, tanh, tensor
from torch.nn import Linear, Module, Sequential, Tanh
from torch.nn.functional import linear

import jet
import jet.utils
from jet import rev_jet
from jet.tracing import capture_graph
from jet.utils import Primal, PrimalAndCoefficients, Value, ValueAndCoefficients
from test.utils import report_nonclose


def compare_jet_results(  # noqa: D103
    out1: ValueAndCoefficients, out2: ValueAndCoefficients
):
    value1, series1 = out1[0], out1[1:]
    value2, series2 = out2[0], out2[1:]

    report_nonclose(value1, value2, name="Values")
    assert len(series1) == len(series2)
    for i, (s1, s2) in enumerate(zip(series1, series2)):
        report_nonclose(s1, s2, name=f"Coefficients {i + 1}")


def check_jet(f: Callable[[Primal], Value], arg: PrimalAndCoefficients):  # noqa: D103
    x, vs = arg

    rev_jet_f = rev_jet(f)
    rev_jet_out = rev_jet_f(x, *vs)

    jet_f = jet.jet(f, derivative_order=len(vs), example_input=x, verbose=True)
    jet_out = jet_f(x, *vs)

    compare_jet_results(jet_out, rev_jet_out)


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


# make generation of test cases deterministic
manual_seed(1)

JET_CASES = [
    # 1d sine function
    {"f": sin, "shape": (1,), "id": "sin"},
    # 2d sine function
    {"f": sin, "shape": (2,), "id": "sin"},
    # 3d cosine function
    {"f": cos, "shape": (3,), "id": "cos"},
    # 3d tanh function
    {"f": tanh, "shape": (5,), "id": "tanh"},
    # 4d sigmoid function
    {"f": sigmoid, "shape": (4,), "id": "sigmoid"},
    # linear layer
    {"f": Linear(4, 2), "shape": (4,), "id": "linear"},
    # 5d power function, two non-vanishing derivatives
    {"f": lambda x: x**2, "shape": (5,), "id": "pow-2"},
    # 5d power function, ten non-vanishing derivatives
    {"f": lambda x: x**10, "shape": (5,), "id": "pow-10"},
    # 5d power function, non-vanishing derivatives
    {"f": lambda x: x**1.5, "shape": (5,), "id": "pow-1.5"},
    # addition of a tensor and a float
    {"f": lambda x: x + 2.0, "shape": (5,), "id": "add-2.0"},
    # subtraction of a tensor and a float
    {"f": lambda x: x - 2.0, "shape": (5,), "id": "sub-2.0"},
    # addition of x with itself
    {"f": lambda x: x + x, "shape": (5,), "id": "add-x-x_"},
    # multiplication of a tensor and a float
    {"f": lambda x: x * 3.0, "shape": (5,), "id": "mul-3.0"},
    # multiplication of x with itself
    {"f": lambda x: x * x, "shape": (5,), "id": "mul-x-x_"},
    {"f": lambda x: jet.utils.replicate(x, 6), "shape": (5,), "id": "replicate-6"},
    # 2d sin(sin) function
    {"f": lambda x: sin(sin(x)), "shape": (2,), "id": "sin-sin"},
    # 2d tanh(tanh) function
    {"f": lambda x: tanh(tanh(x)), "shape": (2,), "id": "tanh-tanh"},
    # 2d linear(tanh) function
    {
        "f": lambda x: linear(
            tanh(x),
            tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]]).double(),
            bias=tensor([0.12, -0.34]).double(),
        ),
        "shape": (3,),
        "id": "tanh-linear",
    },
    # 5d tanh-activated two-layer MLP
    {
        "f": Sequential(
            Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
        ),
        "shape": (5,),
        "id": "two-layer-tanh-mlp",
    },
    # 5d tanh-activated two-layer MLP with batched input
    {
        "f": Sequential(
            Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
        ),
        "shape": (10, 5),
        "is_batched": True,
        "id": "batched-two-layer-tanh-mlp",
    },
    # 3d sigmoid(sigmoid) function
    {"f": lambda x: sigmoid(sigmoid(x)), "shape": (3,), "id": "sigmoid-sigmoid"},
    # 3d sin function with residual connection
    {"f": lambda x: sin(x) + x, "shape": (3,), "id": "sin-residual"},
    # 3d sin function with negative residual connection
    {"f": lambda x: sin(x) - x, "shape": (3,), "id": "sin-neg-residual"},
    # multiplication two variables
    {"f": f_multiply, "shape": (5,), "id": "multiply-variables"},
    # sum
    {"f": lambda x: x.sum(0), "shape": (3, 5), "id": "sum-3"},
]

# set the `is_batched` flag for all cases
for config in JET_CASES:
    config["is_batched"] = config.get("is_batched", False)

JET_CASES_IDS = [config["id"] for config in JET_CASES]

K_MAX = 5
K = list(range(K_MAX + 1))
K_IDS = [f"{k=}" for k in K]


def setup_case(
    config: dict[str, Any], vmapsize: int = 0, derivative_order: int | None = None
) -> tuple[Callable[[Primal], Value], Primal, tuple[Primal, ...]]:
    """Instantiate the function, its input, and Taylor coefficients.

    Args:
        config: Configuration dictionary of the test case.
        vmapsize: Whether to generate inputs and Taylor coefficients for a vmap-ed
            operation. `0` means no vmap is applied. Default: `0`.
        derivative_order: The number of Taylor coefficients to generate. No coefficients are generated
            if `None`. Default: `None`.

    Returns:
        tuple containing the function, the input tensor, and the Taylor coefficients.
        All are in double precision to avoid numerical issues.
    """
    manual_seed(0)
    f = config["f"]
    shape = config["shape"]

    if isinstance(f, Module):
        f = f.double()

    vmap_shape = shape if vmapsize == 0 else (vmapsize, *shape)
    x = rand(*vmap_shape).double()
    vs = (
        ()
        if derivative_order is None
        else tuple(rand(*vmap_shape).double() for _ in range(derivative_order))
    )

    return f, x, vs


@mark.parametrize("k", K, ids=K_IDS)
@mark.parametrize("config", JET_CASES, ids=JET_CASES_IDS)
def test_jet(config: dict[str, Any], k: int):
    """Compare forward jet with reverse-mode reference implementation.

    Args:
        config: Configuration dictionary of the test case.
        k: The order of the jet to compute.
    """
    f, x, vs = setup_case(config, derivative_order=k)
    check_jet(f, (x, vs))


@mark.parametrize("k", K, ids=K_IDS)
@mark.parametrize("config", JET_CASES, ids=JET_CASES_IDS)
def test_symbolic_trace_jet(config: dict[str, Any], k: int):
    """Test whether the function produced by jet can be traced.

    Args:
        config: Configuration dictionary of the test case.
        k: The order of the jet to compute.
    """
    f, x, vs = setup_case(config, derivative_order=k)
    # generate the jet's compute graph
    jet_f = jet.jet(f, k, example_input=x)

    # try tracing it with example inputs (primal + coefficients)
    capture_graph(jet_f, x if not vs else vs[0])
