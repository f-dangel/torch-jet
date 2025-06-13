"""Tests for jet/__init__.py."""

from test.utils import report_nonclose
from typing import Any, Callable, Dict, Tuple

from pytest import mark
from torch import Tensor, cos, manual_seed, rand, sigmoid, sin, tanh, tensor
from torch.fx import symbolic_trace
from torch.nn import Linear, Module, Sequential, Tanh
from torch.nn.functional import linear

from jet import jet, rev_jet
from jet.utils import Primal, PrimalAndCoefficients, Value, ValueAndCoefficients


def compare_jet_results(out1: ValueAndCoefficients, out2: ValueAndCoefficients):
    value1, series1 = out1[0], out1[1:]
    value2, series2 = out2[0], out2[1:]

    report_nonclose(value1, value2, name="Values")
    assert len(series1) == len(series2)
    for i, (s1, s2) in enumerate(zip(series1, series2)):
        report_nonclose(s1, s2, name=f"Coefficients {i + 1}")


def check_jet(f: Callable[[Primal], Value], arg: PrimalAndCoefficients):
    x, vs = arg

    rev_jet_f = rev_jet(f)
    rev_jet_out = rev_jet_f(x, *vs)

    jet_f = jet(f, k=len(vs), verbose=True)
    jet_out = jet_f(x, *vs)

    compare_jet_results(jet_out, rev_jet_out)


INF = float("inf")

# contains only atomic functions
ATOMIC_CASES = [
    # 1d sine function
    {
        "f": sin,
        "shape": (1,),
        "k_max": INF,
        "id": "sin",
        "first_op_vanishing_derivatives": None,
    },
    # 2d sine function
    {
        "f": sin,
        "shape": (2,),
        "k_max": INF,
        "id": "sin",
        "first_op_vanishing_derivatives": None,
    },
    # 3d cosine function
    {
        "f": cos,
        "shape": (3,),
        "k_max": INF,
        "id": "cos",
        "first_op_vanishing_derivatives": None,
    },
    # 3d tanh function
    {
        "f": tanh,
        "shape": (5,),
        "k_max": INF,
        "id": "tanh",
        "first_op_vanishing_derivatives": None,
    },
    # 4d sigmoid function
    {
        "f": sigmoid,
        "shape": (4,),
        "k_max": INF,
        "id": "sigmoid",
        "first_op_vanishing_derivatives": None,
    },
    # linear layer
    {
        "f": Linear(4, 2),
        "shape": (4,),
        "k_max": INF,
        "id": "linear",
        "first_op_vanishing_derivatives": 2,
    },
    # 5d power function, two non-vanishing derivatives
    {
        "f": lambda x: x**2,
        "shape": (5,),
        "k_max": INF,
        "id": "pow-2",
        "first_op_vanishing_derivatives": 3,
    },
    # 5d power function, ten non-vanishing derivatives
    {
        "f": lambda x: x**10,
        "shape": (5,),
        "k_max": INF,
        "id": "pow-10",
        "first_op_vanishing_derivatives": 11,
    },
    # 5d power function, non-vanishing derivatives
    {
        "f": lambda x: x**1.5,
        "shape": (5,),
        "k_max": INF,
        "id": "pow-1.5",
        "first_op_vanishing_derivatives": None,
    },
    # addition of a tensor and a float
    {
        "f": lambda x: x + 2.0,
        "shape": (5,),
        "k_max": INF,
        "id": "add-2.0",
        "first_op_vanishing_derivatives": 2,
    },
    # subtraction of a tensor and a float
    {
        "f": lambda x: x - 2.0,
        "shape": (5,),
        "k_max": INF,
        "id": "sub-2.0",
        "first_op_vanishing_derivatives": 2,
    },
    # multiplication of a tensor and a float
    {
        "f": lambda x: x * 3.0,
        "shape": (5,),
        "k_max": INF,
        "id": "mul-3.0",
        "first_op_vanishing_derivatives": 2,
    },
]
ATOMIC_CASE_IDS = []
for atomic in ATOMIC_CASES:
    shape = atomic["shape"]
    ID = f"{atomic['id']}-{'_'.join([str(s) for s in shape])}d"
    atomic["is_batched"] = atomic.get("is_batched", False)
    ATOMIC_CASE_IDS.append(ID)


def f_multiply(x: Tensor) -> Tensor:
    """Test function for multiplication of two variables.

    Args:
        x: Input tensor.

    Returns:
        Tensor resulting from the multiplication of sin(x) and cos(sin(x)).
    """
    y = sin(x)
    return sin(y) * cos(y)


# fix seed when creating test cases with NN functions
manual_seed(1)

# contains only composed atomic functions
CASES_COMPACT = [
    *ATOMIC_CASES,
    # 2d sin(sin) function
    {
        "f": lambda x: sin(sin(x)),
        "shape": (2,),
        "k_max": INF,
        "id": "sin-sin",
        "first_op_vanishing_derivatives": None,
    },
    # 2d tanh(tanh) function
    {
        "f": lambda x: tanh(tanh(x)),
        "shape": (2,),
        "k_max": INF,
        "id": "tanh-tanh",
        "first_op_vanishing_derivatives": None,
    },
    # 2d linear(tanh) function
    {
        "f": lambda x: linear(
            tanh(x),
            tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]]).double(),
            bias=tensor([0.12, -0.34]).double(),
        ),
        "shape": (3,),
        "k_max": INF,
        "id": "tanh-linear",
        "first_op_vanishing_derivatives": None,
    },
    # 5d tanh-activated two-layer MLP
    {
        "f": Sequential(
            Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
        ),
        "shape": (5,),
        "k_max": INF,
        "id": "two-layer-tanh-mlp",
        "first_op_vanishing_derivatives": 2,
    },
    # 5d tanh-activated two-layer MLP with batched input
    {
        "f": Sequential(
            Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
        ),
        "shape": (10, 5),
        "k_max": INF,
        "is_batched": True,
        "id": "batched-two-layer-tanh-mlp",
        "first_op_vanishing_derivatives": 2,
    },
    # 3d sigmoid(sigmoid) function
    {
        "f": lambda x: sigmoid(sigmoid(x)),
        "shape": (3,),
        "k_max": INF,
        "id": "sigmoid-sigmoid",
        "first_op_vanishing_derivatives": None,
    },
    # 3d sin function with residual connection
    {
        "f": lambda x: sin(x) + x,
        "shape": (3,),
        "k_max": INF,
        "id": "sin-residual",
        "first_op_vanishing_derivatives": None,
    },
    # 3d sin function with negative residual connection
    {
        "f": lambda x: sin(x) - x,
        "shape": (3,),
        "k_max": INF,
        "id": "sin-neg-residual",
        "first_op_vanishing_derivatives": None,
    },
    # multiplication two variables
    {
        "f": f_multiply,
        "shape": (5,),
        "k_max": INF,
        "id": "multiply-variables",
        "first_op_vanishing_derivatives": None,
    },
]
CASES_COMPACT_IDS = []
for compact in CASES_COMPACT:
    shape = compact["shape"]
    ID = f"{compact['id']}-{'_'.join([str(s) for s in shape])}d"
    compact["is_batched"] = compact.get("is_batched", False)
    CASES_COMPACT_IDS.append(ID)

# expand compact definition of cases for k=0, ... min(k_max, K_MAX)
K_MAX = 5
CASES = []
CASE_IDS = []
for compact in CASES_COMPACT:
    k_expand = min(compact["k_max"], K_MAX)
    for k in range(k_expand + 1):
        shape = compact["shape"]
        ID = f"{k}-jet-{compact['id']}-{'_'.join([str(s) for s in shape])}d"
        expanded = {
            "f": compact["f"],
            "k": k,
            "shape": shape,
            "is_batched": compact["is_batched"],
        }
        CASES.append(expanded)
        CASE_IDS.append(ID)


def setup_case(
    config: Dict[str, Any],
    vmapsize: int = 0,
    taylor_coefficients: bool = True,
) -> Tuple[Callable[[Primal], Value], Primal, Tuple[Primal], bool]:
    """Instantiate the function, its input, and Taylor coefficients.

    Args:
        config: Configuration dictionary of the test case.
        vmapsize: Whether to generate inputs and Taylor coefficients for a vmap-ed
            operation. `0` means no vmap is applied. Default: `0`.
        taylor_coefficients: Whether to instantiate the Taylor coefficients.
            This is not necessary for some tests.

    Returns:
        Tuple containing the function, the input tensor, and the Taylor coefficients,
        and whether the case represents a batched setting. All are in double precision
        to avoid numerical issues.
    """
    manual_seed(0)
    f = config["f"]
    shape = config["shape"]

    if isinstance(f, Module):
        f = f.double()

    vmap_shape = shape if vmapsize == 0 else (vmapsize, *shape)
    x = rand(*vmap_shape).double()
    vs = (
        tuple(rand(*vmap_shape).double() for _ in range(config["k"]))
        if taylor_coefficients
        else ()
    )

    return f, x, vs, config["is_batched"]


@mark.parametrize("config", CASES, ids=CASE_IDS)
def test_jet(config: Dict[str, Any]):
    """Compare forward jet with reverse-mode reference implementation.

    Args:
        config: Configuration dictionary of the test case.
    """
    f, x, vs, _ = setup_case(config)
    check_jet(f, (x, vs))


@mark.parametrize("config", CASES, ids=CASE_IDS)
def test_symbolic_trace_jet(config: Dict[str, Any]):
    """Test whether the function produced by jet can be traced.

    Args:
        config: Configuration dictionary of the test case.
    """
    f, _, _, _ = setup_case(config, taylor_coefficients=False)
    k = config["k"]
    # generate the jet's compute graph
    jet_f = jet(f, k)

    # try tracing it
    symbolic_trace(jet_f)
