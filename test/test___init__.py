"""Tests for jet/__init__.py."""

from typing import Any, Callable, Dict, Tuple

import pytest
from torch import Tensor, isclose, manual_seed, rand, sigmoid, sin, stack, tanh, tensor
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
        report_nonclose(s1, s2, name=f"Coefficients {i+1}")


def report_nonclose(
    a: Tensor, b: Tensor, rtol: float = 1e-5, atol: float = 1e-8, name: str = "Tensors"
):
    """Report non-closeness of two tensors.

    Args:
        a: First tensor.
        b: Second tensor.
        rtol: Relative tolerance. Default: `1e-5`.
        atol: Absolute tolerance. Default: `1e-8`.
        name: Name of the tensors. Default: `"Tensors"`.
    """
    close = a.allclose(b, rtol=rtol, atol=atol)
    if not close:
        print(f"{name} are not close.")
        for idx, (x, y) in enumerate(zip(a.flatten(), b.flatten())):
            if not isclose(x, y, rtol=rtol, atol=atol):
                print(f"Index {idx}: {x} != {y} (ratio: {x / y})")
    else:
        print(f"{name} are close.")
    assert close


def check_jet(f: Callable[[Primal], Value], arg: PrimalAndCoefficients, vmap: bool):
    x, vs = arg

    # create a function that manually vmaps `rev_jet` (calling `torch.vmap` fails)
    if not vmap:
        rev_jet_f = rev_jet(f)
    else:
        _rev_jet_f = rev_jet(f)

        def rev_jet_f(x, *vs):
            (num,) = set([x.shape[0]] + [v.shape[0] for v in vs])
            out, vs_out = [], [[] for _ in range(len(vs))]

            # loop over vmap dimension
            for n in range(num):
                print("Hello", x[n].shape, [v[n].shape for v in vs])
                result_n = _rev_jet_f(x[n], *[v[n] for v in vs])
                out.append(result_n[0])
                for i, v_out_n in enumerate(result_n[1:]):
                    vs_out[i].append(v_out_n)

            # stack results
            out = stack(out)
            vs_out = tuple(stack(v) for v in vs_out)

            return (out, *vs_out)

    rev_jet_out = rev_jet_f(x, *vs)

    jet_f = jet(f, k=len(vs), vmap=vmap, verbose=True)
    jet_out = jet_f(x, *vs)

    compare_jet_results(jet_out, rev_jet_out)


CASES_COMPACT = [
    # 2-jet of the 1d sine function
    {"f": lambda x: sin(x), "shape": (1,), "k": 2, "id": "sin"},
    # 2-jet of the 2d sin(sin) function
    {"f": lambda x: sin(sin(x)), "shape": (2,), "k": 2, "id": "sin-sin"},
    # 3-jet of the 2d sine function
    {"f": lambda x: sin(x), "shape": (2,), "k": 3, "id": "sin"},
    # 4-jet of the 2d sine function
    {"f": lambda x: sin(x), "shape": (2,), "k": 4, "id": "sin"},
    # 5-jet of the 2d sine function
    {"f": lambda x: sin(x), "shape": (2,), "k": 5, "id": "sin"},
    # 5-jet of the 2d sin(sin) function
    {"f": lambda x: sin(sin(x)), "shape": (2,), "k": 5, "id": "sin-sin"},
    # 3-jet of the 2d tanh(tanh) function
    {"f": lambda x: tanh(tanh(x)), "shape": (2,), "k": 3, "id": "tanh-tanh"},
    # 3-jet of the 2d linear(tanh) function
    {
        "f": lambda x: linear(
            tanh(x),
            tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]]).double(),
            bias=tensor([0.12, -0.34]).double(),
        ),
        "shape": (3,),
        "k": 3,
        "id": "tanh-linear",
    },
    # 3-jet of a tanh-activated two-layer MLP
    {
        "f": Sequential(
            Linear(5, 4, bias=False), Tanh(), Linear(4, 3, bias=True), Tanh()
        ),
        "shape": (5,),
        "k": 3,
        "id": "two-layer-tanh-mlp",
    },
    # 2-jet of a 3d sigmoid(sigmoid) function
    {
        "f": lambda x: sigmoid(sigmoid(x)),
        "shape": (3,),
        "k": 2,
        "id": "sigmoid-sigmoid",
    },
]
# expand compact definition of cases
CASES = []
CASE_IDS = []
for compact in CASES_COMPACT:
    k = compact["k"]
    shape = compact["shape"]
    ID = f"{k}-jet-{compact['id']}-{'_'.join([str(s) for s in shape])}d"
    expanded = {"f": compact["f"], "k": compact["k"], "shape": shape}
    CASES.append(expanded)
    CASE_IDS.append(ID)


def setup_case(
    config: Dict[str, Any], vmapsize: int = 0
) -> Tuple[Callable[[Primal], Value], Primal, Tuple[Primal]]:
    """Instantiate the function, its input, and Taylor coefficients.

    Args:
        config: Configuration dictionary of the test case.
        vmapsize: Whether to generate inputs and Taylor coefficients for a vmap-ed
            operation. `0` means no vmap is applied. Default: `0`.

    Returns:
        Tuple containing the function, the input tensor, and the Taylor coefficients.
        All are in double precision to avoid numerical issues.
    """
    manual_seed(0)
    f = config["f"]
    k = config["k"]
    shape = config["shape"]

    if isinstance(f, Module):
        f = f.double()

    vmap_shape = shape if vmapsize == 0 else (vmapsize, *shape)
    x = rand(*vmap_shape).double()
    vs = tuple(rand(*vmap_shape).double() for _ in range(k))

    return f, x, vs


VMAPSIZES = [0, 4]
VMAPSIZE_IDS = ["novmap" if v == 0 else f"vmapsize={v}" for v in VMAPSIZES]


@pytest.mark.parametrize("config", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("vmapsize", VMAPSIZES, ids=VMAPSIZE_IDS)
def test_jet(config: Dict[str, Any], vmapsize: int):
    """Compare forward jet with reverse-mode reference implementation.

    Args:
        config: Configuration dictionary of the test case.
        vmapsize: The size of the vmaped dimension. `0` means no vmap.
    """
    f, x, vs = setup_case(config, vmapsize=vmapsize)
    check_jet(f, (x, vs), vmapsize != 0)
