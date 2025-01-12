"""Tests for jet/__init__.py."""

from typing import Callable, Dict

import pytest
from torch import isclose, sin, tensor

from jet import jet, rev_jet
from jet.operations import Primal, PrimalAndCoefficients, Value, ValueAndCoefficients


def compare_jet_results(out1: ValueAndCoefficients, out2: ValueAndCoefficients):
    value1, series1 = out1
    value2, series2 = out2

    report_nonclose(value1, value2, name="Values")
    assert len(series1) == len(series2)
    for i, (s1, s2) in enumerate(zip(series1, series2)):
        report_nonclose(s1, s2, name=f"Coefficients {i+1}")


def report_nonclose(a, b, rtol=1e-5, atol=1e-8, name: str = "Tensors"):
    """Report non-closeness of two tensors."""
    close = a.allclose(b, rtol=rtol, atol=atol)
    if not close:
        print(f"{name} are not close.")
        for idx, (x, y) in enumerate(zip(a.flatten(), b.flatten())):
            if not isclose(x, y, rtol=rtol, atol=atol):
                print(f"Index {idx}: {x} != {y}")
    else:
        print(f"{name} are close.")
    assert close


def check_jet(f: Callable[[Primal], Value], arg: PrimalAndCoefficients):
    rev_jet_f = rev_jet(f)
    rev_jet_out = rev_jet_f(arg)

    jet_f = jet(f, verbose=True)
    jet_out = jet_f(arg)

    compare_jet_results(jet_out, rev_jet_out)


CASES = [
    # 2-jet of the 1d sine function
    {
        "f": lambda x: sin(x),
        "primal": lambda: tensor([0.1]),
        "coefficients": lambda: (tensor([0.2]), tensor([0.3])),
        "id": "2-jet-sin-1d",
    },
    # 2-jet of the 2d sine function
    {
        "f": lambda x: sin(sin(x)),
        "primal": lambda: tensor([0.1, 0.15]),
        "coefficients": lambda: (tensor([0.2, 0.25]), tensor([0.3, 0.35])),
        "id": "2-jet-sinsin-2d",
    },
    # 3-jet of the 2d sine function
    {
        "f": lambda x: sin(x),
        "primal": lambda: tensor([0.1, 0.12]),
        "coefficients": lambda: (
            tensor([0.2, 0.22]),
            tensor([0.3, 0.32]),
            tensor([0.4, 0.42]),
        ),
        "id": "3-jet-sin-2d",
    },
    # 4-jet of the 2d sine function
    {
        "f": lambda x: sin(x),
        "primal": lambda: tensor([0.1, 0.12]),
        "coefficients": lambda: (
            tensor([0.2, 0.22]),
            tensor([0.3, 0.32]),
            tensor([0.4, 0.42]),
            tensor([0.5, 0.52]),
        ),
        "id": "4-jet-sin-2d",
    },
    # 5-jet of the 2d sine function
    {
        "f": lambda x: sin(x),
        "primal": lambda: tensor([0.1, 0.12]),
        "coefficients": lambda: (
            tensor([0.2, 0.22]),
            tensor([0.3, 0.32]),
            tensor([0.4, 0.42]),
            tensor([0.5, 0.52]),
            tensor([0.6, 0.62]),
        ),
        "id": "5-jet-sin-2d",
    },
    # 5-jet of the 2d sin(sin) function
    {
        "f": lambda x: sin(sin(x)),
        "primal": lambda: tensor([0.1, 0.12]),
        "coefficients": lambda: (
            tensor([0.2, 0.22]),
            tensor([0.3, 0.32]),
            tensor([0.4, 0.42]),
            tensor([0.5, 0.52]),
            tensor([0.6, 0.62]),
        ),
        "id": "5-jet-sinsin-2d",
    },
]


@pytest.mark.parametrize("config", CASES, ids=lambda c: c["id"])
def test_jet(config: Dict[str, Callable]):
    """Compare forward jet with reverse-mode reference implementation.

    Args:
        config: Configuration dictionary containing the function, input, and Taylor
            coefficients.
    """
    f = config["f"]
    x = config["primal"]()
    vs = config["coefficients"]()

    # run everything in double precision to avoid round-off errors
    x = x.double()
    vs = tuple(v.double() for v in vs)
    check_jet(f, (x, vs))
