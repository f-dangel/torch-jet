"""Implementation of AD primitives in Taylor-mode arithmetic."""

from math import factorial, prod
from typing import Tuple

from torch import Tensor, cos, sin, zeros_like

from jet.utils import integer_partitions, tensor_prod

# type annotation for arguments and Taylor coefficients in input and output space
Primal = Tensor
PrimalAndCoefficients = Tuple[Primal, Tuple[Primal, ...]]
Value = Tensor
ValueAndCoefficients = Tuple[Value, Tuple[Value, ...]]


def jet_sin(arg: PrimalAndCoefficients) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the sine function.

    Args:
        arg: Input tensor and its Taylor coefficients.

    Returns:
        Tuple containing the value of the sine function and its Taylor coefficients.
    """
    (x, vs) = arg
    sin_x = sin(x)
    cos_x = cos(x)

    def dn_sin(*vs) -> Tensor:
        """Contract the derivative tensor along the vectors."""
        n = len(vs)
        sign = 1 if n % 4 in [0, 1] else -1
        func = sin_x if n % 2 == 0 else cos_x
        return sign * func * tensor_prod(*vs)

    vs_out = [zeros_like(sin_x) for _ in vs]
    order = len(vs)

    def multiplicity(sigma: Tuple[int, ...]) -> int:
        # see the scheme above the 'Variations' section here:
        # https://en.wikipedia.org/wiki/Fa%C3%A0_di_Bruno%27s_formula
        k = sum(sigma)
        n_i = {i + 1: sigma.count(i + 1) for i in range(k)}
        multiplicity = (
            factorial(k)
            / prod(factorial(eta) for eta in sigma)
            / prod(factorial(n) for n in n_i.values())
        )
        assert int(multiplicity) == multiplicity
        return int(multiplicity)

    for k in range(order):
        for sigma in integer_partitions(k + 1):
            vs_contract = [vs[i - 1] for i in sigma]
            nu = multiplicity(sigma)
            vs_out[k].add_(dn_sin(*vs_contract), alpha=nu)

    return sin_x, tuple(vs_out)


MAPPING = {sin: jet_sin}
