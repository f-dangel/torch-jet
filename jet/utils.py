"""Utility functions for computing jets."""

from math import factorial, prod
from typing import Callable, Tuple

from torch import Tensor, einsum
from torch.nn import Module

# type annotation for arguments and Taylor coefficients in input and output space
Primal = Tensor
PrimalAndCoefficients = Tuple[Primal, Tuple[Primal, ...]]
Value = Tensor
ValueAndCoefficients = Tuple[Value, Tuple[Value, ...]]


def integer_partitions(n: int, I: int = 1):  # noqa: E741
    """Compute the integer partitions of a positive integer.

    Taken from: https://stackoverflow.com/a/44209393.

    Args:
        n: Positive integer.
    """
    yield (n,)
    for i in range(I, n // 2 + 1):
        for p in integer_partitions(n - i, i):
            yield (i,) + p


def tensor_prod(*tensors: Tensor) -> Tensor:
    """Compute the element-wise product of tensors.

    Args:
        tensors: Tensors to be multiplied.

    Returns:
        Element-wise product of the tensors.
    """
    (ndim,) = {t.ndim for t in tensors}
    idxs = "".join(chr(ord("a") + i) for i in range(ndim))
    equation = ",".join(len(tensors) * [idxs]) + f"->{idxs}"
    return einsum(equation, *tensors)


def multiplicity(sigma: Tuple[int, ...]) -> float:
    """Compute the scaling of a summand in Faa di Bruno's formula.

    Args:
        sigma: Tuple of integers representing the integer partitioning.

    Returns:
        Multiplicity of the summand.

    Raises:
        ValueError: If the multiplicity is not an integer.
    """
    # see the scheme above the 'Variations' section here:
    # https://en.wikipedia.org/wiki/Fa%C3%A0_di_Bruno%27s_formula
    k = sum(sigma)
    n_i = {i + 1: sigma.count(i + 1) for i in range(k)}
    multiplicity = (
        factorial(k)
        / prod(factorial(eta) for eta in sigma)
        / prod(factorial(n) for n in n_i.values())
    )
    if not multiplicity.is_integer():
        raise ValueError(f"Multiplicity should be an integer, but got {multiplicity}.")
    return multiplicity


class WrapperModule(Module):
    """Wraps a function in a module."""

    def __init__(self, f: Callable[[Primal], Value]):
        """Initialize the module.

        Args:
            f: Function to wrap.
        """
        super().__init__()
        self.f = f

    def forward(self, x: Primal) -> Value:
        """Forward pass of the module.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.f(x)
