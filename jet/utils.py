"""Utility functions for computing jets."""

from math import factorial, prod
from typing import Callable, List, Optional, Set, Tuple

from torch import Tensor, device, dtype, einsum, empty
from torch.nn import Module

# type annotation for arguments and Taylor coefficients in input and output space
Primal = Tensor
Value = Tensor
# primals and values form a tuple
PrimalAndCoefficients = Tuple[Primal, ...]
ValueAndCoefficients = Tuple[Value, ...]


def integer_partitions(n: int, I: int = 1):  # noqa: E741
    """Compute the integer partitions of a positive integer.

    Taken from: https://stackoverflow.com/a/44209393.

    Args:
        n: Positive integer.
        I: Minimum value of the partition's first entry. Default: `1`.

    Yields:
        Tuple of integers representing the integer partition.
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
    equation = ",".join(len(tensors) * ["..."]) + "->..."
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


def replicate(x: Tensor, times: int) -> Tensor:
    """Repeat a tensor along a new leading axis.

    Args:
        x: Tensor to repeat.
        times: Number of times to repeat the tensor.

    Returns:
        Tensor with a new leading axis repeated `times` times.
    """
    repeat = [times] + x.ndim * [-1]
    return x.unsqueeze(0).expand(*repeat)


def sum_vmapped(x: Tensor) -> Tensor:
    """Sum out a vmap-ed axis.

    Args:
        x: Vmap-ed tensor.

    Returns:
        Sum of the vmap-ed tensor.
    """
    return x.sum(0)


def rademacher(
    *shape: int, dtype: Optional[dtype] = None, device: Optional[device] = None
):
    """Sample from Rademacher distribution.

    Args:
        shape: Shape of the output tensor.
        dtype: Data type of the output tensor. Default: `None`.
        device: Device of the output tensor. Default: `None`.

    Returns:
        Tensor sampled from Rademacher distribution (+1 and -1 entries).
    """
    return (
        empty(*shape, dtype=dtype, device=device)
        .fill_(0.5)
        .bernoulli_()
        .mul_(2)
        .sub_(1)
    )


def get_letters(num_letters: int, blocked: Optional[Set] = None) -> List[str]:
    """Return a list of ``num_letters`` unique letters for an einsum equation.

    Args:
        num_letters: Number of letters to return.
        blocked: Set of letters that should not be used.

    Returns:
        List of ``num_letters`` unique letters.

    Raises:
        ValueError: If ``num_letters`` cannot be satisfies with einsum-supported
            letters.
    """
    if num_letters == 0:
        return []

    max_letters = 26
    blocked = set() if blocked is None else blocked
    letters = []

    for i in range(max_letters):
        letter = chr(ord("a") + i)
        if letter not in blocked:
            letters.append(letter)
            if len(letters) == num_letters:
                return letters

    raise ValueError(
        f"Ran out of letters. PyTorch's einsum supports {max_letters} letters."
        + f" Requested {num_letters}, blocked: {len(blocked)}.)"
    )
