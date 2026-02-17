"""Utility functions for computing jets."""

from math import factorial, prod

import torch
from torch import Tensor, device, dtype, empty, randn

# type annotation for arguments and Taylor coefficients in input and output space
Primal = Tensor
Value = Tensor
# primals and values form a tuple
PrimalAndCoefficients = tuple[Primal, ...]
ValueAndCoefficients = tuple[Value, ...]


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


def multiplicity(sigma: tuple[int, ...]) -> float:
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


@torch.library.custom_op("jet::replicate", mutates_args=())
def replicate(x: Tensor, times: int, pos: int = 0) -> Tensor:
    """Repeat a tensor along a new axis.

    Args:
        x: Tensor to repeat.
        times: Number of times to repeat the tensor.
        pos: Position of the new axis. Default: `0`.

    Returns:
        Tensor with a new axis repeated `times` times.
    """
    repeat = x.ndim * [-1]
    repeat = repeat[:pos] + [times] + repeat[pos:]
    return x.unsqueeze(pos).expand(*repeat).clone()


@replicate.register_fake
def _replicate_fake(x: Tensor, times: int, pos: int = 0) -> Tensor:
    new_shape = list(x.shape)
    new_shape.insert(pos, times)
    return x.new_empty(new_shape)


def _replicate_setup_context(ctx, inputs, output):
    _, _, pos = inputs
    ctx.pos = pos


def _replicate_backward(ctx, grad_output):
    return grad_output.sum(ctx.pos), None, None


replicate.register_autograd(_replicate_backward, setup_context=_replicate_setup_context)


@torch.library.custom_op("jet::sum_vmapped", mutates_args=())
def sum_vmapped(x: Tensor, pos: int = 0) -> Tensor:
    """Sum out a vmap-ed axis.

    Args:
        x: Vmap-ed tensor.
        pos: Position of the vmap-ed axis to sum out. Default: `0`.

    Returns:
        Sum of the vmap-ed tensor.
    """
    return x.sum(pos)


@sum_vmapped.register_fake
def _sum_vmapped_fake(x: Tensor, pos: int = 0) -> Tensor:
    new_shape = list(x.shape)
    del new_shape[pos]
    return x.new_empty(new_shape)


def _sum_vmapped_setup_context(ctx, inputs, output):
    x, pos = inputs
    ctx.pos = pos
    ctx.x_shape = x.shape


def _sum_vmapped_backward(ctx, grad_output):
    return grad_output.unsqueeze(ctx.pos).expand(ctx.x_shape), None


sum_vmapped.register_autograd(
    _sum_vmapped_backward, setup_context=_sum_vmapped_setup_context
)


def rademacher(*shape: int, dtype: dtype | None = None, device: device | None = None):
    """Sample from Rademacher distribution.

    Args:
        shape: Shape of the output tensor.
        dtype: Data type of the output tensor. Default: `None`.
        device: Device of the output tensor. Default: `None`.

    Returns:
        Tensor sampled from Rademacher distribution (+1 and -1 entries).
    """
    return (
        empty(*shape, dtype=dtype, device=device).fill_(0.5).bernoulli().mul_(2).sub_(1)
    )


def sample(x_meta: Tensor, distribution: str, shape: tuple[int, ...]) -> Tensor:
    """Sample a random tensor with the same dtype and device as a given tensor.

    Args:
        x_meta: Tensor whose dtype and device are to be matched.
        distribution: Distribution to sample from. Supported: "normal", "rademacher".
        shape: Shape of the output tensor.

    Returns:
        Sampled tensor.
    """
    sample_func = {"normal": randn, "rademacher": rademacher}[distribution]
    return sample_func(*shape, dtype=x_meta.dtype, device=x_meta.device)
