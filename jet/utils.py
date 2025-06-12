"""Utility functions for computing jets."""

from collections import defaultdict
from math import factorial, prod
from typing import Any, Callable, Optional, Tuple

from torch import Tensor, device, dtype, empty
from torch.fx import GraphModule, Node
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
    return x.unsqueeze(pos).expand(*repeat)


def sum_vmapped(x: Tensor, pos: int = 0) -> Tensor:
    """Sum out a vmap-ed axis.

    Args:
        x: Vmap-ed tensor.
        pos: Position of the vmap-ed axis to sum out. Default: `0`.

    Returns:
        Sum of the vmap-ed tensor.
    """
    return x.sum(pos)


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


def recursive_getattr(obj: Any, attr: str) -> Any:
    """Recursively retrieve a nested attribute from an object.

    This function allows access to attributes that are nested within submodules or
    objects, using a dot-separated string (e.g., 'foo.bar.baz'). It is useful for
    retrieving parameters or buffers from submodules in a torch.fx.GraphModule, where
    attribute names may refer to nested modules (e.g., 'layer1.0.weight').

    Args:
        obj: The root object from which to retrieve the attribute.
        attr: Dot-separated string specifying the attribute path.

    Returns:
        The value of the nested attribute.
    """
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def print_tensor_constants_and_shapes(mod: GraphModule):
    """Print names, shapes, and usage counts of all tensor constants in a graph module.

    Args:
        mod: The GraphModule to inspect.
    """
    # Count usages of each get_attr node by target name
    usage_counts = defaultdict(int)  # noqa: B910
    for node in mod.graph.nodes:
        for arg in node.args:
            if isinstance(arg, Node) and arg.op == "get_attr":
                usage_counts[arg.target] += 1
        for kwarg in node.kwargs.values():
            if isinstance(kwarg, Node) and kwarg.op == "get_attr":
                usage_counts[kwarg.target] += 1

    # Print the names, shapes, usage counts, and total number of elements of tensor
    # constants
    total = 0
    for node in mod.graph.nodes:
        if node.op != "get_attr":
            continue
        if "_tensor_constant" not in node.target:
            continue

        tensor = recursive_getattr(mod, node.target)
        if not isinstance(tensor, Tensor):
            continue

        count = usage_counts[node.target]
        total += tensor.numel()
        print(f"Name: {node.target}, Shape: {tuple(tensor.shape)}, Usages: {count}")

    print(f"Total number of elements in tensor constants: {total}")
