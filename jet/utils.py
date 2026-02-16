"""Utility functions for computing jets."""

from collections import defaultdict
from math import factorial, prod
from typing import Any

import torch
from torch import Tensor, device, dtype, empty, randn
from torch.fx import GraphModule, Node
from torch.nn import Module

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


def recursive_setattr(obj: Any, attr: str, value: Any) -> None:
    """Recursively set a nested attribute on an object.

    This function allows setting attributes that are nested within submodules or
    objects, using a dot-separated string (e.g., 'foo.bar.baz').

    If the object is an `nn.Module` and the attribute is a `Tensor`, it registers the
    tensor as a buffer.

    Args:
        obj: The root object on which to set the attribute.
        attr: Dot-separated string specifying the attribute path.
        value: The value to set for the nested attribute.

    Raises:
        RuntimeError: If the attribute already exists.
    """
    parts = attr.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)

    if hasattr(obj, parts[-1]):
        raise RuntimeError(
            f"Attribute {parts[-1]!r} already exists in {'.'.join(parts[:-1])!r}. "
        )
    if isinstance(obj, Module) and isinstance(value, Tensor):
        obj.register_buffer(parts[-1], value)
    else:
        setattr(obj, parts[-1], value)


def recursive_delattr(obj: Any, attr: str) -> None:
    """Recursively delete a nested attribute from an object.

    This function allows deleting attributes that are nested within submodules or
    objects, using a dot-separated string (e.g., 'foo.bar.baz').

    Args:
        obj: The root object from which to delete the attribute.
        attr: Dot-separated string specifying the attribute path.
    """
    parts = attr.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)

    delattr(obj, parts[-1])


def print_tensor_constants_and_shapes(mod: GraphModule):
    """Print names, shapes, and usage counts of all tensor constants in a graph module.

    Args:
        mod: The GraphModule to inspect.
    """
    # Count usages of each get_attr node by target name
    usage_counts = defaultdict(int)
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


def recursive_hasattr(obj: Any, attr: str) -> bool:
    """Recursively check if a nested attribute exists in an object.

    This function allows checking for attributes that are nested within submodules or
    objects, using a dot-separated string (e.g., 'foo.bar.baz').

    Args:
        obj: The root object to check for the attribute.
        attr: Dot-separated string specifying the attribute path.

    Returns:
        True if the nested attribute exists, False otherwise.
    """
    try:
        for part in attr.split("."):
            obj = getattr(obj, part)
        return True
    except AttributeError:
        return False
