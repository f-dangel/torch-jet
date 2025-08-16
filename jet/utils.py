"""Utility functions for computing jets."""

from collections import defaultdict
from inspect import signature
from math import factorial, prod
from typing import Any, Callable

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


def separate_args_and_kwargs(
    f: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Return mandatory arguments in positional and optional arguments in keyword form.

    This function standardizes the arguments and keyword arguments of a callable to
    ensure that mandatory arguments are passed positionally and optional arguments are
    passed as keyword arguments. It uses the signature of the callable to determine
    which arguments are mandatory and which are optional.

    Args:
        f: The callable whose signature is to be analyzed.
        args: Positional arguments passed to the callable.
        kwargs: Keyword arguments passed to the callable.

    Returns:
        A tuple containing:
        - A tuple of positional arguments that are mandatory
        - A dictionary of optional arguments, where keys are argument names and values
          are the corresponding values from `kwargs` or their default values if not
          provided.
    """
    sig = signature(f)
    bound_args = sig.bind(*args, **kwargs)

    positional = tuple(
        bound_args.arguments[name]
        for name, param in sig.parameters.items()
        if name in bound_args.arguments and param.default is param.empty
    )
    optional = {
        name: (
            bound_args.arguments[name]
            if name in bound_args.arguments
            else param.default
        )
        for name, param in sig.parameters.items()
        if param.default is not param.empty
    }
    return positional, optional


def standardize_signature(node: Node, verbose: bool = False):
    """Standardize the args and kwargs of a node (inplace).

    This function modifies the node's args and kwargs to match the signature of the
    function it calls. It ensures that positional arguments are grouped together and
    keyword arguments are separated, with defaults applied where necessary.

    E.g., the call `replicate(x, times=4)` is standardized to `replicate(x, 4, pos=0)`.

    Args:
        node: The node to standardize. It should be a call_function node with a target
            that has a signature.
        verbose: If True, print the node's args and kwargs before and after.
            Default: `False`.

    Raises:
        ValueError: If the node is not a call_function node.
    """
    if node.op != "call_function":
        raise ValueError(f"{node=} is not a call_function node {node.op=}.")

    if verbose:
        print(f"Standardizing {node=}: {node.args=}, {node.kwargs=} ->", end=" ")

    new_args, new_kwargs = separate_args_and_kwargs(node.target, node.args, node.kwargs)
    node.args, node.kwargs = new_args, new_kwargs

    if verbose:
        print(f"{node.args=}, {node.kwargs=}.")


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
