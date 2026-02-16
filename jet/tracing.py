"""Utility functions for capturing compute graphs in PyTorch."""

from typing import Callable

from torch import Tensor
from torch.func import functionalize
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import Module

from jet.utils import Primal, Value


def capture_graph(
    f: Module | Callable[[Primal], Value] | GraphModule,
    example_input: Tensor | None = None,
) -> GraphModule:
    """Capture the compute graph of a function using make_fx.

    The function is wrapped with ``functionalize`` to convert in-place operations
    to their out-of-place equivalents, ensuring a purely functional graph that is
    safe for transformations like common subexpression elimination.

    Args:
        f: The (graph) module or callable to trace.
        example_input: A concrete example input tensor for tracing. Required
            unless ``f`` is already a ``GraphModule``.

    Returns:
        The traced module with the captured compute graph.

    Raises:
        ValueError: If ``example_input`` is ``None`` and ``f`` is not a
            ``GraphModule``.
    """
    if isinstance(f, GraphModule):
        return f

    if example_input is None:
        raise ValueError("example_input is required for tracing with make_fx.")

    mod = make_fx(functionalize(f))(example_input)
    mod.graph.eliminate_dead_code()
    mod.recompile()
    return mod
