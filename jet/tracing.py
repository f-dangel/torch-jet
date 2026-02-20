"""Utility functions for capturing compute graphs in PyTorch."""

from typing import Callable

from torch import Tensor, ops
from torch.func import functionalize
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import Module

from jet.utils import Primal, Value

# Map in-place ATen ops to their out-of-place equivalents.
_INPLACE_TO_FUNCTIONAL = {
    ops.aten.squeeze_.dim: ops.aten.squeeze.dim,
}


def capture_graph(
    f: Module | Callable[[Primal], Value] | GraphModule,
    mock_x: Tensor,
) -> GraphModule:
    """Capture the compute graph of a function using make_fx.

    The function is wrapped with ``functionalize`` and in-place operations are
    replaced with their out-of-place equivalents, ensuring a purely functional
    graph that is safe for transformations like common subexpression elimination.

    Args:
        f: The (graph) module or callable to trace.
        mock_x: A mock input tensor for tracing. Does not need to be the actual
            input; only the shape and dtype matter.

    Returns:
        The traced module with the captured compute graph.
    """
    mod = make_fx(functionalize(f))(mock_x)
    _replace_inplace_ops(mod)
    mod.graph.eliminate_dead_code()
    mod.recompile()
    return mod


def _replace_inplace_ops(mod: GraphModule) -> None:
    """Replace in-place operations with their out-of-place equivalents.

    Args:
        mod: The graph module to modify in place.
    """
    for node in mod.graph.nodes:
        if node.op == "call_function" and node.target in _INPLACE_TO_FUNCTIONAL:
            node.target = _INPLACE_TO_FUNCTIONAL[node.target]
