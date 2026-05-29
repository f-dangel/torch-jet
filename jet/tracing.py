"""Utility functions for capturing compute graphs in PyTorch."""

from typing import Any, Callable

from torch import Tensor, ops
from torch.func import functionalize
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import Module
from torch.utils._pytree import tree_flatten, tree_unflatten

# Map in-place ATen ops to their out-of-place equivalents.
_INPLACE_TO_FUNCTIONAL = {
    ops.aten.squeeze_.dim: ops.aten.squeeze.dim,
}


def capture_graph(
    f: Module | Callable[..., Any] | GraphModule,
    *mock_args: Tensor,
) -> GraphModule:
    """Capture the compute graph of a function using make_fx.

    The function is wrapped with ``functionalize`` and in-place operations are
    replaced with their out-of-place equivalents, ensuring a purely functional
    graph that is safe for transformations like common subexpression elimination.

    Args:
        f: The (graph) module or callable to trace.
        *mock_args: Mock input tensors for tracing. Only shapes and dtypes matter.

    Returns:
        The traced module with the captured compute graph.
    """
    mod = make_fx(functionalize(f))(*mock_args)
    _replace_inplace_ops(mod)
    mod.graph.eliminate_dead_code()
    mod.recompile()
    return mod


def capture_flat_graph(
    f: Callable[..., Any], mock_args: tuple[Any, ...]
) -> tuple[GraphModule, int]:
    """Capture the compute graph of ``f`` over its flattened pytree leaves.

    ``make_fx`` creates one symbolic proxy per positional tensor argument and
    cannot trace through nested pytree containers. This flattens ``mock_args``
    into tensor leaves, wraps ``f`` in a shim that unflattens them back into the
    original structure, and traces that shim.

    Args:
        f: Function to trace. May accept pytrees of tensors as positional args.
        mock_args: Mock inputs (pytrees of tensors) matching ``f``'s positional
            args, provided as a tuple. Only shapes and dtypes matter.

    Returns:
        A tuple ``(mod, num_leaves)`` with the traced graph module over flat
        tensor inputs and the number of tensor leaves.
    """
    flat_mocks, in_spec = tree_flatten(mock_args)

    def flat_f(*flat_tensors: Tensor) -> Any:
        return f(*tree_unflatten(list(flat_tensors), in_spec))

    return capture_graph(flat_f, *flat_mocks), len(flat_mocks)


def _replace_inplace_ops(mod: GraphModule) -> None:
    """Replace in-place operations with their out-of-place equivalents.

    Args:
        mod: The graph module to modify in place.
    """
    for node in mod.graph.nodes:
        if node.op == "call_function" and node.target in _INPLACE_TO_FUNCTIONAL:
            node.target = _INPLACE_TO_FUNCTIONAL[node.target]
