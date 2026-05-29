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


def build_input_tuples(
    primals: tuple[Any, ...],
    taylor_coeffs: tuple[tuple[Any, ...], ...],
    num_leaves: int,
    derivative_order: int,
) -> list[tuple[Tensor, ...]]:
    """Assemble per-leaf ``(primal, *coeffs)`` tuples for the jet interpreter.

    Flattens the arg-major ``primals`` and ``taylor_coeffs`` pytrees and regroups
    them by tensor leaf, so each leaf gets a tuple of its primal followed by its
    ``derivative_order`` Taylor coefficients (in increasing order).

    Args:
        primals: Tuple of primal values matching the traced function's arguments.
        taylor_coeffs: Arg-major coefficients, where ``taylor_coeffs[arg][order]``
            holds the order-1..K coefficient of each argument.
        num_leaves: Number of tensor leaves across all arguments.
        derivative_order: Order ``K`` of the Taylor expansion.

    Returns:
        A list of ``num_leaves`` tuples, each ``(primal, coeff_1, ..., coeff_K)``.
    """
    flat_primals = tree_flatten(primals)[0]
    flat_by_order = [
        [
            coefficient
            for arg_taylor_coeffs in taylor_coeffs
            for coefficient in tree_flatten(arg_taylor_coeffs[order])[0]
        ]
        for order in range(derivative_order)
    ]
    return [
        (flat_primals[i], *(coeffs_at_order[i] for coeffs_at_order in flat_by_order))
        for i in range(num_leaves)
    ]


def _replace_inplace_ops(mod: GraphModule) -> None:
    """Replace in-place operations with their out-of-place equivalents.

    Args:
        mod: The graph module to modify in place.
    """
    for node in mod.graph.nodes:
        if node.op == "call_function" and node.target in _INPLACE_TO_FUNCTIONAL:
            node.target = _INPLACE_TO_FUNCTIONAL[node.target]
