"""Utility functions for capturing compute graphs in PyTorch."""

from functools import partial
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

#: Standardized ``make_fx`` configuration used by every trace in ``jet``.
#: Fake-tensor mode skips kernel execution and just propagates shape/dtype,
#: cutting trace time substantially on deep models. ``_allow_non_fake_inputs``
#: is needed because callers' modules close over real ``nn.Parameter`` tensors.
_make_fx = partial(make_fx, tracing_mode="fake", _allow_non_fake_inputs=True)


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
    mod = _make_fx(functionalize(f))(*mock_args)
    _replace_inplace_ops(mod)
    mod.graph.eliminate_dead_code()
    mod.recompile()
    return mod


def capture_flat_graph(
    f: Callable[..., Any], mock_args: tuple[Any, ...]
) -> GraphModule:
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
        The traced graph module over flat tensor inputs.
    """
    _assert_traceable_signature(mock_args)
    flat_mocks, in_spec = tree_flatten(mock_args)

    def flat_f(*flat_tensors: Tensor) -> Any:
        return f(*tree_unflatten(list(flat_tensors), in_spec))

    return capture_graph(flat_f, *flat_mocks)


def _assert_traceable_signature(args: tuple[Any, ...]) -> None:
    """Reject the argument signature that ``make_fx`` mistraces.

    ``make_fx`` misreads a two-argument call whose first argument is tuple-rooted
    and whose second is a ``dict`` as an ``(args, kwargs)`` call and emits a
    broken input template (pytorch/pytorch#185640). Every other signature -- a
    lone ``dict``, a ``dict`` first, three or more arguments, or ``list`` +
    ``dict`` -- traces correctly, as do ``dict`` outputs.

    Args:
        args: The positional arguments that will be passed to the traced
            function (e.g. ``mock_args``).

    Raises:
        NotImplementedError: If the signature is the unsupported
            ``(tensor-or-tuple, dict)``.
    """
    if (
        len(args) == 2
        and isinstance(args[0], (Tensor, tuple))
        and isinstance(args[1], dict)
    ):
        raise NotImplementedError(
            "make_fx cannot trace a two-argument function whose first argument "
            "is a tensor/tuple and whose second is a dict, due to a codegen bug "
            "(pytorch/pytorch#185640). Work around it by putting the dict "
            "argument first, adding another argument, or bundling the arguments "
            "into a single tuple/list."
        )


def _replace_inplace_ops(mod: GraphModule) -> None:
    """Replace in-place operations with their out-of-place equivalents.

    Args:
        mod: The graph module to modify in place.
    """
    for node in mod.graph.nodes:
        if node.op == "call_function" and node.target in _INPLACE_TO_FUNCTIONAL:
            node.target = _INPLACE_TO_FUNCTIONAL[node.target]
