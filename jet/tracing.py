"""Utility functions for capturing compute graphs in PyTorch."""

from functools import partial
from typing import Any, Callable

from torch import Tensor, ops
from torch.func import functionalize
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

# Map in-place ATen ops to their out-of-place equivalents.
_INPLACE_TO_FUNCTIONAL = {
    ops.aten.squeeze_.dim: ops.aten.squeeze.dim,
}

#: Fake-tensor mode skips kernel execution and just propagates shape/dtype,
#: cutting trace time substantially on deep models. ``_allow_non_fake_inputs``
#: is needed because callers' modules close over real ``nn.Parameter`` tensors.
_make_fx = partial(make_fx, tracing_mode="fake", _allow_non_fake_inputs=True)


def capture_graph(
    f: Callable[..., Any],
    mock_args: tuple[Any, ...],
) -> tuple[GraphModule, TreeSpec]:
    """Capture the compute graph of ``f`` as a ``GraphModule``.

    The returned ``GraphModule``'s ``forward`` takes the flat tensor leaves
    of ``mock_args`` as positional arguments -- ``make_fx`` cannot trace
    through pytree containers, so ``mock_args`` is flattened and ``f`` is
    traced over those flat leaves. To call the captured graph with ``f``'s
    original pytree shape, use the second return value, ``in_spec``, to
    flatten new arguments in the order the graph expects::

        mod, in_spec = capture_graph(f, mock_args)
        out = mod(*in_spec.flatten_up_to(args))

    Output structure is passed through unchanged -- whatever ``f`` returns
    (single tensor, tuple, dict, arbitrary pytree), ``make_fx``'s pytree
    codegen reconstructs on each call.

    Args:
        f: Callable to trace (plain function, ``nn.Module``, ``GraphModule``,
            etc.). May accept pytrees of tensors as positional args.
        mock_args: Mock inputs (pytrees of tensors) matching ``f``'s positional
            args, provided as a tuple. Only shapes and dtypes matter.

    Returns:
        A pair ``(mod, in_spec)`` where ``mod`` is a ``GraphModule`` whose
        forward takes the flat tensor leaves of ``mock_args``, and
        ``in_spec`` is the ``TreeSpec`` of ``mock_args``. Use
        ``in_spec.flatten_up_to(args)`` to obtain the flat leaves in the
        order ``mod`` expects.

    Raises:
        TypeError: If ``mock_args`` is not a ``tuple``. Wrap a single
            positional argument as ``(x,)``.
        NotImplementedError: If ``mock_args`` hits the unsupported
            ``(tensor-or-tuple, dict)`` signature; see
            :func:`_assert_traceable_signature`.
    """
    if not isinstance(mock_args, tuple):
        raise TypeError(
            f"mock_args must be a tuple of f's positional arguments, got "
            f"{type(mock_args).__name__}; wrap a single argument as ``(x,)``."
        )
    _assert_traceable_signature(mock_args)
    flat_mocks, in_spec = tree_flatten(mock_args)

    def flat_f(*flat_tensors: Tensor) -> Any:
        return f(*tree_unflatten(list(flat_tensors), in_spec))

    mod = _make_fx(functionalize(flat_f))(*flat_mocks)
    _replace_inplace_ops(mod)
    mod.graph.eliminate_dead_code()
    mod.recompile()
    return mod, in_spec


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
