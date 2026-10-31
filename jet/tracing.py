"""Utility functions for capturing compute graphs in PyTorch."""

from functools import partial
from typing import Any, Callable

from torch import Tensor, ops
from torch._decomp import decomposition_table
from torch.func import functionalize
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from jet.utils import PyTree

# Map in-place ATen ops to their out-of-place equivalents.
_INPLACE_TO_FUNCTIONAL = {
    ops.aten.squeeze_.dim: ops.aten.squeeze.dim,
    ops.aten.as_strided_.default: ops.aten.as_strided.default,
    ops.aten.add_.Scalar: ops.aten.add.Scalar,
}

#: Backward ops that are traced through their ATen decomposition instead of
#: getting a jet rule. Each decomposition is the formula a hand-written rule
#: would compute anyway, and every op in it has a rule. Only graphs that contain
#: one of these ops change. The cudnn/miopen batch-norm backward variants are
#: deliberately absent: their decomposition takes the training path, which
#: reads batch statistics the forward rule does not provide.
DECOMPOSED = {
    op: decomposition_table[op]
    for op in (
        ops.aten.native_layer_norm_backward.default,
        ops.aten.native_batch_norm_backward.default,
        ops.aten._softmax_backward_data.default,
        ops.aten._log_softmax_backward_data.default,
    )
}

#: Fake-tensor mode skips kernel execution and just propagates shape/dtype,
#: cutting trace time substantially on deep models. ``_allow_non_fake_inputs``
#: is needed because callers' modules close over real ``nn.Parameter`` tensors.
_make_fx = partial(
    make_fx,
    tracing_mode="fake",
    _allow_non_fake_inputs=True,
    decomposition_table=DECOMPOSED,
)


def capture_graph(
    f: Callable[[*tuple[PyTree[Tensor], ...]], PyTree[Tensor]],
    mock_args: tuple[PyTree[Tensor], ...],
) -> tuple[GraphModule, TreeSpec]:
    """Capture the compute graph of ``f`` as a ``GraphModule``.

    The returned ``GraphModule``'s ``forward`` takes the flat tensor leaves
    of ``mock_args`` as positional arguments -- ``make_fx`` cannot trace
    through pytree containers, so ``mock_args`` is flattened and ``f`` is
    traced over those flat leaves. To call the captured graph with ``f``'s
    original pytree shape, use the second return value, ``in_spec``, to
    flatten new arguments in the order the graph expects:

    ```python
    mod, in_spec = capture_graph(f, mock_args)
    out = mod(*in_spec.flatten_up_to(args))
    ```

    Output structure is passed through unchanged -- whatever ``f`` returns
    (single tensor, tuple, dict, arbitrary pytree), ``make_fx``'s pytree
    codegen reconstructs on each call.

    Two small passes clean up the traced graph: in-place ops are replaced by
    their out-of-place equivalents, and ``aten.view`` by ``aten.reshape``.
    ``make_fx`` bakes the trace-time strides into every ``view``; a compiler
    (e.g. inductor) may lay out an upstream tensor differently and then refuse
    the ``view``, whereas ``reshape`` views where it can and copies otherwise.

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

    Examples:
        >>> from torch import allclose, ones, sin, zeros
        >>> from jet import capture_graph
        >>> mod, in_spec = capture_graph(sin, (zeros(3),))
        >>> x = ones(3)
        >>> allclose(mod(*in_spec.flatten_up_to((x,))), sin(x))
        True
    """
    if not isinstance(mock_args, tuple):
        raise TypeError(
            f"mock_args must be a tuple of f's positional arguments, got "
            f"{type(mock_args).__name__}; wrap a single argument as ``(x,)``."
        )
    _assert_traceable_signature(mock_args)
    flat_mocks, in_spec = tree_flatten(mock_args)

    def flat_f(*flat_tensors: Tensor) -> PyTree[Tensor]:
        return f(*tree_unflatten(list(flat_tensors), in_spec))

    mod = _make_fx(functionalize(flat_f))(*flat_mocks)
    _replace_inplace_ops(mod)
    _replace_views(mod)
    mod.graph.eliminate_dead_code()
    mod.recompile()
    return mod, in_spec


def freeze(
    f: Callable[[*tuple[PyTree[Tensor], ...]], PyTree[Tensor]],
    args: tuple[PyTree[Tensor], ...],
) -> Callable[[*tuple[PyTree[Tensor], ...]], PyTree[Tensor]]:
    """Trace ``f`` once on ``args`` and return a callable on the same pytree.

    The callable returned by :func:`jet.jet` interprets the captured graph of
    ``f`` node by node at call time, so the Taylor order, the number of
    directions, and the leaf shapes may change between calls. Dynamo cannot
    trace that interpreter, so ``torch.compile`` does not work on it. This
    helper captures the graph of the jet computation for one fixed set of
    arguments and wraps it in a callable that takes the same pytree, which
    ``torch.compile`` handles.

    Everything is fixed at trace time: the pytree structure of ``args`` (for
    jet arguments: the order ``K``, the direction count ``R``, and collapsed
    or not), and every leaf's shape, dtype, and device. Calling the frozen
    function with different arguments raises. The plain jet callable stays
    flexible.

    Args:
        f: Callable to trace (e.g. a jet function). May accept pytrees of
            tensors as positional args.
        args: Positional arguments (pytrees of tensors) to trace ``f`` on,
            provided as a tuple.

    Returns:
        A callable that accepts arguments of the same pytree structure,
        shapes, dtypes, and devices as ``args`` and runs the captured graph.

    Raises:
        ValueError: At call time, if the arguments differ from ``args`` in
            pytree structure or in a leaf's shape, dtype, or device.

    Examples:
        >>> from torch import allclose, ones, sin, zeros
        >>> from jet import freeze
        >>> frozen = freeze(sin, (zeros(3),))
        >>> allclose(frozen(ones(3)), sin(ones(3)))
        True
    """
    mod, in_spec = capture_graph(f, args)
    leaves = tree_flatten(args)[0]
    signature = [(t.shape, t.dtype, t.device) for t in leaves]

    def frozen(*new_args: PyTree[Tensor]) -> PyTree[Tensor]:
        new_leaves, new_spec = tree_flatten(new_args)
        if new_spec != in_spec:
            raise ValueError(
                f"frozen function expects {len(leaves)} tensor leaves with the "
                f"trace-time pytree structure, got {len(new_leaves)} leaves with "
                "a different structure (order, direction count, or nesting)."
            )
        for i, (t, (shape, dtype, device)) in enumerate(zip(new_leaves, signature)):
            if (t.shape, t.dtype, t.device) != (shape, dtype, device):
                raise ValueError(
                    f"leaf {i} has shape {tuple(t.shape)}, dtype {t.dtype}, "
                    f"device {t.device}; expected shape {tuple(shape)}, dtype "
                    f"{dtype}, device {device}."
                )
        return mod(*new_leaves)

    return frozen


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


def _replace_views(mod: GraphModule) -> None:
    """Replace ``aten.view`` with ``aten.reshape`` (same result, never fails).

    Args:
        mod: The graph module to modify in place.
    """
    for node in mod.graph.nodes:
        if node.op == "call_function" and node.target is ops.aten.view.default:
            node.target = ops.aten.reshape.default
