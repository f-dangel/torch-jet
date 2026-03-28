"""Interpreter-based collapsed Taylor mode automatic differentiation.

This module defines the ``CollapsedJetInterpreter``, which extends
``torch.fx.Interpreter`` to propagate ``CollapsedJetTuple`` values through a
traced graph, dispatching to collapsed jet operations from
``jet.collapsed_operations``.
"""

from typing import Callable

from torch import Tensor, zeros_like
from torch.fx import GraphModule, Interpreter
from torch.utils._pytree import tree_flatten, tree_unflatten

from jet.collapsed_operations import COLLAPSED_MAPPING, CollapsedJetTuple
from jet.tracing import capture_graph
from jet.utils import Value


class CollapsedJetInterpreter(Interpreter):
    """Interpreter that propagates CollapsedJetTuples through a traced graph."""

    def __init__(self, module: GraphModule, derivative_order: int):
        """Initialize with a graph module and derivative order."""
        super().__init__(module)
        self.derivative_order = derivative_order

    def placeholder(self, target, args, kwargs):
        """Wrap placeholder values in a CollapsedJetTuple."""
        value = super().placeholder(target, args, kwargs)
        return CollapsedJetTuple(value)

    def call_function(self, target, args, kwargs):
        """Dispatch to collapsed jet rules when arguments contain CollapsedJetTuples."""
        has_jet_arg = any(isinstance(a, CollapsedJetTuple) for a in args)
        if has_jet_arg:
            if target not in COLLAPSED_MAPPING:
                raise NotImplementedError(f"No collapsed jet rule for {target}.")
            return COLLAPSED_MAPPING[target](
                *args, derivative_order=self.derivative_order
            )
        return super().call_function(target, args, kwargs)


def _is_collapsed_or_tensor(x):
    return isinstance(x, (CollapsedJetTuple, Tensor))


def _transpose_collapsed_output(result, derivative_order):
    """Transpose pytree-of-CollapsedJetTuples into tuple-of-pytrees."""
    flat, out_spec = tree_flatten(result, is_leaf=_is_collapsed_or_tensor)
    k = derivative_order + 1
    outputs = []
    for order in range(k):
        flat_order = [
            jt[order]
            if isinstance(jt, CollapsedJetTuple)
            else (jt if order == 0 else zeros_like(jt))
            for jt in flat
        ]
        outputs.append(tree_unflatten(flat_order, out_spec))
    return tuple(outputs)


def collapsed_jet(
    f: Callable[..., Value],
    derivative_order: int,
    mock_args: tuple,
    verbose: bool = False,
) -> Callable[..., tuple[Value, ...]]:
    """Overload f with collapsed Taylor-mode equivalent.

    Same API as ``jet()``, but expects mixed-shape series:
      - series[0..K-2]: tensors with leading batch dim R
      - series[K-1]: tensors without batch dim (collapsed)

    The K-th output coefficient is automatically collapsed (summed over
    directions), so no ``.sum(0)`` or PullSum graph rewrites are needed.
    """
    flat_mocks, in_spec = tree_flatten(mock_args)
    num_leaves = len(flat_mocks)

    def flat_f(*flat_tensors):
        args = tree_unflatten(list(flat_tensors), in_spec)
        return f(*args)

    mod = capture_graph(flat_f, *flat_mocks)
    if verbose:
        print(f"Traced graph:\n{mod.graph}")

    interp = CollapsedJetInterpreter(mod, derivative_order)

    def cjet_f(primals, series):
        flat_primals = tree_flatten(primals)[0]
        flat_series = [tree_flatten(s)[0] for s in series]
        input_tuples = [
            (flat_primals[i], *(fs[i] for fs in flat_series)) for i in range(num_leaves)
        ]
        result = interp.run(*input_tuples)
        all_orders = _transpose_collapsed_output(result, derivative_order)
        return all_orders[0], all_orders[1:]

    return cjet_f
