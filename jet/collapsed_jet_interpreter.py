"""Interpreter-based collapsed Taylor mode automatic differentiation.

This module defines the ``CollapsedJetInterpreter``, which extends
``torch.fx.Interpreter`` to propagate ``CollapsedJetTuple`` values through a
traced graph, dispatching to collapsed jet operations from
``jet.collapsed_operations``.
"""

from torch import Tensor, zeros_like
from torch.fx import GraphModule, Interpreter
from torch.utils._pytree import tree_flatten, tree_unflatten

from jet.collapsed_operations import COLLAPSED_MAPPING, CollapsedJetTuple


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
