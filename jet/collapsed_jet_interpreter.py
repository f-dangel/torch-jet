"""Interpreter-based collapsed Taylor mode automatic differentiation.

This module defines the ``CollapsedJetInterpreter``, which extends
``torch.fx.Interpreter`` to propagate ``CollapsedJetTuple`` values through a
traced graph, dispatching to collapsed jet operations from
``jet.collapsed_operations``.
"""

from torch.fx import Interpreter

from jet.collapsed_operations import COLLAPSED_MAPPING, CollapsedJetTuple


class CollapsedJetInterpreter(Interpreter):
    """Interpreter that propagates CollapsedJetTuples through a traced graph.

    For each ``call_function`` node, the interpreter checks whether any
    positional argument is a ``CollapsedJetTuple`` (i.e. a Taylor-expanded
    value). If so, it dispatches to the corresponding collapsed jet operation
    from ``jet.collapsed_operations.COLLAPSED_MAPPING``; otherwise it falls
    through to the original ATen operation. The Taylor-expansion order ``K``
    is inferred inside each collapsed jet op from its arguments (a
    ``CollapsedJetTuple`` is ``(primal, c_1, ..., c_K)``).
    """

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
            return COLLAPSED_MAPPING[target](*args)
        return super().call_function(target, args, kwargs)
