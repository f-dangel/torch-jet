"""Interpreter-based Taylor mode automatic differentiation (jets).

This module defines the `JetInterpreter`, which extends `torch.fx.Interpreter`
to execute a traced PyTorch computation graph while substituting ATen operations
with their Taylor mode (jet) equivalents on-the-fly. Unlike the graph-rewriting
approach of a `Transformer`, the interpreter propagates real values (or proxy
tensors under `make_fx`) and uses `isinstance` against the chosen jet type to
distinguish Taylor-expanded arguments from constants.

The same interpreter handles both standard and collapsed Taylor mode via the
``collapsed`` constructor flag, which selects the dispatch table
(``MAPPING`` vs ``COLLAPSED_MAPPING``) and the placeholder wrapper
(``JetTuple`` vs ``CollapsedJetTuple``).
"""

from typing import Any

from torch.fx import GraphModule, Interpreter
from torch.fx.node import Argument, Target

from jet.collapsed_operations import COLLAPSED_MAPPING, CollapsedJetTuple
from jet.operations import MAPPING, JetTuple


class JetInterpreter(Interpreter):
    """Interpreter that swaps in jet operations during execution.

    For each ``call_function`` node, the interpreter checks whether any
    positional argument is a jet tuple (a Taylor-expanded value). If so, it
    dispatches to the corresponding jet operation from the configured mapping;
    otherwise it falls through to the original ATen operation. The Taylor-
    expansion order ``K`` is inferred inside each jet op from its arguments (a
    jet tuple is ``(primal, c_1, ..., c_K)``).

    Args:
        module: The traced computation graph module to interpret.
        collapsed: If ``True``, propagate ``CollapsedJetTuple`` values and
            dispatch via ``COLLAPSED_MAPPING`` (collapsed Taylor mode, where
            the highest-order coefficient is already summed over directions).
            If ``False`` (default), propagate ``JetTuple`` values and dispatch
            via ``MAPPING`` (standard Taylor mode).
    """

    def __init__(self, module: GraphModule, collapsed: bool = False) -> None:
        """Initialize the JetInterpreter."""
        super().__init__(module)
        self.jet_type: type = CollapsedJetTuple if collapsed else JetTuple
        self.mapping: dict = COLLAPSED_MAPPING if collapsed else MAPPING
        self.label: str = "collapsed jet" if collapsed else "jet"

    def placeholder(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Wrap each placeholder value in ``self.jet_type``."""
        value = super().placeholder(target, args, kwargs)
        return self.jet_type(value)

    def call_function(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Dispatch to ``self.mapping[target]`` when any arg is a jet tuple.

        Args:
            target: The function or callable to execute.
            args: Positional arguments of the node.
            kwargs: Keyword arguments of the node.

        Returns:
            The result of the jet operation (a jet tuple) or the original op.

        Raises:
            NotImplementedError: If a Taylor-dependent op has no jet rule, or
                if the node has non-empty ``kwargs``.
        """
        # TODO Only checks top-level args. Jets nested inside tuple/list/dict
        # args (e.g. for aten.stack, aten.cat) will be missed, causing a
        # TypeError instead of a clear NotImplementedError.
        has_jet_arg = any(isinstance(a, self.jet_type) for a in args)
        if has_jet_arg:
            if target not in self.mapping:
                raise NotImplementedError(
                    f"No {self.label} rule for {target}. "
                    "Please file an issue or add a rule."
                )
            if kwargs:
                raise NotImplementedError(
                    f"{self.label.capitalize()} dispatch does not support kwargs "
                    f"for {target} (got {kwargs})."
                )
            return self.mapping[target](*args)
        return super().call_function(target, args, kwargs)
