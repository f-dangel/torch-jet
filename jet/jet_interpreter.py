"""Interpreter-based Taylor mode automatic differentiation (jets).

This module defines the `JetInterpreter`, which extends `torch.fx.Interpreter`
to execute a traced PyTorch computation graph while substituting ATen operations
with their Taylor mode (jet) equivalents on-the-fly. Unlike the graph-rewriting
approach of a `Transformer`, the interpreter propagates real values (or proxy
tensors under `make_fx`) and uses `isinstance(arg, JetTuple)` to distinguish
Taylor-expanded arguments from constants.
"""

from typing import Any

from torch.fx import GraphModule, Interpreter
from torch.fx.node import Argument, Target

from jet.operations import MAPPING, JetTuple


class JetInterpreter(Interpreter):
    """Interpreter that swaps in jet operations during execution.

    For each ``call_function`` node, the interpreter checks whether any
    positional argument is a ``JetTuple`` (i.e. a Taylor-expanded value). If so,
    it dispatches to the corresponding jet operation from
    ``jet.operations.MAPPING``; otherwise it falls through to the original
    ATen operation.

    Args:
        module: The traced computation graph module to interpret.
        derivative_order: The order of the Taylor expansion.
    """

    def __init__(self, module: GraphModule, derivative_order: int):
        """Initialize the JetInterpreter.

        Args:
            module: The traced computation graph module to interpret.
            derivative_order: The order of the Taylor expansion.
        """
        super().__init__(module)
        self.derivative_order = derivative_order

    def placeholder(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> JetTuple:
        """Wrap placeholder values in a JetTuple.

        Args:
            target: The name of the placeholder.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            The placeholder value wrapped in a JetTuple.
        """
        value = super().placeholder(target, args, kwargs)
        return JetTuple(value)

    def call_function(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Execute a function node, substituting jet operations when needed.

        Args:
            target: The function or callable to execute.
            args: Positional arguments of the node.
            kwargs: Keyword arguments of the node.

        Returns:
            The result of the jet operation (a JetTuple) or the original operation.

        Raises:
            NotImplementedError: If a Taylor-dependent operation has no jet rule,
                or if the node has non-empty kwargs.
        """
        # TODO Only checks top-level args. JetTuples nested inside
        # tuple/list/dict args (e.g. for aten.stack, aten.cat) will be missed,
        # causing a TypeError instead of a clear NotImplementedError.
        has_jet_arg = any(isinstance(a, JetTuple) for a in args)
        if has_jet_arg:
            if target not in MAPPING:
                raise NotImplementedError(
                    f"No jet rule for {target}. Please file an issue or add a rule."
                )
            if kwargs:
                raise NotImplementedError(
                    f"Jet dispatch does not support kwargs for {target} "
                    f"(got {kwargs})."
                )
            return MAPPING[target](*args, derivative_order=self.derivative_order)
        return super().call_function(target, args, kwargs)
