"""Graph transformation for Taylor mode automatic differentiation (jets).

This module defines the `JetTransformer`, which extends
`torch.fx.Transformer` to replace operations in a traced PyTorch
computation graph with their Taylor mode (jet) equivalents. The transformer
uses dependency information to distinguish nodes that depend on placeholder
inputs from those that depend only on constants and substitutes the
corresponding jet operations from `jet.operations.MAPPING`.
"""

from torch.fx import GraphModule, Proxy, Transformer
from torch.fx.node import Argument, Target
from torch.fx.traceback import get_current_meta

from jet.operations import MAPPING, JetInfo


class JetTransformer(Transformer):
    """Transformer that replaces nodes with their Taylor mode (jet) equivalents.

    The `JetTransformer` inspects each node in a traced
    `torch.fx.GraphModule` and determines whether it depends on
    placeholder inputs (i.e., variables) or only on constants. Nodes that
    depend only on constants are left unchanged, while those depending on
    placeholders are replaced by the corresponding jet operations defined
    in `jet.operations.MAPPING`.
    """

    def __init__(
        self,
        module: GraphModule,
        derivative_order: int,
        dependent_on_placeholders: set[str],
        dependent_on_constants: set[str],
    ):
        """Initialize the JetTransformer.

        Sets up the transformer for converting a traced computation graph into
        its Taylor mode (jet) equivalent. The transformer tracks which nodes
        depend on placeholders (inputs) and which depend only on constants to
        determine how each operation should be replaced.

        Args:
            module: The traced computation graph module to be transformed.
            derivative_order: The order of the Taylor expansion.
            dependent_on_placeholders: set of node names that depend on
                placeholder (input) nodes.
            dependent_on_constants: set of node names that depend only on
                constant (attribute) nodes.
        """
        super().__init__(module)
        self.derivative_order = derivative_order
        self.dependent_on_placeholders = dependent_on_placeholders
        self.dependent_on_constants = dependent_on_constants

    def _is_taylor(
        self, target: Target, args: tuple[Argument, ...]
    ) -> tuple[bool, ...]:
        """Determine whether node arguments depend on placeholders.

        For each argument, checks whether it corresponds to a node that depends
        on placeholders or constants. Returns a tuple of booleans indicating
        for each argument whether it should be treated as part of a Taylor
        expansion.

        Args:
            target: The function or operation associated with the current node.
            args: The node arguments to inspect.

        Returns:
            Tuple of booleans, where each entry indicates whether the
            corresponding argument depends on placeholders (``True``) or
            constants (``False``).

        Raises:
            RuntimeError: If dependency status cannot be determined for any argument.
            RuntimeError: If an argument dependent either on placeholders and only on
            constants or neither on placeholders nor only on constants
        """
        is_taylor = []
        for arg in args:
            if isinstance(arg, Proxy):
                in_placeholders = arg.node.name in self.dependent_on_placeholders
                in_constants = arg.node.name in self.dependent_on_constants
                if not (in_placeholders ^ in_constants):
                    raise RuntimeError(
                        f"Node {arg.node=} can not depend on placeholders and only on constants!"
                        if in_placeholders  # both are true
                        else f"Node {arg.node=} should either depend on placeholders or only on constants!"
                    )
                is_taylor.append(in_placeholders)

            elif isinstance(arg, tuple) and all(isinstance(a, Proxy) for a in arg):
                is_taylor.append(True)

            elif isinstance(arg, (int, float)) or arg is None:
                is_taylor.append(False)

            else:
                raise RuntimeError(
                    f"Could not detect dependency of {arg} for {target=}."
                )
        return tuple(is_taylor)

    def _constant_proxy(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Argument]
    ) -> Proxy:
        """Create a proxy node representing a constant operation.

        For operations whose arguments depend only on constants, this method
        creates a new proxy node in the graph. The node is recorded as
        constant-dependent, and its name is preserved to avoid duplication.

        Args:
            target: The function or operation being called.
            args: The node arguments.
            kwargs: The keyword arguments.

        Returns:
            The created `torch.fx.Proxy` node corresponding to a constant operation.
        """
        # Fetch information of the current node that we are going to replace.
        # This works because torch.fx uses fx_traceback.preserve_node_meta() in
        # .transform()
        from_nodes = get_current_meta().get("from_node", None)
        new_proxy = self.tracer.create_proxy(
            "call_function",
            target,
            args,
            kwargs,
            name=from_nodes[0].name if from_nodes else None,
        )
        self.dependent_on_constants.add(new_proxy.node.name)
        return new_proxy

    def _jet_proxy(
        self,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        is_taylor: tuple[bool, ...],
    ) -> Proxy:
        """Create a proxy node representing a jet (Taylor mode) operation.

        For operations whose arguments depend on placeholders, replaces the
        target operation with its corresponding jet operation from
        `jet.operations.MAPPING`. The created proxy node is marked as
        placeholder-dependent.

        Args:
            target: The function or operation being replaced.
            args: The node arguments.
            kwargs: The keyword arguments.
            is_taylor: Tuple of booleans indicating argument dependency on placeholders.

        Returns:
            The created `torch.fx.Proxy` node corresponding to a jet operation.

        Raises:
            NotImplementedError: If no jet operation is defined for the given target.
        """
        if target not in MAPPING.keys():
            raise NotImplementedError(f"Unsupported {target=}.")

        new_proxy = self.tracer.create_proxy(
            "call_function",
            MAPPING[target],
            args,
            {
                **kwargs,
                "_jet_info": JetInfo(
                    derivative_order=self.derivative_order, is_taylor=is_taylor
                ),
            },
        )
        self.dependent_on_placeholders.add(new_proxy.node.name)
        return new_proxy

    def call_function(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Argument]
    ) -> Proxy:
        """Override function call transformation logic.

        Determines whether the current node depends on placeholders or constants
        and creates the corresponding proxy node by dispatching to either
        `_constant_proxy` or `_jet_proxy`.

        Args:
            target: The function or callable to transform.
            args: Positional arguments of the node.
            kwargs: Keyword arguments of the node.

        Returns:
            The transformed `torch.fx.Proxy` node.
        """
        is_taylor = self._is_taylor(target, args)
        return (
            self._constant_proxy(target, args, kwargs)
            if not any(is_taylor)
            else self._jet_proxy(target, args, kwargs, is_taylor)
        )

    def call_module(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Argument]
    ) -> Proxy:
        """Handle module call nodes.

        This implementation currently disallows module calls.
        Consider extending :func:`JetTracer.is_leaf_module`.

        Args:
            target: Name of the module to be called.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Raises:
            NotImplementedError: Always, as module calls are not supported.
        """
        raise NotImplementedError(
            f"Unsupported module: {target=}. Consider adding it to the"
            " `JetTracer.is_leaf_module` function."
        )
