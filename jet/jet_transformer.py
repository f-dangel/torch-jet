"""Graph transformation for Taylor mode automatic differentiation (jets).

This module defines the `JetTransformer`, which extends
`torch.fx.Transformer` to replace operations in a traced PyTorch
computation graph with their Taylor mode (jet) equivalents. The transformer
uses dependency information to distinguish nodes that depend on placeholder
inputs from those that depend only on constants and substitutes the
corresponding jet operations from `jet.operations.MAPPING`.
"""

from torch import device, dtype, memory_format
from torch.fx import GraphModule, Proxy, Transformer
from torch.fx.node import Argument, Target
from torch.fx.traceback import get_current_meta

from jet.operations import MAPPING, IsTaylorType, JetInfo


class JetTransformer(Transformer):
    """Transformer that replaces nodes with their Taylor mode (jet) equivalents.

    The `JetTransformer` inserts additional placeholder nodes corresponding to
    higher-order Taylor (jet) coefficients for each original placeholder node.
    These additional placeholders represent the higher-order derivatives of the
    independent variable in a Taylor mode automatic differentiation context.

    The transformed graph will contain one new placeholder for each Taylor
    coefficient, and a new node combining all of them into a tuple. This tuple
    acts as the unified input representation for the variable and all its
    associated jet coefficients.

    Additionally the `JetTransformer` inspects each node in a traced
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
        # used as name for the combined node
        self.placeholder_and_coefficient_name = "jet_placeholder_coefficients"

    def get_attr(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Argument]
    ) -> Proxy:
        """Handle get_attr nodes and register them as constant-dependent.

        Args:
            target: The attribute name.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            The created proxy node.
        """
        proxy = super().get_attr(target, args, kwargs)
        self.dependent_on_constants.add(proxy.node.name)
        return proxy

    def _check_dependency(self, arg: Argument) -> bool:
        """Determine if an argument depends on placeholders.

        Args:
            arg: The argument to check.

        Returns:
            True if the argument depends on placeholders (Taylor variable),
            False if it depends only on constants.

        Raises:
            RuntimeError: if the argument’s dependency cannot be determined or is contradictory.
        """
        if isinstance(arg, Proxy):
            in_placeholders = arg.node.name in self.dependent_on_placeholders
            in_constants = arg.node.name in self.dependent_on_constants
            if not in_placeholders ^ in_constants:
                raise RuntimeError(
                    f"Node {arg.node=} can not depend on placeholders and only on constants!"
                    if in_placeholders  # both are true
                    else f"Node {arg.node=} should either depend on placeholders or only on constants!"
                )
            return in_placeholders

        elif isinstance(arg, tuple) and all(isinstance(a, Proxy) for a in arg):
            return True

        elif isinstance(arg, (int, float, bool, str, list)) or arg is None:
            return False

        elif isinstance(arg, (dtype, device, memory_format)):
            return False

        else:
            raise RuntimeError(f"Could not detect dependency of {arg}.")

    def _constant_proxy(self, target: Target, args: tuple[Argument, ...]) -> Proxy:
        """Create a proxy node representing a constant operation.

        For operations whose arguments depend only on constants, this method
        creates a new proxy node in the graph. The node is recorded as
        constant-dependent, and its name is preserved to avoid duplication.

        Args:
            target: The function or operation being called.
            args: The node arguments.

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
            {},
            name=from_nodes[0].name if from_nodes else None,
        )
        self.dependent_on_constants.add(new_proxy.node.name)
        return new_proxy

    def _jet_proxy(
        self,
        target: Target,
        args: tuple[Argument, ...],
        is_taylor: IsTaylorType,
    ) -> Proxy:
        """Create a proxy node representing a jet (Taylor mode) operation.

        For operations whose arguments depend on placeholders, replaces the
        target operation with its corresponding jet operation from
        `jet.operations.MAPPING`. The created proxy node is marked as
        placeholder-dependent.

        Args:
            target: The function or operation being replaced.
            args: The node arguments.
            is_taylor: Indicating which positional args depend on placeholders.

        Returns:
            The created `torch.fx.Proxy` node corresponding to a jet operation.

        Raises:
            NotImplementedError: If no jet operation is defined for the given target.
        """
        if target not in MAPPING:
            raise NotImplementedError(f"Unsupported {target=}.")

        new_proxy = self.tracer.create_proxy(
            "call_function",
            MAPPING[target],
            args,
            {
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
        # make_fx always places all arguments into node.args (node.kwargs is
        # always empty), so we only need to track positional arg dependencies.
        is_taylor = tuple(self._check_dependency(arg) for arg in args)

        return (
            self._jet_proxy(target, args, is_taylor)
            if any(is_taylor)
            else self._constant_proxy(target, args)
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

    def call_method(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Argument]
    ) -> Proxy:
        """Handle method call nodes such as `x.sin()`.

        This implementation currently disallows method calls.

        Args:
            target: Name of the method to be called.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Raises:
            NotImplementedError: Always, as method calls are not supported.
        """
        raise NotImplementedError(
            f"Encountered unsupported call_method node ({target})"
        )

    def placeholder(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Argument]
    ):
        """Replace a single placeholder node with a tuple of placeholder coefficients.

        This method intercepts the creation of placeholder nodes during graph
        transformation. For each original placeholder, it:
          1. Creates the base placeholder (e.g., ``x``).
          2. Creates additional placeholder nodes ``v1, v2, ..., vN`` for each
             derivative coefficient up to ``derivative_order``.
          3. Combines them into a tuple node ``(x, v1, ..., vN)``.
          4. Returns this tuple node as the replacement for the original placeholder.

        The tuple node will later serve as the single argument representing
        the independent variable and its jet coefficients throughout the transformed graph.

        Args:
            target: The name of the placeholder being transformed (e.g., ``"x"``).
            args: Positional arguments (unused for placeholders).
            kwargs: Keyword arguments (unused for placeholders).

        Returns:
            A proxy representing a tuple of the base variable and
            its coefficient placeholders.
        """
        # Create the base placeholder (the independent variable)
        base_proxy = super().placeholder(target, args, kwargs)

        # Create additional placeholders representing higher-order coefficients
        coeffs = [
            self.tracer.create_proxy(
                "placeholder", f"{target}_v{i}", (), {}, name=f"{target}_v{i}"
            )
            for i in range(1, self.derivative_order + 1)
        ]

        # Combine the base and coefficient placeholders into a single tuple node
        # Note: tuple() accepts a single iterable argument, hence the double parentheses.
        tuple_proxy = self.tracer.create_proxy(
            "call_function",
            tuple,
            ((base_proxy, *coeffs),),  # Pass as one iterable argument
            {},
            name=f"{target}_{self.placeholder_and_coefficient_name}",
        )
        self.dependent_on_placeholders.add(tuple_proxy.node.name)
        return tuple_proxy
