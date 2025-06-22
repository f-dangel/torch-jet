"""Implements individual simplification rules."""

import operator
from abc import ABC, abstractmethod
from typing import Any, Callable
from warnings import warn

from torch import Tensor, add, cos, cosh, div, mul
from torch import pow as torch_pow
from torch import sigmoid, sin, sub, tanh
from torch.fx import Graph, GraphModule, Node
from torch.nn.functional import linear

import jet.utils


def is_replicate(arg: Any) -> bool:
    """Check if the argument is a `replicate` node.

    Args:
        arg: Input to the function.arg` tuple.

    Returns:
        Whether the argument is a `replicate` node.
    """
    return (
        isinstance(arg, Node)
        and arg.op == "call_function"
        and arg.target == jet.utils.replicate
    )


def is_sum_vmapped(arg: Any) -> bool:
    """Check if an argument of a node is a `sum_vmapped` node.

    Args:
        arg: An entry from a `Node.arg` tuple.

    Returns:
        Whether the argument is a `sum_vmapped` node.
    """
    return (
        isinstance(arg, Node)
        and arg.op == "call_function"
        and arg.target == jet.utils.sum_vmapped
    )


class Rule(ABC):
    """Base class for graph-based simplification rules."""

    @abstractmethod
    def match(self, node: Node) -> bool:
        """Detect a match with a simplification's entry point.

        Args:
            node: A node in a computation graph.
        """
        pass

    @abstractmethod
    def apply(self, node: Node, graph: Graph) -> None:
        """Apply the simplification rule.

        Args:
            node: A node in a computation graph that represents the rule's entry point.
            graph: The computation graph to which the rule is applied.
        """
        pass


class ModuleRule(ABC):
    """Base class for simplification rules that act on a GraphModule.

    Such rules can access and modify tensor constants of the module.
    """

    @abstractmethod
    def match(self, node: Node) -> bool:
        """Detect a match with a simplification's entry point.

        Args:
            node: A node in a computation graph.
        """
        pass

    @abstractmethod
    def apply(self, node: Node, module: GraphModule) -> None:
        """Apply the simplification rule.

        Args:
            node: A node in a computation graph that represents the rule's entry point.
            module: A GraphModule representing the computation graph.
        """
        pass


class PushReplicateElementwise(Rule):
    """Rule for simplifying `replicate(f(x))` into `f(replicate(x))`.

    `f` is an elementwise function, such as `sin`, `cos`, or `tanh`, `sigmoid`.

    Attributes:
        OPERATIONS: List of elementwise operations that can be simplified.
    """

    OPERATIONS: list[Callable[[Tensor], Tensor]] = [cos, sin, tanh, sigmoid, cosh]

    def match(self, node: Node) -> bool:
        """Detect a match with the simplification's entry point.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern `replicate(f(x))`, False otherwise.
        """
        return (
            node.op == "call_function"
            and node.users  # must be used by other nodes
            and len(node.args) == 1
            and node.kwargs == {}
            and node.target in self.OPERATIONS  # must be elementwise
            and len(node.all_input_nodes) == 1  # must consume a single input tensor...
            and is_replicate(node.all_input_nodes[0])  # ... which is a replicate tensor
        )

    def apply(self, f_node: Node, graph: Graph) -> None:
        """Apply the simplification rule to the node, modifying the graph.

        Args:
            f_node: A elementwise function node in the graph that consumes a `replicate`
                node.
            graph: The computation graph to which the rule is applied.
        """
        # find the `replicate` node and its input tensor
        (rep_node,) = f_node.all_input_nodes
        (x,) = rep_node.all_input_nodes

        # swap the order of the `replicate` and the elementwise function `f`
        with graph.inserting_after(rep_node):
            new_f_node = graph.call_function(f_node.target, args=(x,))

        with graph.inserting_after(new_f_node):
            new_rep_node = graph.call_function(
                jet.utils.replicate,
                args=(new_f_node, *rep_node.args[1:]),
                kwargs=rep_node.kwargs,
            )

        # replace the old node with its simplified node in the entire graph
        f_node.replace_all_uses_with(new_rep_node)


class PushReplicateScalarArithmetic(Rule):
    """Rule for simplifying `replicate(x ∘ y)` with ∘ an arithmetic op (+, -, *, /, **).

    We assume that one of `x, y` is a float or integer.

    The following two cases simplify to the same result:

    1. `x` scalar, `y` tensor: `replicate(x ∘ y) -> x ∘ replicate(y)`.
    2. `x` tensor, `y` scalar: `replicate(x ∘ y) -> replicate(x) ∘ y`.

    Attributes:
        OPERATIONS: List of arithmetic operations that can be simplified.
            Includes addition, subtraction, multiplication, division & exponentiation.
    """

    OPERATIONS: list[Callable[[Tensor | float | int, Tensor | float | int], Tensor]] = [
        # addition
        add,
        operator.add,
        # subtraction
        sub,
        operator.sub,
        # multiplication
        mul,
        operator.mul,
        # division
        div,
        operator.truediv,
        # exponentiation
        torch_pow,
        operator.pow,
    ]

    def match(self, node: Node) -> bool:
        """Match for arithmetic operations with of scalar and a replicate tensor.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern `replicate(x ∘ y)`, where ∘ is an
            arithmetic operation and either `x` or `y` is a scalar, False otherwise.
        """
        return (
            node.op == "call_function"
            and node.users
            and node.target in self.OPERATIONS
            and len(node.args) == 2
            and node.kwargs == {}
            and sum(is_replicate(a) for a in node.args) == 1
            and sum(isinstance(a, (float, int)) for a in node.args) == 1
        )

    def apply(self, arith_node: Node, graph: Graph) -> None:
        """Apply the simplification rule to the node, modifying the graph.

        Args:
            arith_node: An arithmetic operation node in the graph that consumes a
                `replicate` node and a scalar. Must satisfy the match condition.
            graph: The computation graph to which the rule is applied.
        """
        # find the `replicate` node and its input tensor
        (rep_node,) = arith_node.all_input_nodes
        rep_pos = arith_node.args.index(rep_node)
        (x,) = rep_node.all_input_nodes

        # swap the order of the `replicate` and the arithmetic operation
        with graph.inserting_after(rep_node):
            new_args = tuple(
                x if idx == rep_pos else arg for idx, arg in enumerate(arith_node.args)
            )
            new_arith_node = graph.call_function(arith_node.target, args=new_args)

        with graph.inserting_after(new_arith_node):
            new_rep_node = graph.call_function(
                jet.utils.replicate,
                args=(new_arith_node, *rep_node.args[1:]),
                kwargs=rep_node.kwargs,
            )

        # replace the old node with its simplified node in the entire graph
        arith_node.replace_all_uses_with(new_rep_node)


class PushReplicateTensorArithmetic(Rule):
    """Rule to simplify `f(replicate(x1), replicate(x2))` into `replicate(f(x1, x2))`.

    This rule applies when both `replicate` nodes have the same `times` and `pos`
    values.

    Attributes:
        OPERATIONS: List of arithmetic operations that can be simplified.
            Includes addition, subtraction, multiplication, division & exponentiation.
    """

    OPERATIONS: list[Callable[[Tensor, Tensor], Tensor]] = [
        # addition
        add,
        operator.add,
        # subtraction
        sub,
        operator.sub,
        # multiplication
        mul,
        operator.mul,
        # division
        div,
        operator.truediv,
        # exponentiation
        torch_pow,
        operator.pow,
    ]

    def match(self, node: Node) -> bool:
        """Match for arithmetic operations that consume two replicate nodes.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern `f(replicate(x1), replicate(x2))` with
            identical `times` and `pos` values, False otherwise.
        """
        return (
            node.op == "call_function"
            and node.users
            and node.target in self.OPERATIONS
            and len(node.args) == 2
            and node.kwargs == {}
            and all(is_replicate(arg) for arg in node.args)
            # same `times` argument
            and len({arg.args[1] for arg in node.args}) == 1
            # same `pos` argument
            and len({arg.kwargs["pos"] for arg in node.args}) == 1
        )

    def apply(self, arith_node: Node, graph: Graph) -> None:
        """Apply the simplification rule.

        Args:
            arith_node: A node in a computation graph that represents the arithmetic
                operation that consumes two replicate tensors.
            graph: The computation graph to which the rule is applied.
        """
        # find the tensors that are being replicated
        mapping = {}
        for rep in arith_node.all_input_nodes:
            (x,) = rep.all_input_nodes
            mapping[rep] = x

        # determine the times and pos arguments
        (times,) = {rep.args[1] for rep in arith_node.all_input_nodes}
        (pos,) = {rep.kwargs["pos"] for rep in arith_node.all_input_nodes}

        # swap the order of the `replicate` and the arithmetic operation
        with graph.inserting_before(arith_node):
            new_args = tuple(mapping[rep] for rep in arith_node.args)
            new_arith_node = graph.call_function(arith_node.target, args=new_args)

        with graph.inserting_after(new_arith_node):
            new_rep_node = graph.call_function(
                jet.utils.replicate, args=(new_arith_node, times), kwargs={"pos": pos}
            )

        # replace the old node with its simplified node in the entire graph
        arith_node.replace_all_uses_with(new_rep_node)


class PushReplicateLinear(Rule):
    """Rule to simplify `linear(replicate(x), W, b)` to `replicate(linear(x, W, b))`."""

    def match(self, node: Node) -> bool:
        """Detect a linear operation that consumes a replicated input.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern `torch.nn.linear(replicate(x), W, b)`,
            False otherwise.
        """
        return (
            node.op == "call_function"
            and node.users
            and node.target == linear
            and is_replicate(node.args[0])  # x must be a replicate node
        )

    def apply(self, linear_node: Node, graph: Graph) -> None:
        """Apply the simplification rule.

        Warning:
            This simplification rule will fail if the replication happens along the
            last axis. The current implementation has no means to figure out if the
            replicated axis represents the last; it always assumes it is not.

        Args:
            linear_node: A node in a computation graph that represents the linear
                operation consuming a replicate node.
            graph: The computation graph to which the rule is applied.
        """
        # find the tensors that are being replicated
        rep_node = linear_node.args[0]
        (x,) = rep_node.all_input_nodes
        times, pos = rep_node.args[1], rep_node.kwargs["pos"]

        if pos > 0:
            warn(
                "The `PushReplicateLinear` rule assumes that the replicated axis is"
                f" not the last axis. If it is, the rule will fail. Got {pos=}.",
            )

        # Create a new linear node
        with graph.inserting_after(linear_node):
            new_linear_node = graph.call_function(
                linear, args=(x, *linear_node.args[1:]), kwargs=linear_node.kwargs
            )

        # Create a new replicate node
        with graph.inserting_after(new_linear_node):
            new_rep_node = graph.call_function(
                jet.utils.replicate, args=(new_linear_node, times), kwargs={"pos": pos}
            )

        # Replace the old node with its simplified node in the entire graph
        linear_node.replace_all_uses_with(new_rep_node)


class PushReplicateSumVmapped(Rule):
    """Rule for simplifying `sum_vmapped(replicate(x, times, pos=pos1), pos=pos2)`.

    Consider `sum_vmapped(replicate(x, times, pos1), pos2)`.
    There are three different scenarios how to simplify this:

    1. `pos1 == pos2`: `times * x`
    2. `pos1 > pos2`: `replicate(sum_vmapped(x, pos2), times, pos1 - 1)`
    3. `pos1 < pos2`: `replicate(sum_vmapped(x, pos2 - 1), times, pos1)`
    """

    def match(self, node: Node) -> bool:
        """Match for a `sum_vmapped` node that consumes a `replicate` node.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern
            `sum_vmapped(replicate(x, times, pos=pos1), pos=pos2)`, False otherwise.
        """
        return (
            node.op == "call_function"
            and node.users
            and node.target == jet.utils.sum_vmapped
            and len(node.args) == 1
            and list(node.kwargs.keys()) == ["pos"]
            and is_replicate(node.args[0])
        )

    def apply(self, sum_node: Node, graph: Graph) -> None:
        """Apply the simplification rule.

        Args:
            sum_node: The `sum_vmapped` node that consumes a `replicate` node.
            graph: The computation graph to which the rule is applied.
        """
        (rep_node,) = sum_node.all_input_nodes
        (x,) = rep_node.all_input_nodes
        pos_rep = rep_node.kwargs["pos"]
        pos_sum = sum_node.kwargs["pos"]
        times = rep_node.args[1]

        if pos_sum == pos_rep:
            # Insert a multiplication node before the replicate node
            with graph.inserting_before(rep_node):
                mul_node = graph.call_function(operator.mul, args=(x, times))
            sum_node.replace_all_uses_with(mul_node)

        else:
            # Insert a new sum node before the sum node
            with graph.inserting_before(sum_node):
                new_sum_node = graph.call_function(
                    jet.utils.sum_vmapped,
                    args=(x,),
                    kwargs={"pos": pos_sum if pos_rep > pos_sum else pos_sum - 1},
                )
            # Insert a new replicate node after the new sum node
            with graph.inserting_after(new_sum_node):
                new_rep_node = graph.call_function(
                    jet.utils.replicate,
                    args=(new_sum_node, times),
                    kwargs={"pos": pos_rep - 1 if pos_rep > pos_sum else pos_rep},
                )

            # Replace the old node with its simplified node in the entire graph
            sum_node.replace_all_uses_with(new_rep_node)


class PullSumVmappedScalarMultiplication(Rule):
    """Rule for simplifying `sum_vmapped(x * y)` with one scalar argument.

    The following two cases simplify to the same result:

    1. `x` scalar: `sum_vmapped(x * y)` -> `x * sum_vmapped(y)`.
    2. `y` scalar: `sum_vmapped(x * y)` -> `replicate(x) * y`.

    Attributes:
        OPERATIONS: List of operations that can be simplified.
    """

    OPERATIONS: list[Callable[[Tensor | float | int, Tensor | float | int], Tensor]] = [
        mul,
        operator.mul,
    ]

    def match(self, node: Node) -> bool:
        """Match for sum_vmapped nodes that consume multiplications with a scalar.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern `sum_vmapped(x * y)`, where * is
            multiplication and either `x` or `y` a scalar, False otherwise.
        """
        if not is_sum_vmapped(node) or not node.users:
            return False

        (in_node,) = node.all_input_nodes
        return (
            in_node.op == "call_function"
            and in_node.target in self.OPERATIONS
            and len(in_node.args) == 2
            and in_node.kwargs == {}
            and sum(isinstance(a, (float, int)) for a in in_node.args) == 1
        )

    def apply(self, sum_node: Node, graph: Graph) -> None:
        """Apply the simplification rule to the node, modifying the graph.

        Args:
            sum_node: A `sum_vmapped` node that consumes a node representing multipli-
                cation operation with a scalar/float. Must satisfy the match condition.
            graph: The computation graph to which the rule is applied.
        """
        # find the multiplication node and its input tensor
        (mul_node,) = sum_node.all_input_nodes
        (x,) = mul_node.all_input_nodes
        x_pos = mul_node.args.index(x)

        # swap the order of the `sum_vmapped` and the arithmetic operation
        with graph.inserting_after(sum_node):
            new_sum_node = graph.call_function(
                jet.utils.sum_vmapped, args=(x,), kwargs=sum_node.kwargs
            )
        # Insert a new multiplication node after the new sum node
        with graph.inserting_after(new_sum_node):
            new_args = tuple(
                new_sum_node if idx == x_pos else arg
                for idx, arg in enumerate(mul_node.args)
            )
            new_mul_node = graph.call_function(mul_node.target, args=new_args)

        # replace the old node with its simplified node in the entire graph
        sum_node.replace_all_uses_with(new_mul_node)


class PullSumVmappedTensorAddition(Rule):
    """Rule for simplifying `sum_vmapped(x + y)` where x and y are tensors.

    The simplified result is `sum_vmapped(x) + sum_vmapped(y)`.
    Same for subtraction.

    Warning:
        This rule assumes no broadcasting, i.e. `x` and `y` must have the same shape.

    The following two cases simplify to the same result (for * and /):

    Attributes:
        OPERATIONS: List of operations that can be simplified.
            Includes addition and subtraction.
    """

    OPERATIONS: list[Callable[[Tensor | float | int, Tensor | float | int], Tensor]] = [
        # addition
        add,
        operator.add,
        # subtraction
        sub,
        operator.sub,
    ]

    def match(self, node: Node) -> bool:
        """Match for sum_vmapped nodes that consumes a summation/subtraction node.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern `sum_vmapped(x + y)` (or -), where
            `x` and `y` are tensors, False otherwise.
        """
        if not is_sum_vmapped(node) or not node.users:
            return False

        (in_node,) = node.all_input_nodes
        return (
            in_node.op == "call_function"
            and in_node.target in self.OPERATIONS
            and len(in_node.args) == 2
            and in_node.kwargs == {}
            and sum(isinstance(a, Node) for a in in_node.args) == 2
        )

    def apply(self, sum_node: Node, graph: Graph) -> None:
        """Apply the simplification rule to the node, modifying the graph.

        Args:
            sum_node: A `sum_vmapped` node that consumes a node representing addition/
                subtraction of two tensors. Must satisfy the match condition.
            graph: The computation graph to which the rule is applied.
        """
        # find the addition/subtraction node and its input tensor
        (add_node,) = sum_node.all_input_nodes

        mapping = {}
        # swap the order of the `sum_vmapped` and the addition/subtraction operation
        for x in add_node.all_input_nodes:
            with graph.inserting_after(x):
                new_sum_node = graph.call_function(
                    jet.utils.sum_vmapped, args=(x,), kwargs=sum_node.kwargs
                )
            mapping[x] = new_sum_node

        # Insert a new addition/subtraction node after the new sum nodes
        with graph.inserting_after(add_node):
            new_args = tuple(mapping[x] for x in add_node.args)
            new_add_node = graph.call_function(add_node.target, args=new_args)

        # replace the old node with its simplified node in the entire graph
        sum_node.replace_all_uses_with(new_add_node)


class PullSumVmappedLinear(Rule):
    """Simplify `sum_vmapped(linear(x, W, 0))` into `linear(sum_vmapped(x), W, 0)`."""

    def match(self, node: Node) -> bool:
        """Match for sum_vmapped nodes that consume a linear operation.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern `sum_vmapped(linear(x, W, b))`, False
            otherwise.
        """
        if not is_sum_vmapped(node) or not node.users:
            return False

        (in_node,) = node.all_input_nodes
        is_linear = in_node.op == "call_function" and in_node.target == linear

        if not is_linear:
            return False

        # check that the linear node has no bias (b = 0)
        if len(in_node.args) < 3:
            return in_node.kwargs.get("bias", None) is None

        return in_node.args[2] is None

    def apply(self, sum_node: Node, graph: Graph) -> None:
        """Apply the simplification rule to the node, modifying the graph.

        Args:
            sum_node: A `sum_vmapped` node that consumes a `linear` node. Must satisfy
                the match condition.
            graph: The computation graph to which the rule is applied.
        """
        (linear_node,) = sum_node.all_input_nodes
        x = linear_node.args[0]
        pos = sum_node.kwargs["pos"]

        warn(
            "The `PullSumVmappedLinear` rule assumes that the summed axis is not "
            f"the last axis. If it is, the rule will fail. Got {pos=}.",
        )

        # swap the order of the `sum_vmapped` and the linear operation
        with graph.inserting_after(x):
            new_sum_node = graph.call_function(
                jet.utils.sum_vmapped, args=(x,), kwargs=sum_node.kwargs
            )
        with graph.inserting_after(linear_node):
            new_linear_node = graph.call_function(
                linear,
                args=(new_sum_node, *linear_node.args[1:]),
                kwargs=linear_node.kwargs,
            )

        # replace the old node with its simplified node in the entire graph
        sum_node.replace_all_uses_with(new_linear_node)


class PullSumVmappedReplicateMultiplication(Rule):
    """Simplify `sum_vmapped(y * replicate(x, times, pos=pos1), pos=pos2)`.

    This rule applies when `pos1 == pos2` and simplifies the expression into
    `sum_vmapped(y, pos=pos2) * x`.
    It also assumes that both tensors that are being multiplied have the same shape.

    Attributes:
        OPERATIONS: List of multiplication operations that can be simplified.
    """

    OPERATIONS = [operator.mul, mul]

    def match(self, node: Node) -> bool:
        """Detect a match with sum_vmapped(y * replicate(x, times, pos=pos), pos=pos).

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern
            `sum_vmapped(y * replicate(x, times, pos=pos1), pos=pos2)` with
            `pos1 == pos2`, False otherwise.
        """
        if not is_sum_vmapped(node) or not node.users:
            return False

        (in_node,) = node.all_input_nodes
        if in_node.op != "call_function" or in_node.target not in self.OPERATIONS:
            return False

        if (
            in_node.kwargs == {}
            and sum(is_replicate(arg) for arg in in_node.args) == 1
            and sum(isinstance(arg, Node) for arg in in_node.args) == 2
        ):
            (rep_node,) = [arg for arg in in_node.args if is_replicate(arg)]
            sum_pos = node.kwargs["pos"]
            rep_pos = rep_node.kwargs["pos"]
            return sum_pos == rep_pos

        return False

    def apply(self, sum_node: Node, graph: Graph) -> None:
        """Apply the simplification rule.

        Args:
            sum_node: The `sum_vmapped` node that consumes a multiplication node.
            graph: The computation graph to which the rule is applied.
        """
        (mul_node,) = sum_node.all_input_nodes
        (rep_node,) = [n for n in mul_node.all_input_nodes if is_replicate(n)]
        (x_node,) = rep_node.all_input_nodes
        (other_node,) = [n for n in mul_node.all_input_nodes if not is_replicate(n)]

        (pos,) = {rep_node.kwargs["pos"], sum_node.kwargs["pos"]}

        # Create a new sum_vmapped node for
        with graph.inserting_after(other_node):
            new_sum_node = graph.call_function(
                jet.utils.sum_vmapped, args=(other_node,), kwargs={"pos": pos}
            )

        # Create a new multiplication node for `sum_vmapped(y) * x`
        with graph.inserting_after(new_sum_node):
            mapping = {rep_node: x_node, other_node: new_sum_node}
            new_mul_node = graph.call_function(
                mul_node.target, args=tuple(mapping[arg] for arg in mul_node.args)
            )

        # Replace the old node with the simplified node
        sum_node.replace_all_uses_with(new_mul_node)


class MergeSumVmappedConstant(ModuleRule):
    """Simplify `sum_vmapped(constant_tensor, pos=pos)` by precomputing the sum.

    This rule applies when the input to `sum_vmapped` is a constant tensor.
    """

    def match(self, node: Node) -> bool:
        """Detect a match with the simplification's entry point.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern
            `sum_vmapped(constant_tensor, pos=pos)`, False otherwise.
        """
        return (
            is_sum_vmapped(node)
            and node.users
            and node.all_input_nodes[0].op == "get_attr"
        )

    def apply(self, sum_node: Node, mod: GraphModule) -> None:
        """Apply the simplification rule.

        Args:
            sum_node: The `sum_vmapped` node that consumes a constant tensor.
            mod: A GraphModule representing the computation graph.
        """
        (const_node,) = sum_node.all_input_nodes
        sum_pos = sum_node.kwargs["pos"]

        const = jet.utils.recursive_getattr(mod, const_node.target)
        prefix = ".".join(const_node.target.split(".")[:-1])
        name = const_node.target.split(".")[-1]

        new_const = const.sum(dim=sum_pos)
        new_name = f"{name}sum{sum_pos}"
        jet.utils.recursive_setattr(mod, f"{prefix}.{new_name}", new_const)

        # add a new node
        with mod.graph.inserting_after(const_node):
            new_const_node = mod.graph.create_node("get_attr", f"{prefix}.{new_name}")
        # replace the old node with the new constant node
        sum_node.replace_all_uses_with(new_const_node)

        # if the sum node is not used, remove it from the graph
        if not sum_node.users:
            mod.graph.erase_node(sum_node)
        # if the constant node is not used anymore, remove the tensor from the module
        if not const_node.users:
            mod.graph.erase_node(const_node)
            jet.utils.recursive_delattr(mod, f"{prefix}.{name}")
