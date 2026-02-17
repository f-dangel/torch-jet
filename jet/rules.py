"""Implements individual simplification rules."""

from abc import ABC, abstractmethod
from typing import Any, Callable
from warnings import warn

import torch
from torch import Tensor
from torch.fx import Graph, Node
from torch.nn.functional import linear

# ATen and custom op references used by rules
_aten = torch.ops.aten
_jet_ops = torch.ops.jet

# Op targets for pattern matching
_replicate_target = _jet_ops.replicate.default
_sum_vmapped_target = _jet_ops.sum_vmapped.default


def is_replicate(arg: Any) -> bool:
    """Check if the argument is a ``replicate`` node.

    Args:
        arg: Input to the function.

    Returns:
        Whether the argument is a ``replicate`` node.
    """
    return (
        isinstance(arg, Node)
        and arg.op == "call_function"
        and arg.target == _replicate_target
    )


def is_sum_vmapped(arg: Any) -> bool:
    """Check if an argument is a ``sum_vmapped`` node.

    Args:
        arg: An entry from a ``Node.arg`` tuple.

    Returns:
        Whether the argument is a sum_vmapped node.
    """
    return (
        isinstance(arg, Node)
        and arg.op == "call_function"
        and arg.target == _sum_vmapped_target
    )


def _get_replicate_pos(node: Node) -> int:
    """Extract ``pos`` from a replicate node (make_fx positional format).

    With make_fx, ``replicate(x, times)`` has ``args=(x, times)`` when pos=0
    (default omitted), and ``replicate(x, times, pos)`` has ``args=(x, times, pos)``
    when pos != 0.

    Args:
        node: A replicate node.

    Returns:
        The position argument.
    """
    return node.args[2] if len(node.args) > 2 else 0


def _get_replicate_times(node: Node) -> int:
    """Extract ``times`` from a replicate node.

    Args:
        node: A replicate node.

    Returns:
        The times argument.
    """
    return node.args[1]


def _get_sum_vmapped_pos(node: Node) -> int:
    """Extract ``pos`` from a ``sum_vmapped`` node.

    With make_fx, ``sum_vmapped(x)`` has ``args=(x,)`` when pos=0 (default),
    and ``sum_vmapped(x, pos)`` has ``args=(x, pos)`` when pos != 0.

    Args:
        node: A sum_vmapped node.

    Returns:
        The position along which the sum is performed.
    """
    return node.args[1] if len(node.args) > 1 else 0


def _make_sum_vmapped_args(x: Any, pos: int) -> tuple[tuple, dict]:
    """Create args and kwargs for a new ``sum_vmapped`` node.

    Args:
        x: The input tensor node.
        pos: The position to sum along.

    Returns:
        Tuple of (args, kwargs) for graph.call_function.
    """
    if pos != 0:
        return (x, pos), {}
    return (x,), {}


def _make_replicate_args(x: Any, times: int, pos: int) -> tuple[tuple, dict]:
    """Create args and kwargs for a new ``replicate`` node.

    Args:
        x: The input tensor node.
        times: Number of replications.
        pos: Position of the new axis.

    Returns:
        Tuple of (args, kwargs) for graph.call_function.
    """
    if pos == 0:
        return (x, times), {}
    return (x, times, pos), {}


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


class PushReplicateElementwise(Rule):
    """Rule for simplifying ``replicate(f(x))`` into ``f(replicate(x))``.

    ``f`` is an elementwise function, such as ``sin``, ``cos``, ``tanh``, ``sigmoid``.

    Attributes:
        OPERATIONS: List of elementwise operations that can be simplified.
    """

    OPERATIONS: list[Callable[[Tensor], Tensor]] = [
        _aten.cos.default,
        _aten.sin.default,
        _aten.tanh.default,
        _aten.sigmoid.default,
        _aten.cosh.default,
    ]

    def match(self, node: Node) -> bool:
        """Detect a match with the simplification's entry point.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern ``replicate(f(x))``, False otherwise.
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
            f_node: A elementwise function node in the graph that consumes a ``replicate``
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
            args, kwargs = _make_replicate_args(
                new_f_node,
                _get_replicate_times(rep_node),
                _get_replicate_pos(rep_node),
            )
            new_rep_node = graph.call_function(
                _replicate_target, args=args, kwargs=kwargs
            )

        # replace the old node with its simplified node in the entire graph
        f_node.replace_all_uses_with(new_rep_node)


class PushReplicateScalarArithmetic(Rule):
    """Rule for simplifying ``replicate(x ∘ y)`` with ∘ an arithmetic op.

    We assume that one of ``x, y`` is a float or integer.

    The following two cases simplify to the same result:

    1. ``x`` scalar, ``y`` tensor: ``replicate(x ∘ y) -> x ∘ replicate(y)``.
    2. ``x`` tensor, ``y`` scalar: ``replicate(x ∘ y) -> replicate(x) ∘ y``.

    Attributes:
        OPERATIONS: List of arithmetic operations that can be simplified.
            Includes addition, subtraction, multiplication, division & exponentiation.
    """

    OPERATIONS: list[Callable[[Tensor | float | int, Tensor | float | int], Tensor]] = [
        _aten.add.Tensor,
        _aten.sub.Tensor,
        _aten.rsub.Scalar,
        _aten.mul.Tensor,
        _aten.div.Tensor,
        _aten.pow.Tensor_Scalar,
        _aten.pow.Scalar,
    ]

    def match(self, node: Node) -> bool:
        """Match for arithmetic operations with of scalar and a replicate tensor.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern ``replicate(x ∘ y)``, where ∘ is an
            arithmetic operation and either ``x`` or ``y`` is a scalar, False otherwise.
        """
        return (
            node.op == "call_function"
            and node.users
            and node.target in self.OPERATIONS
            and len(node.args) == 2
            and sum(is_replicate(a) for a in node.args) == 1
            and sum(isinstance(a, (float, int)) for a in node.args) == 1
        )

    def apply(self, arith_node: Node, graph: Graph) -> None:
        """Apply the simplification rule to the node, modifying the graph.

        Args:
            arith_node: An arithmetic operation node in the graph that consumes a
                ``replicate`` node and a scalar. Must satisfy the match condition.
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
            args, kwargs = _make_replicate_args(
                new_arith_node,
                _get_replicate_times(rep_node),
                _get_replicate_pos(rep_node),
            )
            new_rep_node = graph.call_function(
                _replicate_target, args=args, kwargs=kwargs
            )

        # replace the old node with its simplified node in the entire graph
        arith_node.replace_all_uses_with(new_rep_node)


class PushReplicateTensorArithmetic(Rule):
    """Rule to simplify ``f(replicate(x1), replicate(x2))`` to ``replicate(f(x1, x2))``.

    This rule applies when both ``replicate`` nodes have the same ``times`` and ``pos``
    values.

    Attributes:
        OPERATIONS: List of arithmetic operations that can be simplified.
            Includes addition, subtraction, multiplication, division & exponentiation.
    """

    OPERATIONS: list[Callable[[Tensor, Tensor], Tensor]] = [
        _aten.add.Tensor,
        _aten.sub.Tensor,
        _aten.mul.Tensor,
        _aten.div.Tensor,
        _aten.pow.Tensor_Scalar,
        _aten.pow.Tensor_Tensor,
    ]

    def match(self, node: Node) -> bool:
        """Match for arithmetic operations that consume two replicate nodes.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern ``f(replicate(x1), replicate(x2))`` with
            identical ``times`` and ``pos`` values, False otherwise.
        """
        return (
            node.op == "call_function"
            and node.users
            and node.target in self.OPERATIONS
            and len(node.args) == 2
            and node.kwargs == {}
            and all(is_replicate(arg) for arg in node.args)
            # same `times` argument
            and len({_get_replicate_times(arg) for arg in node.args}) == 1
            # same `pos` argument
            and len({_get_replicate_pos(arg) for arg in node.args}) == 1
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
        (times,) = {_get_replicate_times(rep) for rep in arith_node.all_input_nodes}
        (pos,) = {_get_replicate_pos(rep) for rep in arith_node.all_input_nodes}

        # swap the order of the `replicate` and the arithmetic operation
        with graph.inserting_before(arith_node):
            new_args = tuple(mapping[rep] for rep in arith_node.args)
            new_arith_node = graph.call_function(arith_node.target, args=new_args)

        with graph.inserting_after(new_arith_node):
            rep_args, rep_kwargs = _make_replicate_args(new_arith_node, times, pos)
            new_rep_node = graph.call_function(
                _replicate_target, args=rep_args, kwargs=rep_kwargs
            )

        # replace the old node with its simplified node in the entire graph
        arith_node.replace_all_uses_with(new_rep_node)


class PushReplicateLinear(Rule):
    """Rule to simplify ``linear(replicate(x), W, b)`` to ``replicate(linear(x, W, b))``.

    Note:
        With ``make_fx`` tracing, ``linear`` decomposes into lower-level ops
        (``mm``, ``addmm``, ``t``, ``view``), so this rule will not match.
        It is kept for backward compatibility.
    """

    def match(self, node: Node) -> bool:
        """Detect a linear operation that consumes a replicated input.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern ``torch.nn.linear(replicate(x), W, b)``,
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
        times = _get_replicate_times(rep_node)
        pos = _get_replicate_pos(rep_node)

        if pos > 0:
            warn(
                "The `PushReplicateLinear` rule assumes that the replicated axis is"
                f" not the last axis. If it is, the rule will fail. Got {pos=}.",
                stacklevel=2,
            )

        # Create a new linear node
        with graph.inserting_after(linear_node):
            new_linear_node = graph.call_function(
                linear, args=(x, *linear_node.args[1:]), kwargs=linear_node.kwargs
            )

        # Create a new replicate node
        with graph.inserting_after(new_linear_node):
            rep_args, rep_kwargs = _make_replicate_args(new_linear_node, times, pos)
            new_rep_node = graph.call_function(
                _replicate_target, args=rep_args, kwargs=rep_kwargs
            )

        # Replace the old node with its simplified node in the entire graph
        linear_node.replace_all_uses_with(new_rep_node)


class PushReplicateSumVmapped(Rule):
    """Rule for simplifying ``sum_vmapped(replicate(x, times, pos1), pos2)``.

    Consider ``sum_vmapped(replicate(x, times, pos1), pos2)``.
    There are three different scenarios how to simplify this:

    1. ``pos1 == pos2``: ``times * x``
    2. ``pos1 > pos2``: ``replicate(sum_vmapped(x, pos2), times, pos1 - 1)``
    3. ``pos1 < pos2``: ``replicate(sum_vmapped(x, pos2 - 1), times, pos1)``
    """

    def match(self, node: Node) -> bool:
        """Match for a ``sum_vmapped`` node that consumes a ``replicate`` node.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern
            ``sum_vmapped(replicate(x, times, pos1), pos2)``, False otherwise.
        """
        return (
            node.op == "call_function"
            and node.users
            and node.target == _sum_vmapped_target
            and node.all_input_nodes
            and is_replicate(node.all_input_nodes[0])
        )

    def apply(self, sum_node: Node, graph: Graph) -> None:
        """Apply the simplification rule.

        Args:
            sum_node: The ``sum_vmapped`` node that consumes a ``replicate`` node.
            graph: The computation graph to which the rule is applied.
        """
        (rep_node,) = sum_node.all_input_nodes
        (x,) = rep_node.all_input_nodes
        pos_rep = _get_replicate_pos(rep_node)
        pos_sum = _get_sum_vmapped_pos(sum_node)
        times = _get_replicate_times(rep_node)

        if pos_sum == pos_rep:
            # Insert a multiplication node before the replicate node
            with graph.inserting_before(rep_node):
                mul_node = graph.call_function(_aten.mul.Tensor, args=(x, times))
            sum_node.replace_all_uses_with(mul_node)

        else:
            # Insert a new sum_vmapped node before the sum node
            new_sum_pos = pos_sum if pos_rep > pos_sum else pos_sum - 1
            with graph.inserting_before(sum_node):
                sum_args, sum_kwargs = _make_sum_vmapped_args(x, new_sum_pos)
                new_sum_node = graph.call_function(
                    _sum_vmapped_target, args=sum_args, kwargs=sum_kwargs
                )
            # Insert a new replicate node after the new sum node
            new_rep_pos = pos_rep - 1 if pos_rep > pos_sum else pos_rep
            with graph.inserting_after(new_sum_node):
                rep_args, rep_kwargs = _make_replicate_args(
                    new_sum_node, times, new_rep_pos
                )
                new_rep_node = graph.call_function(
                    _replicate_target, args=rep_args, kwargs=rep_kwargs
                )

            # Replace the old node with its simplified node in the entire graph
            sum_node.replace_all_uses_with(new_rep_node)


class PullSumVmappedScalarMultiplication(Rule):
    """Rule for simplifying ``sum_vmapped(x * y)`` with one scalar argument.

    The following two cases simplify to the same result:

    1. ``x`` scalar: ``sum_vmapped(x * y)`` -> ``x * sum_vmapped(y)``.
    2. ``y`` scalar: ``sum_vmapped(x * y)`` -> ``sum_vmapped(x) * y``.

    Attributes:
        OPERATIONS: List of operations that can be simplified.
    """

    OPERATIONS: list[Callable[[Tensor | float | int, Tensor | float | int], Tensor]] = [
        _aten.mul.Tensor,
    ]

    def match(self, node: Node) -> bool:
        """Match for sum_vmapped nodes that consume multiplications with a scalar.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern ``sum_vmapped(x * y)``, where ``*`` is
            multiplication and either ``x`` or ``y`` a scalar, False otherwise.
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
            sum_node: A ``sum_vmapped`` node that consumes a node representing
                multiplication with a scalar/float. Must satisfy the match condition.
            graph: The computation graph to which the rule is applied.
        """
        # find the multiplication node and its input tensor
        (mul_node,) = sum_node.all_input_nodes
        (x,) = mul_node.all_input_nodes
        x_pos = mul_node.args.index(x)

        # swap the order of the `sum_vmapped` and the arithmetic operation
        pos = _get_sum_vmapped_pos(sum_node)
        with graph.inserting_after(sum_node):
            sum_args, sum_kwargs = _make_sum_vmapped_args(x, pos)
            new_sum_node = graph.call_function(
                _sum_vmapped_target, args=sum_args, kwargs=sum_kwargs
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
    """Rule for simplifying ``sum_vmapped(x + y)`` where x and y are tensors.

    The simplified result is ``sum_vmapped(x) + sum_vmapped(y)``.
    Same for subtraction.

    Warning:
        This rule assumes no broadcasting, i.e. ``x`` and ``y`` must have the same shape.

    Attributes:
        OPERATIONS: List of operations that can be simplified.
            Includes addition and subtraction.
    """

    OPERATIONS: list[Callable[[Tensor | float | int, Tensor | float | int], Tensor]] = [
        _aten.add.Tensor,
        _aten.sub.Tensor,
    ]

    def match(self, node: Node) -> bool:
        """Match for sum_vmapped nodes that consume a summation/subtraction node.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern ``sum_vmapped(x + y)`` (or -), where
            ``x`` and ``y`` are tensors, False otherwise.
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
            sum_node: A ``sum_vmapped`` node that consumes a node representing addition/
                subtraction of two tensors. Must satisfy the match condition.
            graph: The computation graph to which the rule is applied.
        """
        # find the addition/subtraction node and its input tensor
        (add_node,) = sum_node.all_input_nodes
        pos = _get_sum_vmapped_pos(sum_node)

        mapping = {}
        # swap the order of the `sum_vmapped` and the addition/subtraction operation
        for x in add_node.all_input_nodes:
            with graph.inserting_after(x):
                sum_args, sum_kwargs = _make_sum_vmapped_args(x, pos)
                new_sum_node = graph.call_function(
                    _sum_vmapped_target, args=sum_args, kwargs=sum_kwargs
                )
            mapping[x] = new_sum_node

        # Insert a new addition/subtraction node after the new sum nodes
        with graph.inserting_after(add_node):
            new_args = tuple(mapping[x] for x in add_node.args)
            new_add_node = graph.call_function(add_node.target, args=new_args)

        # replace the old node with its simplified node in the entire graph
        sum_node.replace_all_uses_with(new_add_node)


class PullSumVmappedLinear(Rule):
    """Simplify ``sum_vmapped(linear(x, W, 0))`` into ``linear(sum_vmapped(x), W, 0)``.

    Note:
        With ``make_fx`` tracing, ``linear`` decomposes into lower-level ops
        (``mm``, ``addmm``, ``t``, ``view``), so this rule will not match.
        It is kept for backward compatibility.
    """

    def match(self, node: Node) -> bool:
        """Match for sum_vmapped nodes that consume a linear operation.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern ``sum_vmapped(linear(x, W, b))``,
            False otherwise.
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
            sum_node: A ``sum_vmapped`` node that consumes a ``linear`` node. Must
                satisfy the match condition.
            graph: The computation graph to which the rule is applied.
        """
        (linear_node,) = sum_node.all_input_nodes
        x = linear_node.args[0]
        pos = _get_sum_vmapped_pos(sum_node)

        warn(
            "The `PullSumVmappedLinear` rule assumes that the summed axis is not "
            f"the last axis. If it is, the rule will fail. Got {pos=}.",
            stacklevel=2,
        )

        # swap the order of the `sum_vmapped` and the linear operation
        with graph.inserting_after(x):
            sum_args, sum_kwargs = _make_sum_vmapped_args(x, pos)
            new_sum_node = graph.call_function(
                _sum_vmapped_target, args=sum_args, kwargs=sum_kwargs
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
    """Simplify ``sum_vmapped(y * replicate(x, times, pos1), pos2)``.

    This rule applies when ``pos1 == pos2`` and simplifies the expression into
    ``sum_vmapped(y, pos2) * x``.
    It also assumes that both tensors that are being multiplied have the same shape.

    Attributes:
        OPERATIONS: List of multiplication operations that can be simplified.
    """

    OPERATIONS = [_aten.mul.Tensor]

    def match(self, node: Node) -> bool:
        """Detect a match with ``sum_vmapped(y * replicate(x, times, pos), pos)``.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern
            ``sum_vmapped(y * replicate(x, times, pos1), pos2)`` with
            ``pos1 == pos2``, False otherwise.
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
            sum_pos = _get_sum_vmapped_pos(node)
            rep_pos = _get_replicate_pos(rep_node)
            return sum_pos == rep_pos

        return False

    def apply(self, sum_node: Node, graph: Graph) -> None:
        """Apply the simplification rule.

        Args:
            sum_node: The ``sum_vmapped`` node that consumes a multiplication node.
            graph: The computation graph to which the rule is applied.
        """
        (mul_node,) = sum_node.all_input_nodes
        (rep_node,) = [n for n in mul_node.all_input_nodes if is_replicate(n)]
        (x_node,) = rep_node.all_input_nodes
        (other_node,) = [n for n in mul_node.all_input_nodes if not is_replicate(n)]

        pos = _get_sum_vmapped_pos(sum_node)

        # Create a new sum_vmapped node
        with graph.inserting_before(sum_node):
            sum_args, sum_kwargs = _make_sum_vmapped_args(other_node, pos)
            new_sum_node = graph.call_function(
                _sum_vmapped_target, args=sum_args, kwargs=sum_kwargs
            )

        # Create a new multiplication node for `sum_vmapped(y) * x`
        with graph.inserting_after(new_sum_node):
            mapping = {rep_node: x_node, other_node: new_sum_node}
            new_mul_node = graph.call_function(
                mul_node.target, args=tuple(mapping[arg] for arg in mul_node.args)
            )

        # Replace the old node with the simplified node
        sum_node.replace_all_uses_with(new_mul_node)
