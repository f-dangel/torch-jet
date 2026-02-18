"""Implements individual simplification rules."""

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
from torch import Tensor
from torch.fx import Graph, Node

# ATen op references used by rules
_aten = torch.ops.aten

# Op targets for pattern matching
_sum_target = _aten.sum.dim_IntList


def is_sum(arg: Any) -> bool:
    """Check if an argument is a ``sum`` node.

    Args:
        arg: An entry from a ``Node.arg`` tuple.

    Returns:
        Whether the argument is a sum node.
    """
    return (
        isinstance(arg, Node)
        and arg.op == "call_function"
        and arg.target == _sum_target
    )


def _get_sum_pos(node: Node) -> int:
    """Extract the summed dimension from an ``aten.sum.dim_IntList`` node.

    With make_fx, ``x.sum(pos)`` traces to ``aten.sum.dim_IntList(x, [pos])``,
    so the dimension is always at ``node.args[1][0]``.

    Args:
        node: A sum node.

    Returns:
        The position along which the sum is performed.
    """
    return node.args[1][0]


def _make_sum_args(x: Any, pos: int) -> tuple[tuple, dict]:
    """Create args and kwargs for a new ``aten.sum.dim_IntList`` node.

    Args:
        x: The input tensor node.
        pos: The position to sum along.

    Returns:
        Tuple of (args, kwargs) for graph.call_function.
    """
    return (x, [pos]), {}


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


class PullSumScalarMultiplication(Rule):
    """Rule for simplifying ``sum(x * y)`` with one scalar argument.

    The following two cases simplify to the same result:

    1. ``x`` scalar: ``sum(x * y)`` -> ``x * sum(y)``.
    2. ``y`` scalar: ``sum(x * y)`` -> ``sum(x) * y``.

    Attributes:
        OPERATIONS: List of operations that can be simplified.
    """

    OPERATIONS: list[Callable[[Tensor | float | int, Tensor | float | int], Tensor]] = [
        _aten.mul.Tensor,
    ]

    def match(self, node: Node) -> bool:
        """Match for sum nodes that consume multiplications with a scalar.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern ``sum(x * y)``, where ``*`` is
            multiplication and either ``x`` or ``y`` a scalar, False otherwise.
        """
        if not is_sum(node) or not node.users:
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
            sum_node: A ``sum`` node that consumes a node representing
                multiplication with a scalar/float. Must satisfy the match condition.
            graph: The computation graph to which the rule is applied.
        """
        # find the multiplication node and its input tensor
        (mul_node,) = sum_node.all_input_nodes
        (x,) = mul_node.all_input_nodes
        x_pos = mul_node.args.index(x)

        # swap the order of the `sum` and the arithmetic operation
        pos = _get_sum_pos(sum_node)
        with graph.inserting_after(sum_node):
            sum_args, sum_kwargs = _make_sum_args(x, pos)
            new_sum_node = graph.call_function(
                _sum_target, args=sum_args, kwargs=sum_kwargs
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


class PullSumTensorAddition(Rule):
    """Rule for simplifying ``sum(x + y)`` where x and y are tensors.

    The simplified result is ``sum(x) + sum(y)``.
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
        """Match for sum nodes that consume a summation/subtraction node.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern ``sum(x + y)`` (or -), where
            ``x`` and ``y`` are tensors, False otherwise.
        """
        if not is_sum(node) or not node.users:
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
            sum_node: A ``sum`` node that consumes a node representing addition/
                subtraction of two tensors. Must satisfy the match condition.
            graph: The computation graph to which the rule is applied.
        """
        # find the addition/subtraction node and its input tensor
        (add_node,) = sum_node.all_input_nodes
        pos = _get_sum_pos(sum_node)

        mapping = {}
        # swap the order of the `sum` and the addition/subtraction operation
        for x in add_node.all_input_nodes:
            with graph.inserting_after(x):
                sum_args, sum_kwargs = _make_sum_args(x, pos)
                new_sum_node = graph.call_function(
                    _sum_target, args=sum_args, kwargs=sum_kwargs
                )
            mapping[x] = new_sum_node

        # Insert a new addition/subtraction node after the new sum nodes
        with graph.inserting_after(add_node):
            new_args = tuple(mapping[x] for x in add_node.args)
            new_add_node = graph.call_function(add_node.target, args=new_args)

        # replace the old node with its simplified node in the entire graph
        sum_node.replace_all_uses_with(new_add_node)


class PullSumLinear(Rule):
    """Pull ``sum`` through ``mm`` and ``addmm``.

    Matches ``sum(op(…), pos)`` where ``op`` is ``mm`` or ``addmm``
    and ``pos`` identifies a non-contracted dimension. Moves the summation to
    the corresponding matrix argument:

    For ``mm(X, W)``:

    - ``pos == 0``: → ``squeeze(mm(unsqueeze(sum(X, 0), 0), W), 0)``
    - ``pos == 1``: → ``squeeze(mm(X, unsqueeze(sum(W, 1), 1)), 1)``

    For ``addmm(b, X, W)`` the matrix part is handled identically, plus a
    bias correction (``V * b`` for pos=0, ``sum(b)`` for pos=1).

    Attributes:
        OPERATIONS: Supported ATen matrix-multiply operations.
    """

    OPERATIONS = [_aten.mm.default, _aten.addmm.default]

    @staticmethod
    def _matrix_arg_indices(node: Node) -> tuple[int, int]:
        """Return the argument indices of the two matrix operands.

        Args:
            node: An ``mm`` or ``addmm`` node.

        Returns:
            Tuple of ``(mat1_index, mat2_index)``.
        """
        if node.target == _aten.mm.default:
            return (0, 1)
        return (1, 2)

    def match(self, node: Node) -> bool:
        """Detect ``sum(mm/addmm(…), pos)`` with ``pos`` in {0, 1}.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches, False otherwise.
        """
        if not is_sum(node) or not node.users:
            return False

        (in_node,) = node.all_input_nodes
        if in_node.op != "call_function" or in_node.target not in self.OPERATIONS:
            return False

        pos = _get_sum_pos(node)
        # pos must be 0 (mat1's row dim) or 1 (mat2's col dim)
        return pos in (0, 1)

    def apply(self, sum_node: Node, graph: Graph) -> None:
        """Apply the simplification rule.

        For ``mm``, replaces ``sum(mm(…), pos)`` with
        ``squeeze(mm(…, unsqueeze(sum(arg, pos), pos), …), pos)``.

        For ``addmm``, the same matrix rewrite is applied (using ``mm``),
        and the bias is corrected: ``V * b`` when ``pos == 0`` (the bias was
        broadcast across ``V`` rows), or ``sum(b)`` when ``pos == 1``.

        Args:
            sum_node: The ``sum`` node.
            graph: The computation graph to which the rule is applied.
        """
        (matmul_node,) = sum_node.all_input_nodes
        pos = _get_sum_pos(sum_node)
        i1, i2 = self._matrix_arg_indices(matmul_node)
        # pos=0 → pull sum to mat1, pos=1 → pull to mat2
        target_idx = i1 if pos == 0 else i2
        target_arg = matmul_node.args[target_idx]

        # sum on the target argument
        with graph.inserting_before(matmul_node):
            sv_args, sv_kwargs = _make_sum_args(target_arg, pos)
            new_sv = graph.call_function(_sum_target, args=sv_args, kwargs=sv_kwargs)

        # unsqueeze to restore 2D shape for mm
        with graph.inserting_after(new_sv):
            unsqueeze_node = graph.call_function(
                _aten.unsqueeze.default, args=(new_sv, pos)
            )

        # new mm with sum+unsqueeze replacing the original arg
        with graph.inserting_after(unsqueeze_node):
            mm_args = tuple(
                unsqueeze_node if i == target_idx else matmul_node.args[i]
                for i in (i1, i2)
            )
            new_mm = graph.call_function(_aten.mm.default, args=mm_args)

        # squeeze the summed dimension from the result
        with graph.inserting_after(new_mm):
            squeeze_node = graph.call_function(_aten.squeeze.dim, args=(new_mm, pos))

        if matmul_node.target == _aten.addmm.default:
            bias = matmul_node.args[0]
            if pos == 0:
                # Bias b was broadcast from (n,) to (V, n); summing V rows
                # gives V * b. Extract V from the traced tensor metadata.
                V = target_arg.meta["val"].shape[0]
                with graph.inserting_after(squeeze_node):
                    scaled_b = graph.call_function(_aten.mul.Tensor, args=(bias, V))
                with graph.inserting_after(scaled_b):
                    result = graph.call_function(
                        _aten.add.Tensor, args=(squeeze_node, scaled_b)
                    )
            else:
                # pos == 1: each row had b added; summing over n columns
                # contributes sum(b) to every row.
                with graph.inserting_after(squeeze_node):
                    sum_b = graph.call_function(_aten.sum.default, args=(bias,))
                with graph.inserting_after(sum_b):
                    result = graph.call_function(
                        _aten.add.Tensor, args=(squeeze_node, sum_b)
                    )
            sum_node.replace_all_uses_with(result)
        else:
            sum_node.replace_all_uses_with(squeeze_node)
