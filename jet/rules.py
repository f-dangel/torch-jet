"""Implements individual simplification rules."""

from abc import ABC, abstractmethod
from typing import Any

from torch import cos, sigmoid, sin, tanh
from torch.fx import Graph, Node

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


class Rule(ABC):
    """Base class for simplification rules."""

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


class SwapReplicateElementwise(Rule):
    """Rule for simplifying `replicate(f(x))` into `f(replicate(x))`.

    `f` is an elementwise function, such as `sin`, `cos`, or `tanh`, `sigmoid`.
    """

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
            and node.target in {cos, sin, tanh, sigmoid}  # must be elementwise
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
        (rep_node,) = f_node.all_input_nodes
        (x,) = rep_node.all_input_nodes

        with graph.inserting_after(rep_node):
            new_f_node = graph.call_function(f_node.target, args=(x,))

        with graph.inserting_after(new_f_node):
            new_rep_node = graph.call_function(
                jet.utils.replicate,
                args=(new_f_node, *rep_node.args[1:]),
                kwargs=rep_node.kwargs,
            )

        f_node.replace_all_uses_with(new_rep_node)
