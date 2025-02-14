"""Functions to simplify a compute graph captured with `torch.fx`."""

import operator
from typing import Any, List, Optional, Tuple

from torch import add, cos, cosh, div, einsum, mul
from torch import pow as torch_pow
from torch import sigmoid, sin, sub, tanh
from torch.fx import Graph, GraphModule, Node
from torch.nn.functional import linear

from jet.utils import replicate


class RewriteReplicate:
    """Class for propagating `replicate` nodes down a compute graph.

    Allows to simplify the compute graph by avoiding redundant computations on
    replicated tensors.
    """

    def __init__(self, graph: Graph, verbose: bool = False):
        """Store the graph.

        Args:
            graph: The compute graph to simplify.
            verbose: Whether to print debug information. Default: `False`.
        """
        self.graph = graph
        self.verbose = verbose

    @staticmethod
    def is_replicate(arg: Any) -> bool:
        """Check if an argument of a node is a `replicate` node.

        Args:
            arg: An entry from a `Node.arg` tuple.

        Returns:
            Whether the argument is a `replicate` node.
        """
        return (
            isinstance(arg, Node)
            and arg.op == "call_function"
            and arg.target == replicate
        )

    def parents(self, node: Node) -> List[Node]:
        """Get the parent nodes of a given node.

        Args:
            node: The node whose parents to find.

        Returns:
            The parent nodes of the given node.
        """
        return [n for n in self.graph.nodes if n in node.all_input_nodes]

    def children(self, node: Node) -> List[Node]:
        """Get the children nodes of a given node.

        Args:
            node: The node whose children to find.

        Returns:
            The children nodes of the given node.
        """
        return [n for n in self.graph.nodes if node in n.all_input_nodes]

    def find_pattern(self) -> Optional[Tuple[List[Node], Node]]:
        """Find a pattern that can be simplified.

        Returns:
            A pattern that can be simplified, or `None` if no such pattern is found.
            A pattern consists of two parts: a list of nodes that can be swapped with
            the node returned as second part.
        """
        for node in self.graph.nodes:
            if node.op != "call_function":
                continue

            pattern = None
            parents = self.parents(node)

            # operations that consume a single replicate tensor `x_rep = replicate(x)`
            if (
                node.target
                in {
                    cos,  # torch.cos(x_rep) -> replicate(torch.cos(x))
                    tanh,  # torch.tanh(x_rep) -> replicate(torch.tanh(x))
                    sigmoid,  # torch.sigmoid(x_rep) -> replicate(torch.sigmoid(x))
                    cosh,  # torch.cosh(x_rep) -> replicate(torch.cosh(x))
                    torch_pow,  # torch.pow(x_rep, 2) -> replicate(torch.pow(x, 2))
                    sin,  # torch.sin(x_rep) -> replicate(torch.sin(x))
                    operator.pow,  # x_rep ** 2 -> replicate(x ** 2)
                    mul,  # torch.mul(x_rep, 2) -> replicate(torch.mul(x, 2))
                    operator.mul,  # x_rep * 2 -> replicate(x * 2)
                    div,  # torch.div(x_rep, 2) -> replicate(torch.div(x, 2))
                    operator.truediv,  # x_rep / 2 -> replicate(x / 2)
                    operator.add,  # x_rep + 2 -> replicate(x + 2)
                    add,  # torch.add(x_rep, 2) -> replicate(torch.add(x, 2))
                    operator.sub,  # x_rep - 2 -> replicate(x - 2)
                    sub,  # torch.sub(x_rep, 2) -> replicate(torch.sub(x, 2))
                }
                and len(parents) == 1
                and all(self.is_replicate(p) for p in parents)
            ):
                pattern = [parents, node]

            # a linear layer that processes a replicated input tensor
            # `x_rep = replicate(x)`:
            # `linear(x_rep, W, bias=b)` -> `replicate(linear(x), W, bias=b)`
            elif node.target == linear and self.is_replicate(node.args[0]):
                pattern = [[node.args[0]], node]

            # operations that consume two replicate tensors `x_rep1 = replicate(x1)`,
            # `x_rep2 = replicate(x2)`
            elif (
                node.target
                in {
                    add,  # torch.add(x_rep1, x_rep2) -> replicate(torch.add(x1, x2))
                    operator.add,  # x_rep1 + x_rep2 -> replicate(x1 + x2)
                    mul,  # torch.mul(x_rep1, x_rep2) -> replicate(torch.mul(x1, x2))
                    operator.mul,  # x_rep1 * x_rep2 -> replicate(x1 * x2)
                    sub,  # torch.sub(x_rep1, x_rep2) -> replicate(torch.sub(x1, x2))
                    operator.sub,  # x_rep1 - x_rep2 -> replicate(x1 - x2)
                }
                and len(parents) == 2
                and all(self.is_replicate(arg) for arg in node.args)
            ):
                pattern = [parents, node]

            # operations that consume multiple replicate tensors, e.g.
            # `x_rep1 = replicate(x1)`, `x_rep2 = replicate(x2)` and
            # `x_rep3 = replicate(x3)`. Then
            # `einsum("...,...,...->...", x_rep1, x_rep2, x_rep3)`
            # -> `replicate(einsum("...,...,...->...", x1, x2, x3))`
            elif (
                node.target == einsum
                and node.args[0] == ",".join(len(node.args[1:]) * ["..."]) + "->..."
                and all(self.is_replicate(arg) for arg in node.args[1:])
            ):
                # remove duplicates if the same node feeds multiple times into einsum
                pattern = [list(set(node.args[1:])), node]

            if pattern is not None:
                self.maybe_print(f"Can swap {pattern[0]} and {pattern[1]}")
                return pattern

    def maybe_erase(self, rep: Node):
        """Remove a replicate node if it has no children.

        Args:
            rep: The replicate node.
        """
        children = self.children(rep)
        if len(children) == 0:
            self.maybe_print(f"Erasing {rep}.")
            self.graph.erase_node(rep)
        else:
            self.maybe_print(f"Not removing {rep} because it has children {children}.")

    def replace_pattern(self, pattern: Tuple[List[Node], Node]):
        """Replace a pattern in the graph.

        Args:
            pattern: A pattern returned by `find_pattern`.
        """
        replicates, op = pattern
        # figure out the parent of each replicate node because we want to skip
        # the replicate nodes
        replicate_parents = {}
        for rep in replicates:
            (rep_parent,) = self.parents(rep)
            replicate_parents[rep] = rep_parent

        # create a new replicate node that replaces the old one and is located after
        # the operation node
        with self.graph.inserting_after(op):
            new_rep = self.graph.call_function(replicate, kwargs=replicates[0].kwargs)
        op.replace_all_uses_with(new_rep)

        # rewire the arguments
        op.args = tuple(replicate_parents.get(arg, arg) for arg in op.args)
        new_rep.args = (op,) + replicates[0].args[1:]

        # remove the old replicate nodes if possible
        for rep in replicates:
            self.maybe_erase(rep)

    def maybe_print(self, message: str):
        """Print a message if verbose mode is enabled.

        Args:
            message: The message to print.
        """
        if self.verbose:
            print(message)

    def fuse_replicates_with_einsum(self):
        """Attempt to fuse replicate nodes that act as inputs to einsum.

        E.g. consider einsum('...,...->...', replicate(x), y). This can be simplified
        into einsum('...,a...->a...', x, y)).
        """
        for node in self.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == einsum
                and node.args[0] == ",".join(len(node.args[1:]) * ["..."]) + "->..."
                and any(self.is_replicate(arg) for arg in node.args[1:])
            ):
                replicates = [arg for arg in node.args[1:] if self.is_replicate(arg)]
                replicate_parents = {}
                for rep in replicates:
                    (rep_parent,) = self.parents(rep)
                    replicate_parents[rep] = rep_parent

                # modify the einsum equation and operands
                lhs = [
                    "..." if self.is_replicate(arg) else "a..." for arg in node.args[1:]
                ]
                equation = ",".join(lhs) + "->a..."
                new_args = (
                    equation,
                    *(replicate_parents.get(arg, arg) for arg in node.args[1:]),
                )
                self.maybe_print(f"Fusing {node}: {node.args} into {new_args}")
                node.args = new_args

                # try removing the old replicate nodes
                for rep in replicates:
                    self.maybe_erase(rep)


def simplify(mod: GraphModule, verbose: bool = False) -> GraphModule:
    """Simplify a compute graph.

    At the moment, the following simplifications are implemented:

    - Propagation of `replicate` nodes down the graph as much as possible.
      This avoids redundant computations on replicated tensors.

    Args:
        mod: A graph module whose computation graph will be simplified.
        verbose: Whether to print debug information. Default: `False`.

    Returns:
        The simplified graph module.
    """
    if verbose:
        print(f"Traced graph before simplification:\n{mod.graph}")

    rewriter = RewriteReplicate(mod.graph, verbose=verbose)
    while pattern := rewriter.find_pattern():
        rewriter.replace_pattern(pattern)

    rewriter.fuse_replicates_with_einsum()

    mod.graph.lint()
    mod.recompile()

    if verbose:
        print(f"Traced graph after simplification:\n{mod.graph}")

    return mod
