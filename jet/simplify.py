"""Functions to simplify a compute graph captured with `torch.fx`."""

import operator
from typing import Any, List, Optional, Tuple

from torch import add, cos, cosh, div, einsum, mul
from torch import pow as torch_pow
from torch import sigmoid, sin, sub, tanh
from torch.fx import Graph, GraphModule, Node
from torch.nn.functional import linear

from jet.utils import replicate, sum_vmapped


class RewriteReplicate:
    """Class for propagating `replicate` nodes down a compute graph.

    Allows to simplify the compute graph by avoiding redundant computations on
    replicated tensors.
    """

    def __init__(self, mod: GraphModule, verbose: bool = False):
        """Store the module .

        Args:
            mod: The compute graph module.
            verbose: Whether to print debug information. Default: `False`.
        """
        self.mod = mod
        self.graph = mod.graph
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


class RewriteSumVmapped(RewriteReplicate):
    """Propagates summations over a vmaped axis up a computation graph."""

    @staticmethod
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
            and arg.target == sum_vmapped
        )

    def find_pattern(self) -> Optional[Tuple[Node, Node]]:
        for node in self.graph.nodes:

            if (
                node.op == "call_function"
                and node.target == einsum
                and node.args[0].split("->")[1] == "..."
            ):
                # hoist out tensors with 'a...'
                lhs = node.args[0].split("->")[0].split(",")
                # print(lhs)
                num_usages = {n: node.args[1:].count(n) for n in node.args[1:]}
                for idx, (l, arg) in enumerate(zip(lhs, node.args[1:])):
                    if l == "a..." and num_usages[arg] == 1:
                        with self.graph.inserting_before(node):
                            new_sum = self.graph.call_function(sum_vmapped, args=(arg,))
                        node.replace_input_with(arg, new_sum)
                        lhs[idx] = "..."
                        new_equation = f"{','.join(lhs)}->..."
                        node.args = (new_equation, *node.args[1:])

            if not self.is_sum_vmapped(node):
                continue
            (op,) = self.parents(node)

            pattern = None
            parents = self.parents(op)

            if op.op == "get_attr" and not parents and len(self.children(op)) == 1:
                pattern = [node, op]
            elif (
                op.op == "call_function"
                and op.target in {operator.mul}
                and len(parents) == 1
            ):
                pattern = [node, op]
            elif (
                op.op == "call_function"
                and op.target in {operator.add}
                and len(parents) == 2
            ):
                pattern = [node, op]
            elif op.op == "call_function" and op.target == linear:
                pattern = [node, op]
            elif (
                op.op == "call_function"
                and op.target == einsum
                and op.args[0].split("->")[1] == "a..."
            ):
                pattern = [node, op]

            if pattern is not None:
                self.maybe_print(f"Can propagate {pattern[0]} up {pattern[1]}.")
                return pattern

    def replace_pattern(self, pattern: Tuple[Node, Node]):
        sum_node, op = pattern
        parents = self.parents(op)

        sum_node.replace_all_uses_with(op)
        self.maybe_erase(sum_node)

        if op.op == "get_attr":
            # sum the tensor constant
            old = getattr(self.mod, op.target)
            new = sum_vmapped(old)
            setattr(self.mod, op.target, new)
            self.maybe_print(
                f"Collapsing {op}: {tuple(old.shape)} -> {tuple(new.shape)}."
            )
            return

        if op.target == einsum and op.args[0].split("->")[1] == "a...":
            # change the equation
            lhs, _ = op.args[0].split("->")
            new_equation = f"{lhs}->..."
            op.args = (new_equation, *op.args[1:])

            # check for nodes that can be fused
            lhs_args = lhs.split(",")
            num_usages = {node: op.args[1:].count(node) for node in op.args[1:]}
            for lhs, arg in zip(lhs_args, op.args[1:]):
                # print("\t", lhs, arg, num_usages[arg])
                if num_usages[arg] != 1:
                    continue
                parents = self.parents(arg)

                if not parents:
                    children = self.children(arg)
                    # print("\t\t", f"{arg} has children {children}")
                    if len(children) == 1:
                        # insert a sum_vmaped node
                        with self.graph.inserting_before(op):
                            new_sum = self.graph.call_function(sum_vmapped, args=(arg,))
                        op.replace_input_with(arg, new_sum)
                        where = op.args[1:].index(new_sum)
                        lhs_args[where] = "..."
                        new_equation = f"{','.join(lhs_args)}->..."
                        op.args = (new_equation, *op.args[1:])

            return

        if op.target == linear:
            input_node = op.args[0]
            with self.graph.inserting_before(op):
                new_sum = self.graph.call_function(sum_vmapped, args=(input_node,))
            op.replace_input_with(input_node, new_sum)
            return

        parents_to_sum = {}
        for parent in parents:
            with self.graph.inserting_before(op):
                new_sum = self.graph.call_function(sum_vmapped, args=(parent,))
            parents_to_sum[parent] = new_sum

        op.args = tuple(parents_to_sum.get(arg, arg) for arg in op.args)

        for parent in parents:
            self.maybe_erase(parent)


def remove_duplicate_get_attrs(graph: Graph, verbose: bool = False):
    # find all nodes that getattr the same constant
    mapping = {}
    for node in graph.nodes:
        if node.op != "get_attr":
            continue

        if node.target not in mapping:
            mapping[node.target] = [node]
        else:
            mapping[node.target].append(node)

    for nodes in mapping.values():
        n1 = nodes[0]
        if verbose:
            print(f"Replacing {n1} with {nodes[1:]}")
        for n in nodes[1:]:
            n.replace_all_uses_with(n1)
            graph.erase_node(n)


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

    # this assumes no side effects in the graph, so that it does not matter
    # when we call get_attr on a module's tensor constant
    remove_duplicate_get_attrs(mod.graph, verbose=verbose)

    rewriter = RewriteReplicate(mod, verbose=verbose)
    while pattern := rewriter.find_pattern():
        rewriter.replace_pattern(pattern)

    rewriter.fuse_replicates_with_einsum()

    rewriter = RewriteSumVmapped(mod, verbose=verbose)
    while pattern := rewriter.find_pattern():
        rewriter.replace_pattern(pattern)

    mod.graph.lint()
    mod.recompile()

    if verbose:
        print(f"Traced graph after simplification:\n{mod.graph}")

    return mod
