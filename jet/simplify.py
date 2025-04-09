"""Functions to simplify a compute graph captured with `torch.fx`."""

import operator
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

from torch import Tensor, add, cos, cosh, div, einsum, mul
from torch import pow as torch_pow
from torch import sigmoid, sin, sub, tanh
from torch.fx import Graph, GraphModule, Node
from torch.nn.functional import linear

from jet.utils import get_letters, replicate, sum_vmapped


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
                and all(self.is_replicate(arg) for arg in node.all_input_nodes)
            ):
                pattern = [parents, node]

            # operations that consume multiple replicate tensors, e.g.
            # `x_rep1 = replicate(x1)`, `x_rep2 = replicate(x2)` and
            # `x_rep3 = replicate(x3)`. Then
            # `einsum("...,...,...->...", x_rep1, x_rep2, x_rep3)`
            # -> `replicate(einsum("...,...,...->...", x1, x2, x3))`
            elif (
                node.target == einsum
                # NOTE This assumption is overly simplistic but sufficient for now
                and node.args[0]
                == f"{','.join((node.args[0].count(',') + 1) * ['...'])}->..."
                and all(self.is_replicate(arg) for arg in node.all_input_nodes)
            ):
                pattern = [list(node.all_input_nodes), node]

            if pattern is not None:
                self.maybe_print(f"Can swap {pattern[0]} and {pattern[1]}")
                return pattern

    def maybe_erase(self, node: Node) -> bool:
        """Remove a node if it has no children.

        Args:
            node: The node to be checked for removal.

        Returns:
            Whether the node was removed.
        """
        if node.op == "output":
            self.maybe_print(f"Not removing {node} because it is an output node.")
            return False

        children = self.children(node)
        if len(children) == 0:
            self.maybe_print(f"Erasing {node}.")
            self.graph.erase_node(node)
            return True
        else:
            self.maybe_print(f"Not removing {node} because it has children {children}.")
            return False

    def replace_pattern(self, pattern: Tuple[List[Node], Node]):
        """Replace a pattern in the graph.

        Args:
            pattern: A pattern returned by `find_pattern`.
        """
        replicates, op = pattern

        # create a new replicate node that replaces the old one and is located after
        # the operation node
        with self.graph.inserting_after(op):
            new_rep = self.graph.call_function(replicate, kwargs=replicates[0].kwargs)
        op.replace_all_uses_with(new_rep)
        new_rep.args = (op,) + replicates[0].args[1:]

        # rewire the arguments
        for rep in replicates:
            (parent,) = self.parents(rep)
            op.replace_input_with(rep, parent)

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
                # NOTE This assumption is overly simplistic but sufficient for now
                and node.args[0]
                == f"{','.join((node.args[0].count(',') + 1) * ['...'])}->..."
                and any(self.is_replicate(arg) for arg in node.all_input_nodes)
            ):
                old_args = node.args
                # modify the operands
                positions = [
                    i for i, arg in enumerate(node.args[1:]) if self.is_replicate(arg)
                ]
                for rep in list(node.all_input_nodes):
                    if self.is_replicate(rep):
                        (parent,) = self.parents(rep)
                        node.replace_input_with(rep, parent)
                        # try removing the old replicate nodes
                        self.maybe_erase(rep)

                # modify the einsum equation
                new_lhs = [
                    "..." if i in positions else "a..."
                    for i in range(len(node.args[1:]))
                ]
                new_equation = f"{','.join(new_lhs)}->a..."
                node.args = (new_equation, *node.args[1:])
                self.maybe_print(f"Fusing {node}: {old_args} into {node.args}")

    def remove_unused_nodes(self):
        """Remove unused nodes from the graph."""
        num_removals = 0
        for node in list(self.graph.nodes):
            removed = self.maybe_erase(node)
            num_removals = num_removals + 1 if removed else num_removals

        if num_removals > 0:
            self.maybe_print(f"Removed {num_removals} unused nodes.")
            # the removed nodes might have revealed new unused nodes
            self.remove_unused_nodes()


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
        """Find a pattern that can be simplified.

        Returns:
            A pattern that can be simplified, or `None` if no such pattern is found.
            A pattern consists of two parts: them `sum_vmapped` node that can be
            propagated up through the node returned as second part.
        """
        for node in self.graph.nodes:
            if not self.is_sum_vmapped(node):
                continue
            (op,) = self.parents(node)
            if op.op != "call_function":
                continue

            pattern = None
            parents = self.parents(op)

            # operations that produce a tensor from a single tensor `x`, which is then
            # `sum_vmapped`
            if (
                op.target
                in {
                    operator.mul,  # sum_vmapped(x * 2) -> 2 * sum_vmapped(x)
                }
                and len(parents) == 1
            ):
                pattern = [node, op]
            # operations that produce a tensor from two tensors `x`, `y`, which is then
            # `sum_vmapped`
            elif (
                op.target
                in {
                    # NOTE This assumes there is no broadcasting (x.shape == y.shape)!
                    # sum_vmapped(x + y) -> sum_vmapped(x) + sum_vmapped(y)
                    operator.add,
                }
                and len(parents) == 2
            ):
                pattern = [node, op]
            # sum_vmapped(linear(x, W, b)) -> linear(sum_vmapped(x), W, b)
            elif op.target == linear:
                pattern = [node, op]
            # sum_vmapped(einsum('xyz->a...')) -> einsum('xyz->...')
            elif op.target == einsum:
                pattern = [node, op]

            if pattern is not None:
                self.maybe_print(f"Can propagate {pattern[0]} up {pattern[1]}.")
                return pattern

    def replace_pattern(self, pattern: Tuple[Node, Node]):
        """Replace a pattern in the graph.

        Args:
            pattern: A pattern returned by `find_pattern`. First item is the
                `sum_vmapped` node, second item is the operation node through which
                it can be propagated up.
        """
        sum_node, op = pattern
        sum_node.replace_all_uses_with(op)
        self.maybe_erase(sum_node)

        # for summing out the leading index of an einsum, we have to modify the equation
        if op.target == einsum:
            lhs, rhs = op.args[0].split("->")

            if rhs.startswith("..."):  # we need to introduce a new index
                used_letters = set(lhs)
                (new_letter,) = get_letters(1, blocked=used_letters)
                new_lhs = ",".join(
                    [lh.replace("...", f"{new_letter}...") for lh in lhs.split(",")]
                )
                new_rhs = rhs
            else:
                new_lhs = lhs
                new_rhs = rhs[1:]

            new_equation = f"{new_lhs}->{new_rhs}"
            op.update_arg(0, new_equation)

        # generate new `sum_vmapped` nodes above `op` and rewire the arguments
        else:
            parents = [op.args[0]] if op.target == linear else self.parents(op)
            for parent in parents:
                with self.graph.inserting_before(op):
                    new_sum = self.graph.call_function(sum_vmapped, args=(parent,))
                op.replace_input_with(parent, new_sum)
                self.maybe_erase(parent)

    def raise_sum_vmapped_outside_einsum(self):
        """Hoist out `sum_vmapped` nodes from a einsum operations.

        For instance einsum('a...,a...->...', x, y) can be simplified into
        einsum('...,...->...', sum_vmapped(x), sum_vmapped(y)).
        """
        ein_nodes = [
            n
            for n in self.graph.nodes
            if n.op == "call_function" and n.target == einsum
            # NOTE This assumption is overly simplistic but sufficient for now
            and n.args[0].split("->")[1] == "..."
        ]

        for ein_node in ein_nodes:
            # hoist out tensors with 'a...'
            lhs = ein_node.args[0].split("->")[0].split(",")
            usages = {n: ein_node.args[1:].count(n) for n in ein_node.all_input_nodes}
            for idx, (eq, arg) in enumerate(zip(lhs, ein_node.args[1:])):
                # NOTE This assumption is overly simplistic but sufficient for now
                if (
                    eq == "a..."
                    and usages[arg] == 1
                    # no other operand should contain the index that is summed over
                    and all("a" not in eq for i, eq in enumerate(lhs) if i != idx)
                ):
                    lhs[idx] = "..."
                    with self.graph.inserting_before(ein_node):
                        new_sum = self.graph.call_function(sum_vmapped, args=(arg,))
                    ein_node.replace_input_with(arg, new_sum)

            # update the equation
            new_equation = f"{','.join(lhs)}->..."
            ein_node.update_arg(0, new_equation)

    def fuse_vmapped_sum_with_tensor_constants(self):
        """Fuse tensor constants with `vmapped_sum` nodes.

        For instance, vmapped_sum(mod._tensor_constant0) can be simplified into
        mod._tensor_constant0 = vmapped_sum(mod._tensor_constant0)
        """
        # create a mapping which tensor constants are fetched by which nodes, and how
        # these nodes are used
        attributes = defaultdict(dict)
        for n in self.graph.nodes:
            if n.op == "get_attr":
                attributes[n.target][n] = self.children(n)

        for target, get_attr_to_children in attributes.items():
            all_children = set(sum(get_attr_to_children.values(), start=[]))

            # if all children of a get_attr are sum_vmappeds, we can sum first
            if all(self.is_sum_vmapped(c) for c in all_children):
                replacement = list(get_attr_to_children.keys())[0]
                for sum_node in all_children:
                    if sum_node != replacement:
                        sum_node.replace_all_uses_with(replacement)
                        self.maybe_erase(sum_node)

                # sum the tensor constant
                old = getattr(self.mod, target)
                new = sum_vmapped(old)
                setattr(self.mod, target, new)
                self.maybe_print(
                    f"Collapsing {target}: {tuple(old.shape)} -> {tuple(new.shape)}."
                )

            # remove the get_attr nodes that are not referenced any more
            for get_attr in get_attr_to_children:
                self.maybe_erase(get_attr)


def remove_duplicate_get_attrs(graph: Graph, verbose: bool = False):
    """Remove duplicate `get_attr` nodes.

    Args:
        graph: The compute graph.
        verbose: Whether to print debug information. Default: `False`.
    """
    # find all nodes that getattr the same constant
    mapping = {}
    for node in graph.nodes:
        if node.op == "get_attr":
            mapping[node.target] = mapping.get(node.target, []) + [node]

    for nodes in mapping.values():
        first, tail = nodes[0], nodes[1:]
        if verbose:
            print(f"Replacing {tail} with {first}")
        for n in tail:
            n.replace_all_uses_with(first)
            if verbose:
                print(f"Erasing {n}.")
            graph.erase_node(n)


@contextmanager
def check_unaltered(
    mod: GraphModule, x: Optional[Tensor], rtol: float = 1e-5, atol: float = 1e-8
):
    """Verify that the module still produces the same output before and after the body.

    Args:
        mod: The module to be checked.
        x: Input tensor to the module. If `None`, the check will be skipped.
        rtol: Relative tolerance for comparing outputs. Default: `1e-5`.
        atol: Absolute tolerance for comparing outputs. Default: `1e-8`.

    Yields:
        None

    Raises:
        RuntimeError: If the module output changes after the body.
    """
    if x is not None:
        before_str = str(mod.graph)
        out_before = mod(x)
        yield

        out_after = mod(x)
        close = out_before.allclose(out_after, rtol=rtol, atol=atol)

        if not close:
            print(f"Before:\n{before_str}")
            print(f"After:\n{mod.graph}")
            raise RuntimeError("Module output changed.")

    else:
        yield


def simplify(
    mod: GraphModule,
    push_replicate: bool = True,
    remove_unused: bool = True,
    pull_sum_vmapped: bool = True,
    verbose: bool = False,
    test_x: Optional[Tensor] = None,
) -> GraphModule:
    """Simplify a compute graph.

    At the moment, the following simplifications are implemented:

    - Pushing of `replicate` nodes down the graph as much as possible.
      This avoids redundant computations on replicated tensors.

    - Remove nodes that do not have any users.

    - Pulling of `sum_vmapped` nodes up the graph as much as possible.
      This avoids redundant computations on summed tensors.

    Args:
        mod: A graph module whose computation graph will be simplified.
        push_replicate: Whether to push `replicate` nodes down the graph.
            Default: `True`.
        remove_unused: Whether to remove unused nodes from the graph. Default: `True`.
        pull_sum_vmapped: Whether to pull `sum_vmapped` nodes up the graph.
            Default: `True`.
        verbose: Whether to print debug information. Default: `False`.
        test_x: Input tensor to the module that will be verified after each
            simplification to make sure it does not change the correctness.
            This is expensive and should be considered for debugging purposes only.
            If `None`, the verification step will be skipped. Default: `None`.

    Returns:
        The simplified graph module.
    """
    if verbose:
        print(f"Traced graph before simplification:\n{mod.graph}")

    with check_unaltered(mod, test_x):
        # this assumes no side effects in the graph, so that it does not matter
        # when we call get_attr on a module's tensor constant
        remove_duplicate_get_attrs(mod.graph, verbose=verbose)

    if push_replicate:
        rewriter = RewriteReplicate(mod, verbose=verbose)
        while pattern := rewriter.find_pattern():
            with check_unaltered(mod, test_x):
                rewriter.replace_pattern(pattern)

        with check_unaltered(mod, test_x):
            rewriter.fuse_replicates_with_einsum()

    if remove_unused:
        rewriter = RewriteReplicate(mod, verbose=verbose)
        with check_unaltered(mod, test_x):
            rewriter.remove_unused_nodes()

    if pull_sum_vmapped:
        rewriter = RewriteSumVmapped(mod, verbose=verbose)
        while pattern := rewriter.find_pattern():
            with check_unaltered(mod, test_x):
                rewriter.replace_pattern(pattern)

            with check_unaltered(mod, test_x):
                rewriter.raise_sum_vmapped_outside_einsum()

        with check_unaltered(mod, test_x):
            rewriter.fuse_vmapped_sum_with_tensor_constants()

    mod.graph.lint()
    mod.recompile()

    if verbose:
        print(f"Traced graph after simplification:\n{mod.graph}")

    return mod
