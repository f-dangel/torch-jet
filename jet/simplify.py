"""Functions to simplify a compute graph captured with `torch.fx`."""

import operator
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from torch import Tensor, add, cos, cosh, div, mul
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
        self.last_pattern_pos = 0

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

    def rewrite_pattern(self) -> bool:
        """Try to find the next replicate node and rewrite it.

        Returns:
            Whether a pattern was found and rewritten.
        """
        rewritten = False

        if pattern := self.find_pattern():
            self.replace_pattern(pattern)
            rewritten = True

        return rewritten

    def find_pattern(self) -> Optional[Tuple[List[Node], Node]]:  # noqa: C901
        """Find a pattern that can be simplified.

        Returns:
            A pattern that can be simplified, or `None` if no such pattern is found.
            A pattern consists of two parts: a list of nodes that can be swapped with
            the node returned as second part.
        """
        for pos, node in enumerate(
            list(self.graph.nodes)[self.last_pattern_pos :], start=self.last_pattern_pos
        ):
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

            if pattern is not None:
                self.maybe_print(f"Can swap {pattern[0]} and {pattern[1]}")
                self.last_pattern_pos = max(0, pos - 1)
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

    def remove_unused_nodes(self) -> bool:
        """Find and remove unused nodes from the graph.

        Returns:
            Whether nodes were removed.
        """
        removed = [self.maybe_erase(node) for node in list(self.graph.nodes)]
        return any(removed)


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

    def rewrite_pattern(self) -> bool:
        """Try to find the next sum_vmapped node and rewrite it.

        Returns:
            Whether a pattern was found and rewritten.
        """
        rewritten = False
        if pattern := self.find_pattern():
            self.replace_pattern(pattern)
            rewritten = True
        return rewritten

    def find_pattern(self) -> Optional[Tuple[Node, Node]]:
        """Find a pattern that can be simplified.

        Returns:
            A pattern that can be simplified, or `None` if no such pattern is found.
            A pattern consists of two parts: the `sum_vmapped` node that can be
            propagated up through the node returned as second part.
        """
        for node in self.graph.nodes:
            if not self.is_sum_vmapped(node):
                continue
            (op,) = self.parents(node)
            if op.op != "call_function":
                continue

            # Can only replace nodes that are not consumed by others
            if self.children(op) != [node]:
                continue

            pattern = None
            parents = self.parents(op)

            # operations that produce a tensor from a single tensor `x`, which is then
            # `sum_vmapped`
            if (
                op.target
                in {
                    operator.mul,  # sum_vmapped(x * 2) -> 2 * sum_vmapped(x)
                    operator.truediv,  # sum_vmapped(x / 2) -> sum_vmapped(x) / 2
                }
                and len(parents) == 1
            ):
                pattern = [node, op]

            # operations that produce a tensor from two tensors `x`, `y`, which is then
            # `sum_vmapped`
            # NOTE This assumes there is no broadcasting (x.shape == y.shape)!
            elif (
                op.target
                in {
                    # sum_vmapped(x + y) -> sum_vmapped(x) + sum_vmapped(y)
                    operator.add,
                    # sum_vmapped(x - y) -> sum_vmapped(x) - sum_vmapped(y)
                    operator.sub,
                }
                and len(parents) == 2
            ):
                pattern = [node, op]
            # sum_vmapped(linear(x, W, b)) -> linear(sum_vmapped(x), W, b)
            elif op.target == linear:
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

        # generate new `sum_vmapped` nodes above `op` and rewire the arguments
        parents = [op.args[0]] if op.target == linear else self.parents(op)
        for parent in parents:
            with self.graph.inserting_before(op):
                new_sum = self.graph.call_function(sum_vmapped, args=(parent,))
            op.replace_input_with(parent, new_sum)
            self.maybe_erase(parent)

    def simplify_vmapped_sum_with_replicate(self) -> bool:
        """Simplify nodes that use `sum_vmapped` and `replicate` nodes.

        For instance, `sum_vmapped(x * replicate(y))` can be simplified into
        `sum_vmapped(x) * y`.

        Returns:
            Whether a `sum_vmapped` was fused with a `replicate`.
        """
        simplified = False
        for node in list(self.graph.nodes):
            if not self.is_sum_vmapped(node):
                continue
            (op,) = self.parents(node)
            if op.op not in {"call_function", "call_method"}:
                continue

            parents = self.parents(op)
            if not any(self.is_replicate(p) for p in parents):
                continue

            replicates = [p for p in parents if self.is_replicate(p)]
            non_replicates = [p for p in parents if not self.is_replicate(p)]

            if (  # sum_vmapped(x * replicate(y)) -> sum_vmapped(x) * y
                op.target in {mul, operator.mul} and len(parents) == 2
            ):
                print("Found sum_vmapped with replicate: ", node, op)
                # need to replace replicate(y) with y
                (replicate,) = replicates
                (replicated,) = self.parents(replicate)
                op.replace_input_with(replicate, replicated)
                self.maybe_erase(replicate)

                # need to replace x with sum_vmapped(x)
                (other_node,) = [p for p in parents if not self.is_replicate(p)]
                with self.graph.inserting_after(other_node):
                    new_sum = self.graph.call_function(sum_vmapped, args=(other_node,))
                op.replace_input_with(other_node, new_sum)
                self.maybe_erase(other_node)

                # replace the sum_vmapped node with the operation node
                node.replace_all_uses_with(op)
                self.maybe_erase(node)

                simplified = True

            return simplified

    def fuse_vmapped_sum_with_tensor_constants(self) -> bool:
        """Fuse tensor constants with `vmapped_sum` nodes.

        For instance, vmapped_sum(mod._tensor_constant0) can be simplified into
        mod._tensor_constant0 = vmapped_sum(mod._tensor_constant0)

        Returns:
            Whether a `sum_vmapped` was fused with a tensor constant.
        """
        # create a mapping which tensor constants are fetched by which nodes, and how
        # these nodes are used
        attributes = defaultdict(dict)
        for n in self.graph.nodes:
            if n.op == "get_attr":
                attributes[n.target][n] = self.children(n)

        fused = False

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
                fused = True

            # remove the get_attr nodes that are not referenced any more
            for get_attr in get_attr_to_children:
                self.maybe_erase(get_attr)

        return fused


def common_subexpression_elimination(
    graph: Graph, verbose: bool = False, restrict_ops: Optional[Set[str]] = None
) -> bool:
    """Replace duplicate subexpressions with a single node.

    Args:
        graph: The graph to be optimized.
        verbose: Whether to print debug information. Default: `False`.
        restrict_ops: A set of operations to restrict the optimization to.
            Default: `None`.

    Returns:
        Whether a subexpression was replaced.
    """
    nodes = {}

    replaced = False
    num_replacements = 0

    for node in list(graph.nodes):
        if restrict_ops is not None and node.op not in restrict_ops:
            continue

        node_hash = (node.op, node.target, node.args, node.kwargs)
        if node_hash in nodes:
            # replace the node
            replacement = nodes[node_hash]
            if verbose:
                print(
                    f"Replacing {node}"
                    + f" ({node.op}, {node.target}, {node.args}, {node.kwargs})"
                )
                print(
                    f"with {replacement} ({replacement.op}, {replacement.target},"
                    + f" {replacement.args}, {replacement.kwargs})"
                )
            node.replace_all_uses_with(replacement)

            # remove the node from the graph
            if verbose:
                print(f"Erasing {node}.")
            graph.erase_node(node)

            replaced = True
            num_replacements += 1
        else:
            nodes[node_hash] = node

    if verbose:
        print(f"Replacements: {num_replacements}")

    return replaced


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
        Exception: If the module cannot be compiled or executed anymore.
    """
    if x is not None:
        before_str = str(mod.graph)
        out_before = mod(x)
        yield

        try:
            mod.graph.lint()
            mod.recompile()
            out_after = mod(x)
            close = out_before.allclose(out_after, rtol=rtol, atol=atol)

            if not close:
                print(f"Before:\n{before_str}")
                print(f"After:\n{mod.graph}")
                raise RuntimeError("Module output changed.")
        except Exception as e:
            print(f"Before:\n{before_str}")
            print(f"After:\n{mod.graph}")
            print("Module cannot be compiled or executed anymore.")
            raise e

    else:
        yield


def simplify(  # noqa: C901
    mod: GraphModule,
    push_replicate: bool = True,
    remove_unused: bool = True,
    pull_sum_vmapped: bool = True,
    eliminate_common_subexpressions: bool = True,
    verbose: bool = False,
    test_x: Optional[Tensor] = None,
) -> GraphModule:
    """Simplify a compute graph.

    At the moment, the following simplifications are implemented:

    - Pushing of `replicate` nodes down the graph as much as possible.
      This avoids redundant computations on replicated tensors.

    - Remove nodes that do not have any users.

    - Common subexpression elimination (CSE) to remove duplicate computations.

    - Pulling of `sum_vmapped` nodes up the graph as much as possible.
      This avoids redundant computations on summed tensors.

    Args:
        mod: A graph module whose computation graph will be simplified.
        push_replicate: Whether to push `replicate` nodes down the graph.
            Default: `True`.
        remove_unused: Whether to remove unused nodes from the graph. Default: `True`.
        pull_sum_vmapped: Whether to pull `sum_vmapped` nodes up the graph.
            Default: `True`.
        eliminate_common_subexpressions: Whether to eliminate common subexpressions.
            Default: `True`.
        verbose: Whether to print debug information. Default: `False`.
        test_x: Input tensor to the module that will be verified after each
            simplification to make sure it does not change the correctness.
            This is expensive and should be considered for debugging purposes only.
            If `None`, the verification step will be skipped. Default: `None`.

    Returns:
        The simplified graph module.
    """
    nodes_before = len(list(mod.graph.nodes))
    if verbose:
        print(f"Traced graph before simplification:\n{mod.graph}")

    replicate_rewriter = RewriteReplicate(mod, verbose=verbose)
    sum_vmapped_rewriter = RewriteSumVmapped(mod, verbose=verbose)

    strategies = {
        "remove_unused": replicate_rewriter.remove_unused_nodes,
        "common_subexpression_elimination_get_attr": partial(
            common_subexpression_elimination,
            mod.graph,
            verbose=verbose,
            restrict_ops={"get_attr"},
        ),
        "common_subexpression_elimination": partial(
            common_subexpression_elimination, mod.graph, verbose=verbose
        ),
        "push_replicate": replicate_rewriter.rewrite_pattern,
        "pull_sum_vmapped": sum_vmapped_rewriter.rewrite_pattern,
        "simplify_vmapped_sum_with_replicate": sum_vmapped_rewriter.simplify_vmapped_sum_with_replicate,  # noqa: B950
        "fuse_with_tensor_constant": sum_vmapped_rewriter.fuse_vmapped_sum_with_tensor_constants,  # noqa: B950
    }

    # round 1 of simplifications: remove redundancies in the graph
    round_one = []
    if remove_unused:
        round_one.append("remove_unused")
    if eliminate_common_subexpressions:
        round_one.append("common_subexpression_elimination_get_attr")
    _exhaust_incrementally({s: strategies[s] for s in round_one}, mod, test_x, verbose)

    # round 2 of simplifications: push forward replicate nodes
    round_two = []
    if push_replicate:
        round_two.append("push_replicate")
    _exhaust_incrementally({s: strategies[s] for s in round_two}, mod, test_x, verbose)

    # round 3 of simplifications: pull sum_vmapped nodes up
    round_three = []
    if pull_sum_vmapped:
        round_three.extend(
            [
                "pull_sum_vmapped",
                "simplify_vmapped_sum_with_replicate",
                "fuse_with_tensor_constant",
            ]
        )
    _exhaust_incrementally(
        {s: strategies[s] for s in round_three}, mod, test_x, verbose
    )

    # round 4 of simplifications: remove redundancies in the graph and clean up
    round_four = []
    if eliminate_common_subexpressions:
        round_four.append("common_subexpression_elimination")
    if remove_unused:
        round_four.append("remove_unused")
    _exhaust_incrementally({s: strategies[s] for s in round_four}, mod, test_x, verbose)

    mod.graph.lint()
    mod.recompile()

    if verbose:
        print(f"Traced graph after simplification:\n{mod.graph}")

    if verbose:
        print(f"Number of nodes before simplification: {nodes_before}.")
        nodes_after = len(list(mod.graph.nodes))
        print(f"Number of nodes after simplification: {nodes_after}.")

    return mod


def _exhaust_incrementally(
    strategies: Dict[str, Callable[[], None]],
    mod: GraphModule,
    test_x: Optional[Tensor],
    verbose: bool,
):
    """Apply one round of simplifications.

    Loop through the simplification strategies until one is successful, then start
    from the beginning until we complete one round where none of the strategies is
    successful.

    Args:
        strategies: A dictionary of strategies to be applied.
        mod: The module to be simplified.
        test_x: Input tensor to the module that will be verified after each
            simplification to make sure it does not change the correctness.
            This is expensive and should be considered for debugging purposes only.
            If `None`, the verification step will be skipped. Default: `None`.
        verbose: Whether to print debug information. Default: `False`.
    """
    if not strategies:
        return

    do_simplify = True
    while do_simplify:
        simplified = False
        for name, apply_strategy in strategies.items():
            with check_unaltered(mod, test_x):
                simplified = apply_strategy()
                if verbose:
                    print(f"Applying strategy {name}: {simplified}")

            if simplified:
                break

        do_simplify = simplified
