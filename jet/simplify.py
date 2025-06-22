"""Functions to simplify a compute graph captured with `torch.fx`."""

import operator
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import product
from typing import Any, Callable, Dict, Optional, Set, Tuple

from torch import Tensor, mul
from torch.fx import Graph, GraphModule, Node
from torch.nn import Module
from torch.nn.functional import linear

from jet import JetTracer
from jet.rules import (
    ModuleRule,
    PushReplicateElementwise,
    PushReplicateLinear,
    PushReplicateScalarArithmetic,
    PushReplicateSumVmapped,
    PushReplicateTensorArithmetic,
    Rule,
)
from jet.utils import (
    WrapperModule,
    print_tensor_constants_and_shapes,
    recursive_getattr,
    replicate,
    standardize_signature,
    sum_vmapped,
)


class RewriteSumVmapped:
    """Propagates summations over a vmaped axis up a computation graph."""

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

    def maybe_print(self, message: str):
        """Print a message if verbose mode is enabled.

        Args:
            message: The message to print.
        """
        if self.verbose:
            print(message)

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
            (op,) = node.all_input_nodes
            if op.op != "call_function":
                continue

            # Can only replace nodes that are not consumed by others
            children = list(op.users.keys())
            if children != [node]:
                continue

            pattern = None
            parents = op.all_input_nodes

            # operations that produce a tensor from a single tensor `x`, which is then
            # `sum_vmapped`
            if (
                op.target
                in {
                    operator.mul,  # sum_vmapped(x * 2) -> 2 * sum_vmapped(x)
                    operator.truediv,  # sum_vmapped(x / 2) -> sum_vmapped(x) / 2
                    # sum_vmapped(replicate(x, times, pos1), pos2)
                    # (pos1 != pos2) -> replicate(sum_vmapped(x, pos2'), times, pos1')
                    # (pos1 == pos2) -> times * x
                    replicate,
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

        if not self.is_replicate(op):
            sum_node.replace_all_uses_with(op)

            # generate new `sum_vmapped` nodes above `op` and rewire the arguments
            parents = [op.args[0]] if op.target == linear else op.all_input_nodes
            for parent in parents:
                with self.graph.inserting_before(op):
                    new_sum = self.graph.call_function(
                        sum_vmapped,
                        args=(parent,),
                        kwargs={"pos": sum_node.kwargs["pos"]},
                    )
                op.replace_input_with(parent, new_sum)

        self.graph.eliminate_dead_code()

    def simplify_vmapped_sum_with_replicate(self) -> bool:
        """Simplify nodes that use `sum_vmapped` and `replicate` nodes.

        For instance, `sum_vmapped(x * replicate(y, times, pos=pos), pos=pos)` can be
        simplified into `sum_vmapped(x, pos=pos) * y`.

        Returns:
            Whether a `sum_vmapped` was fused with a `replicate`.
        """
        simplified = False
        for node in [n for n in self.graph.nodes if self.is_sum_vmapped(n)]:
            (op,) = node.all_input_nodes
            if op.op not in {"call_function", "call_method"}:
                continue

            parents = op.all_input_nodes
            if not any(self.is_replicate(p) for p in parents):
                continue

            replicates = [p for p in parents if self.is_replicate(p)]
            pos = node.kwargs["pos"]

            if (  # sum_vmapped(x * replicate(y)) -> sum_vmapped(x) * y
                op.target in {mul, operator.mul}
                and len(parents) == 2
                and len(replicates) == 1
                and replicates[0].kwargs["pos"] == pos
            ):
                # need to replace replicate(y) with y
                (replicate,) = replicates
                (replicated,) = replicate.all_input_nodes
                op.replace_input_with(replicate, replicated)

                # need to replace x with sum_vmapped(x)
                (other_node,) = [p for p in parents if not self.is_replicate(p)]
                with self.graph.inserting_after(other_node):
                    new_sum = self.graph.call_function(
                        sum_vmapped, args=(other_node,), kwargs={"pos": pos}
                    )
                op.replace_input_with(other_node, new_sum)

                # replace the sum_vmapped node with the operation node
                node.replace_all_uses_with(op)

                simplified = True

        if simplified:
            self.graph.eliminate_dead_code()

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
                children = list(n.users.keys())
                attributes[n.target][n] = children

        fused = False

        for target, get_attr_to_children in attributes.items():
            all_children = set(sum(get_attr_to_children.values(), start=[]))

            # if all children of a get_attr are sum_vmappeds, we can sum first
            if all(self.is_sum_vmapped(c) for c in all_children):
                replacement = list(get_attr_to_children.keys())[0]
                for sum_node in all_children:
                    if sum_node != replacement:
                        sum_node.replace_all_uses_with(replacement)

                # sum the tensor constant
                old = getattr(self.mod, target)
                new = sum_vmapped(old)
                setattr(self.mod, target, new)
                self.maybe_print(
                    f"Collapsing {target}: {tuple(old.shape)} -> {tuple(new.shape)}."
                )
                fused = True

        if fused:
            self.graph.eliminate_dead_code()

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
                    + f" ({node.op}, {node.target}, {node.args}, {node.kwargs})\nwith"
                    + f" {replacement} ({replacement.op}, {replacement.target},"
                    + f" {replacement.args}, {replacement.kwargs})"
                )
            node.replace_all_uses_with(replacement)

            replaced = True
            num_replacements += 1
        else:
            nodes[node_hash] = node

    if replaced:
        graph.eliminate_dead_code()

    if verbose:
        print(f"Replacements: {num_replacements}")

    return replaced


def common_tensor_constant_elimination(  # noqa: C901
    mod: GraphModule, verbose: bool = False
) -> bool:
    """Eliminate duplicate tensor constants in a GraphModule by shape and value.

    If two or more tensor constants have the same shape and values, all but one are
    removed and their uses are redirected to the remaining one, saving memory.

    Args:
        mod: The GraphModule to optimize.
        verbose: Whether to print debug information. Default: `False`.

    Returns:
        True if any tensor constants were eliminated, False otherwise.
    """
    if verbose:
        print("Tensor constants and shapes before elimination:")
        print_tensor_constants_and_shapes(mod)

    # Gather all get_attr nodes for tensor constants
    nodes = [
        node
        for node in mod.graph.nodes
        if node.op == "get_attr" and "_tensor_constant" in node.target
    ]

    # Figure out which tensor constants are fetched by which nodes
    constants_to_nodes = defaultdict(list)
    for node in nodes:
        constants_to_nodes[node.target].append(node)

    # Figure out which tensor constants are identical
    def _same(tensor1: Tensor, tensor2: Tensor) -> bool:
        if (
            tensor1.shape != tensor2.shape
            or tensor1.dtype != tensor2.dtype
            or tensor1.device != tensor2.device
        ):
            return False
        return tensor1.allclose(tensor2)

    # Figure out which tensors are the same
    same: Dict[str, list[str]] = {}

    for node in nodes:
        ref = recursive_getattr(mod, node.target)
        matched = False

        for const in same:
            if _same(ref, recursive_getattr(mod, const)):
                same[const].append(node.target)
                matched = True
                break

        if not matched:
            same[node.target] = []

    # Replace the nodes that access the same tensor constant
    replaced = False
    for ref, others in same.items():
        ref_node = constants_to_nodes[ref][0]

        duplicate_nodes = constants_to_nodes[ref][1:]
        for other in others:
            duplicate_nodes.extend(constants_to_nodes[other])

        if duplicate_nodes:
            # replace the nodes
            if verbose:
                print(f"Replacing {duplicate_nodes} with {ref_node}.")
            for node in duplicate_nodes:
                node.replace_all_uses_with(ref_node)
                mod.graph.erase_node(node)

            # delete the tensors
            if verbose:
                print(f"Deleting {others} module attributes.")
            for other in others:
                delattr(mod, other)
            replaced = True
        else:
            if verbose:
                print(f"{ref_node} has no duplicates.")

    if replaced and verbose:
        print("Tensor constants and shapes after elimination:")
        print_tensor_constants_and_shapes(mod)

    return replaced


def apply_once(
    rules: list[Rule | ModuleRule], mod: GraphModule, verbose: bool = False
) -> bool:
    """Apply one of the supplied rules once to a module.

    Args:
        rules: A list of rules to be applied.
        mod: The module to which the rules will be applied.
        verbose: Whether to print debug information. Default: `False`.

    Returns:
        True if any rule was applied, False otherwise.

    Raises:
        TypeError: If a rule is not an instance of `Rule` or `ModuleRule`.
    """
    for node, rule in product(mod.graph.nodes, rules):
        if rule.match(node):
            if verbose:
                print(f"Applying rule {rule} to {node=}.")

            if isinstance(rule, Rule):
                rule.apply(node, mod.graph)
            elif isinstance(rule, ModuleRule):
                rule.apply(node, mod)
            else:
                raise TypeError(f"Unknown rule type: {type(rule)}.")
            return True

    return False


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
            if isinstance(out_before, tuple) and isinstance(out_after, tuple):
                # If both outputs are tuples, compare each element
                close = len(out_before) == len(out_after) and all(
                    a.allclose(b, rtol=rtol, atol=atol)
                    for a, b in zip(out_before, out_after)
                )
            elif isinstance(out_before, Tensor) and isinstance(out_after, Tensor):
                close = out_before.allclose(out_after, rtol=rtol, atol=atol)
            else:
                close = False

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
    mod: GraphModule | Module | Callable,
    push_replicate: bool = True,
    remove_unused: bool = True,
    pull_sum_vmapped: bool = True,
    eliminate_common_subexpressions: bool = True,
    eliminate_tensor_constants: bool = True,
    verbose: bool = False,
    test_x: Optional[Tensor] = None,
) -> GraphModule:
    """Simplify a compute graph.

    At the moment, the following simplifications are implemented:

    - Pushing of `replicate` nodes down the graph as much as possible.
      This avoids redundant computations on replicated tensors.

    - Remove nodes that do not have any users.

    - Common subexpression elimination (CSE) to remove duplicate computations.

    - Eliminating tensor constants which contain the same tensors.

    - Pulling of `sum_vmapped` nodes up the graph as much as possible.
      This avoids redundant computations on summed tensors.

    Args:
        mod: A (graph) module or function whose computation graph will be simplified.
        push_replicate: Whether to push `replicate` nodes down the graph.
            Default: `True`.
        remove_unused: Whether to remove unused nodes from the graph. Default: `True`.
        pull_sum_vmapped: Whether to pull `sum_vmapped` nodes up the graph.
            Default: `True`.
        eliminate_common_subexpressions: Whether to eliminate common subexpressions.
            Default: `True`.
        eliminate_tensor_constants: Whether to eliminate tensor constants.
            Default: `True`.
        verbose: Whether to print debug information. Default: `False`.
        test_x: Input tensor to the module that will be verified after each
            simplification to make sure it does not change the correctness.
            This is expensive and should be considered for debugging purposes only.
            If `None`, the verification step will be skipped. Default: `None`.

    Returns:
        The simplified graph module.
    """
    if not isinstance(mod, GraphModule):
        mod = mod if isinstance(mod, Module) else WrapperModule(mod)
        graph = JetTracer().trace(mod)
        mod = GraphModule(mod, graph)

    nodes_before = len(list(mod.graph.nodes))
    if verbose:
        print(f"Traced graph before simplification:\n{mod.graph}")

    # Replace all call_method[mul] with call_function[operator.mul] because the
    # simplification logic is only supported for call_function nodes at the moment
    graph = mod.graph
    for node in [n for n in graph.nodes if n.op == "call_method" and n.target == "mul"]:
        with check_unaltered(mod, test_x), graph.inserting_before(node):
            # replace the node with a call_function node
            new_node = graph.call_function(
                operator.mul, args=node.args, kwargs=node.kwargs
            )
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)

    # Unify the args/kwargs of replicate and sum_vmapped nodes
    for node in [
        n
        for n in graph.nodes
        if n.op == "call_function" and n.target in {replicate, sum_vmapped}
    ]:
        with check_unaltered(mod, test_x):
            standardize_signature(node, verbose=verbose)

    # Initialize PushReplicate* rules
    replicate_rules = [
        PushReplicateElementwise(),
        PushReplicateScalarArithmetic(),
        PushReplicateTensorArithmetic(),
        PushReplicateLinear(),
        PushReplicateSumVmapped(),
    ]

    strategies = {
        "remove_unused": graph.eliminate_dead_code,
        "common_subexpression_elimination_get_attr": partial(
            common_subexpression_elimination,
            mod.graph,
            verbose=verbose,
            restrict_ops={"get_attr"},
        ),
        "common_subexpression_elimination": partial(
            common_subexpression_elimination, mod.graph, verbose=verbose
        ),
        "eliminate_tensor_constants": partial(
            common_tensor_constant_elimination, mod, verbose=verbose
        ),
        "push_replicate": lambda: apply_once(replicate_rules, mod, verbose=verbose),
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
    if eliminate_common_subexpressions:
        round_three.append("common_subexpression_elimination")
    _exhaust_incrementally(
        {s: strategies[s] for s in round_three}, mod, test_x, verbose
    )

    # round 4 of simplifications: remove redundancies in the graph and clean up
    round_four = []
    if eliminate_tensor_constants:
        round_four.append("eliminate_tensor_constants")
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
