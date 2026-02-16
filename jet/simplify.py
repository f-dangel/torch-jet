"""Functions to simplify a compute graph captured with `torch.fx`."""

from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import product
from typing import Callable

from torch import Tensor, manual_seed
from torch.fx import Graph, GraphModule
from torch.nn import Module
from torch.random import fork_rng

from jet.rules import (
    MergeSumVmappedConstant,
    ModuleRule,
    PullSumVmappedLinear,
    PullSumVmappedReplicateMultiplication,
    PullSumVmappedScalarMultiplication,
    PullSumVmappedTensorAddition,
    PushReplicateElementwise,
    PushReplicateLinear,
    PushReplicateScalarArithmetic,
    PushReplicateSumVmapped,
    PushReplicateTensorArithmetic,
    Rule,
)
from jet.tracing import capture_graph
from jet.utils import (
    print_tensor_constants_and_shapes,
    recursive_getattr,
)


def common_subexpression_elimination(graph: Graph, verbose: bool = False) -> bool:
    """Replace duplicate subexpressions with a single node.

    Args:
        graph: The graph to be optimized.
        verbose: Whether to print debug information. Default: `False`.

    Returns:
        Whether a subexpression was replaced.
    """
    nodes = {}

    replaced = False
    num_replacements = 0

    for node in list(graph.nodes):
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


def common_tensor_constant_elimination(  # noqa: C901, PLR0912
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
    same: dict[str, list[str]] = {}

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
        elif verbose:
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
                print(f"Applying rule {rule.__class__.__name__} to {node=}.")

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
    mod: GraphModule,
    x: Tensor | None,
    seed: int = 0,
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    """Verify that the module still produces the same output before and after the body.

    Args:
        mod: The module to be checked.
        x: Input tensor to the module. If `None`, the check will be skipped.
        seed: Random seed to use for reproducibility. Default: `0`.
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
        with fork_rng():
            manual_seed(seed)
            out_before = mod(x)
        yield

        try:
            mod.graph.lint()
            mod.recompile()
            with fork_rng():
                manual_seed(seed)
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
    test_x: Tensor | None = None,
    example_input: Tensor | None = None,
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
        example_input: A concrete example input tensor for tracing with ``make_fx``.
            Required when ``mod`` is not already a ``GraphModule``. Default: `None`.

    Returns:
        The simplified graph module.
    """
    mod = capture_graph(mod, example_input=example_input)

    nodes_before = len(list(mod.graph.nodes))
    if verbose:
        print(f"Traced graph before simplification:\n{mod.graph}")

    graph = mod.graph

    # Initialize PushReplicate* rules
    replicate_rules = [
        PushReplicateElementwise(),
        PushReplicateScalarArithmetic(),
        PushReplicateTensorArithmetic(),
        PushReplicateLinear(),
        PushReplicateSumVmapped(),
    ]
    # Initialize PullSumVmapped* rules
    sum_rules = [
        PullSumVmappedTensorAddition(),
        PullSumVmappedScalarMultiplication(),
        PullSumVmappedReplicateMultiplication(),
        PullSumVmappedLinear(),
        MergeSumVmappedConstant(),
    ]

    strategies = {
        "remove_unused": graph.eliminate_dead_code,
        "common_subexpression_elimination": partial(
            common_subexpression_elimination, mod.graph, verbose=verbose
        ),
        "eliminate_tensor_constants": partial(
            common_tensor_constant_elimination, mod, verbose=verbose
        ),
        "push_replicate": lambda: apply_once(replicate_rules, mod, verbose=verbose),
        "pull_sum_vmapped": lambda: apply_once(sum_rules, mod, verbose=verbose),
    }

    # round 1 of simplifications: remove redundancies in the graph
    round_one = []
    if remove_unused:
        round_one.append("remove_unused")
    _exhaust_incrementally({s: strategies[s] for s in round_one}, mod, test_x, verbose)

    # round 2 of simplifications: push forward replicate nodes
    round_two = []
    if push_replicate:
        round_two.append("push_replicate")
    _exhaust_incrementally({s: strategies[s] for s in round_two}, mod, test_x, verbose)

    # round 3 of simplifications: pull sum_vmapped nodes up
    round_three = []
    if pull_sum_vmapped:
        round_three.append("pull_sum_vmapped")
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
    strategies: dict[str, Callable[[], None]],
    mod: GraphModule,
    test_x: Tensor | None,
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
