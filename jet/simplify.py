"""Functions to simplify a compute graph captured with `torch.fx`."""

from contextlib import contextmanager
from functools import partial
from typing import Callable

from torch import Tensor
from torch.fx import Graph, GraphModule
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn import Module

from jet.rules import (
    PullSumAddMM,
    PullSumAddition,
    PullSumMM,
    PullSumMultiplication,
    PullSumSqueeze,
    PullSumUnsqueeze,
    PullSumView,
    Rule,
)
from jet.tracing import capture_graph
from jet.utils import run_seeded


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


def apply_all(rules: list[Rule], mod: GraphModule, verbose: bool = False) -> bool:
    """Apply matching rules to all sum nodes in a single pass.

    Iterates over all nodes once, applying the first matching rule to each sum
    node. New sum nodes created by rule applications are handled in subsequent
    passes (by the caller).

    Args:
        rules: A list of rules to be applied.
        mod: The module to which the rules will be applied.
        verbose: Whether to print debug information. Default: `False`.

    Returns:
        True if any rule was applied, False otherwise.
    """
    applied = False
    for node in list(mod.graph.nodes):
        for rule in rules:
            if rule.match(node):
                if verbose:
                    print(f"Applying rule {rule.__class__.__name__} to {node=}.")
                rule.apply(node, mod.graph)
                applied = True
                break

    return applied


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
        out_before = run_seeded(mod, seed, x)
        yield

        try:
            mod.graph.lint()
            mod.recompile()
            out_after = run_seeded(mod, seed, x)
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


def simplify(
    mod: GraphModule | Module | Callable,
    mock_x: Tensor,
    remove_unused: bool = True,
    pull_sum: bool = True,
    eliminate_common_subexpressions: bool = True,
    verbose: bool = False,
    test_x: Tensor | None = None,
) -> GraphModule:
    """Simplify a compute graph.

    At the moment, the following simplifications are implemented:

    - Remove nodes that do not have any users.

    - Common subexpression elimination (CSE) to remove duplicate computations.

    - Pulling of ``sum`` nodes up the graph as much as possible.
      This avoids redundant computations on summed tensors.

    Args:
        mod: A (graph) module or function whose computation graph will be simplified.
        mock_x: A mock input tensor for tracing with ``make_fx``.
        remove_unused: Whether to remove unused nodes from the graph. Default: `True`.
        pull_sum: Whether to pull ``sum`` nodes up the graph.
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
    mod = capture_graph(mod, mock_x)

    nodes_before = len(list(mod.graph.nodes))
    if verbose:
        print(f"Traced graph before simplification:\n{mod.graph}")

    graph = mod.graph

    # Initialize PullSum* rules
    sum_rules = [
        PullSumSqueeze(),
        PullSumUnsqueeze(),
        PullSumView(),
        PullSumAddition(),
        PullSumMultiplication(),
        PullSumMM(),
        PullSumAddMM(),
    ]

    strategies = {}
    if remove_unused:
        strategies["remove_unused"] = graph.eliminate_dead_code
    if pull_sum:
        strategies["pull_sum"] = lambda: apply_all(sum_rules, mod, verbose=verbose)
    if eliminate_common_subexpressions:
        strategies["common_subexpression_elimination"] = partial(
            common_subexpression_elimination, mod.graph, verbose=verbose
        )
    _exhaust_incrementally(strategies, mod, mock_x, test_x, verbose)

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
    mock_x: Tensor,
    test_x: Tensor | None,
    verbose: bool,
):
    """Apply one round of simplifications.

    Loop through the simplification strategies until one is successful, then start
    from the beginning until we complete one round where none of the strategies is
    successful. After each successful application, shape metadata is repropagated
    so that newly created nodes have valid ``meta["tensor_meta"]``.

    Args:
        strategies: A dictionary of strategies to be applied.
        mod: The module to be simplified.
        mock_x: A mock input tensor used to repropagate shape metadata after each
            successful strategy application.
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
                mod.recompile()
                ShapeProp(mod).propagate(mock_x)
                break

        do_simplify = simplified
