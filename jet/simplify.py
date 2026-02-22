"""Functions to simplify a compute graph captured with `torch.fx`."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Generator

from torch import Tensor
from torch.fx import Graph, GraphModule
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn import Module

from jet.rules import (
    PullSumAddMM,
    PullSumAddOrSub,
    PullSumMM,
    PullSumMul,
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


@dataclass
class SimplificationStep:
    """Metadata about a single simplification step.

    Attributes:
        strategy: Name of the strategy that fired (e.g. ``"pull_sum"``).
        rule: Name of the rule that fired, or ``None`` for strategies that are
            not rule-based (e.g. ``"remove_unused"``).
        node: Name of the node the rule was applied to, or ``None``.
        step: 1-based index of this step in the simplification sequence.
        mod: Live reference to the :class:`GraphModule` **after** this step.
    """

    strategy: str
    rule: str | None
    node: str | None
    step: int
    mod: GraphModule


def _apply_next(
    rules: list[Rule], mod: GraphModule
) -> tuple[bool, str | None, str | None]:
    """Apply ONE rule to the first matching node.

    Unlike :func:`apply_all` which applies rules to every matching node in a
    single pass, this function returns as soon as the first rule fires on the
    first matching node.

    Args:
        rules: A list of rules to try.
        mod: The module to which the rule will be applied.

    Returns:
        A tuple ``(applied, rule_name, node_name)`` where *applied* is whether
        a rule fired, *rule_name* is the class name of the rule (or ``None``),
        and *node_name* is the name of the node it was applied to (or ``None``).
    """
    for node in list(mod.graph.nodes):
        for rule in rules:
            if rule.match(node):
                rule_name = rule.__class__.__name__
                node_name = node.name
                rule.apply(node, mod.graph)
                return True, rule_name, node_name
    return False, None, None


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

    Note:
        If you suspect that sum nodes are not being fully collapsed (e.g. because
        a rule for a particular operation is missing), use
        ``utils.visualize_graph(custom=True)`` to inspect the graph and look at
        the remaining ``sum`` nodes to understand which simplification is missing.
    """
    mod = capture_graph(mod, mock_x)

    nodes_before = len(list(mod.graph.nodes))
    if verbose:
        print(f"Traced graph before simplification:\n{mod.graph}")

    for step in simplify_iter(
        mod,
        mock_x,
        remove_unused=remove_unused,
        pull_sum=pull_sum,
        eliminate_common_subexpressions=eliminate_common_subexpressions,
        test_x=test_x,
    ):
        if verbose:
            msg = f"Applying strategy {step.strategy}: True"
            if step.rule:
                msg += f" (rule {step.rule} on {step.node})"
            print(msg)

    if verbose:
        print(f"Traced graph after simplification:\n{mod.graph}")
        print(f"Number of nodes before simplification: {nodes_before}.")
        nodes_after = len(list(mod.graph.nodes))
        print(f"Number of nodes after simplification: {nodes_after}.")

    return mod


def simplify_iter(
    mod: GraphModule | Module | Callable,
    mock_x: Tensor,
    remove_unused: bool = True,
    pull_sum: bool = True,
    eliminate_common_subexpressions: bool = True,
    test_x: Tensor | None = None,
) -> Generator[SimplificationStep, None, None]:
    """Yield one :class:`SimplificationStep` after each individual rule or strategy fires.

    This is the iterator counterpart of :func:`simplify`. The yielded ``mod``
    is a **live reference** — the same object is mutated in-place at each step —
    so callers that need a snapshot should deep-copy it.

    Args:
        mod: A (graph) module or function whose computation graph will be simplified.
        mock_x: A mock input tensor for tracing with ``make_fx``.
        remove_unused: Whether to remove unused nodes. Default: ``True``.
        pull_sum: Whether to pull ``sum`` nodes up the graph. Default: ``True``.
        eliminate_common_subexpressions: Whether to eliminate common
            subexpressions. Default: ``True``.
        test_x: Input tensor verified after each simplification to ensure
            correctness. Expensive — for debugging only. Default: ``None``.

    Yields:
        :class:`SimplificationStep` after every individual simplification.
    """
    if not isinstance(mod, GraphModule):
        mod = capture_graph(mod, mock_x)

    sum_rules = [
        PullSumSqueeze(),
        PullSumUnsqueeze(),
        PullSumView(),
        PullSumAddOrSub(),
        PullSumMul(),
        PullSumMM(),
        PullSumAddMM(),
    ]

    graph = mod.graph
    step_idx = 0

    # Build ordered strategy list.  Each entry is
    # (name, callable_returning_(applied, rule_name, node_name))
    strategies: list[tuple[str, Callable[[], tuple[bool, str | None, str | None]]]] = (
        []
    )
    if remove_unused:
        strategies.append(
            ("remove_unused", lambda: (graph.eliminate_dead_code(), None, None))
        )
    if pull_sum:
        strategies.append(("pull_sum", lambda: _apply_next(sum_rules, mod)))
    if eliminate_common_subexpressions:
        strategies.append(
            (
                "common_subexpression_elimination",
                lambda: (
                    common_subexpression_elimination(mod.graph),
                    None,
                    None,
                ),
            )
        )

    if not strategies:
        return

    do_simplify = True
    while do_simplify:
        fired = False
        for name, apply_strategy in strategies:
            with check_unaltered(mod, test_x):
                applied, rule_name, node_name = apply_strategy()

            if applied:
                mod.recompile()
                ShapeProp(mod).propagate(mock_x)
                step_idx += 1
                yield SimplificationStep(
                    strategy=name,
                    rule=rule_name,
                    node=node_name,
                    step=step_idx,
                    mod=mod,
                )
                fired = True
                break

        do_simplify = fired

    mod.graph.lint()
    mod.recompile()
