"""Test simplification mechanism on compute graphs."""

import torch
from torch import Tensor, arange
from torch.fx import Graph, Node

from jet.simplify import common_subexpression_elimination
from jet.tracing import capture_graph
from test.utils import report_nonclose


def count_nodes(graph: Graph, predicate) -> int:
    """Count nodes in an FX graph that satisfy a predicate.

    Args:
        graph: The FX graph to inspect.
        predicate: Function that returns True for nodes to count.

    Returns:
        The number of matching nodes.
    """
    return len([n for n in graph.nodes if predicate(n)])


def get_output_args(graph: Graph) -> tuple[Node, ...]:
    """Get the direct parent nodes of the output node.

    For graphs returning a tuple (e.g., jet functions returning (F0, F1, F2)),
    this returns the individual tensor nodes that form the output tuple.

    Args:
        graph: The FX graph to inspect.

    Returns:
        Tuple of nodes that are the direct arguments of the output node.
    """
    output_node = next(n for n in graph.nodes if n.op == "output")
    args = output_node.args[0]
    if isinstance(args, (tuple, list)):
        return tuple(args)
    return (args,)


def test_common_subexpression_elimination():
    """Test common subexpression elimination."""

    def f(x: Tensor) -> Tensor:
        # NOTE that instead of computing y1, y2, we could simply compute y1 and
        # return y1 + y1
        x1 = x + 1
        x2 = x + 1
        y1 = 2 * x1
        y2 = 2 * x2
        z = y1 + y2
        return z

    x = arange(10)

    f_traced = capture_graph(f, x)
    f_x = f_traced(x)
    nodes_before = len(list(f_traced.graph.nodes))
    # make_fx produces ATen-level nodes; verify duplicate subexpressions exist
    assert nodes_before >= 7

    # Count specific duplicate ops before CSE
    _add_target = torch.ops.aten.add.Tensor
    adds_before = count_nodes(f_traced.graph, lambda n: n.target == _add_target)

    common_subexpression_elimination(f_traced.graph, verbose=True)
    nodes_after = len(list(f_traced.graph.nodes))
    # CSE should have removed at least some duplicate nodes
    assert nodes_after < nodes_before

    # CSE should have eliminated duplicate aten.add.Tensor nodes (x+1 appeared twice)
    adds_after = count_nodes(f_traced.graph, lambda n: n.target == _add_target)
    assert adds_after < adds_before, (
        f"CSE should reduce duplicate add nodes: {adds_before} -> {adds_after}"
    )

    report_nonclose(f_x, f_traced(x), name="f(x)")
