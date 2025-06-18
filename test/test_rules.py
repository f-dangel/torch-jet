"""Test individual simplification rules."""

from typing import Any

from pytest import mark
from torch import cos, manual_seed, rand, sigmoid, sin, tanh
from torch.fx import Graph, GraphModule, Node

import jet.utils
from jet import JetTracer
from jet.rules import SwapReplicateElementwise


def compare_graphs(graph1: Graph, graph2: Graph):
    """Compare two computation graphs for equality.

    Args:
        graph1: First computation graph.
        graph2: Second computation graph.
    """
    assert len(graph1.nodes) == len(graph2.nodes)

    # maps nodes in graph1 to their equivalents in graph2
    node_mapping = {}

    for node1, node2 in zip(graph1.nodes, graph2.nodes):
        assert node1.op == node2.op
        assert node1.target == node2.target
        assert len(node1.args) == len(node2.args)
        for arg1, arg2 in zip(node1.args, node2.args):
            if isinstance(arg1, Node) and isinstance(arg2, Node):
                # node comparison
                assert arg1.op == arg2.op
                assert arg1.target == arg2.target
                assert node_mapping[arg1] == arg2
            else:
                assert arg1 == arg2
        assert node1.kwargs == node2.kwargs

        # nodes match, hence add them to the mapping
        node_mapping[node1] = node2


CASES = [
    *[
        {
            "f": lambda x: f(jet.utils.replicate(x, 5, pos=0)),
            "f_simple": lambda x: jet.utils.replicate(f(x), 5, pos=0),
            "rules": lambda: [SwapReplicateElementwise()],
            "shape": (3,),
            "id": f"replicate-{f.__name__}",
        }
        for f in [cos, sin, tanh, sigmoid]
    ]
]


@mark.parametrize("config", CASES, ids=lambda conf: conf["id"])
def test_simplification_rules(config: dict[str, Any]):
    """Test simplification rules.

    Args:
      config: A dictionary specifying the test case.
    """
    manual_seed(0)
    f, f_simple, shape = config["f"], config["f_simple"], config["shape"]
    x = rand(*shape)
    rules = config["rules"]()

    # simplify the function
    f_simplified = GraphModule({}, JetTracer().trace(f))

    do_simplify = True
    while do_simplify:
        do_simplify = False
        for rule in rules:
            for node in f_simplified.graph.nodes:
                if rule.match(node):
                    rule.apply(node, f_simplified.graph)
                    do_simplify = True
    f_simplified.graph.eliminate_dead_code()

    # make sure all functions yield the same result
    x = rand(10)
    f_x = f(x)
    assert f_x.allclose(f_simple(x))
    assert f_x.allclose(f_simplified(x))

    # compare the graphs of f_simplified and f_simple
    f_simple_graph = JetTracer().trace(f_simple)
    compare_graphs(f_simple_graph, f_simplified.graph)
