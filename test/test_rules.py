"""Test individual simplification rules."""

from typing import Any

from pytest import mark
from torch import linspace, manual_seed, rand
from torch.fx import Graph, GraphModule, Node
from torch.nn.functional import linear

import jet.utils
from jet import JetTracer
from jet.rules import (
    SwapReplicateElementwise,
    SwapReplicateLinear,
    SwapReplicateScalarArithmetic,
    SwapReplicateSumVmapped,
    SwapReplicateTensorArithmetic,
)
from jet.utils import WrapperModule


def compare_graphs(graph1: Graph, graph2: Graph):
    """Compare two computation graphs for equality.

    Args:
        graph1: First computation graph.
        graph2: Second computation graph.
    """
    print(f"Comparing graphs: {graph1}\n{graph2}")
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
    # swapping replicate nodes with elementwise functions
    *[
        {
            "f": lambda x: f(jet.utils.replicate(x, 5, pos=0)),
            "f_simple": lambda x: jet.utils.replicate(f(x), 5, pos=0),
            "rules": lambda: [SwapReplicateElementwise()],
            "shape": (3,),
            "id": f"replicate-{f.__module__}.{f.__name__}",
        }
        for f in SwapReplicateElementwise.OPERATIONS
    ],
    # swapping replicate nodes with arithmetic operations involving one integer/float
    *[
        {
            "f": lambda x: f(jet.utils.replicate(x, 5, pos=0), 3.0),
            "f_simple": lambda x: jet.utils.replicate(f(x, 3.0), 5, pos=0),
            "rules": lambda: [SwapReplicateScalarArithmetic()],
            "shape": (4,),
            "id": f"replicate-{f.__module__}.{f.__name__}-float",
        }
        for f in SwapReplicateScalarArithmetic.OPERATIONS
    ],
    # swapping arithmetic operations that consume two replicate nodes
    *[
        {
            "f": lambda x: f(  # y = x + 1 here
                jet.utils.replicate(x, 5, pos=0), jet.utils.replicate(x + 1, 5, pos=0)
            ),
            "f_simple": lambda x: jet.utils.replicate(f(x, x + 1), 5, pos=0),
            "rules": lambda: [SwapReplicateTensorArithmetic()],
            "shape": (4,),
            "id": f"replicate-{f.__module__}.{f.__name__}-two-tensors",
        }
        for f in SwapReplicateTensorArithmetic.OPERATIONS
    ],
    # swapping arithmetic operations that consume the same replicate node twice
    *[
        {
            "f": lambda x: f(
                jet.utils.replicate(x, 5, pos=0), jet.utils.replicate(x, 5, pos=0)
            ),
            "f_simple": lambda x: jet.utils.replicate(f(x, x), 5, pos=0),
            "rules": lambda: [SwapReplicateTensorArithmetic()],
            "shape": (4,),
            "id": f"replicate-{f.__module__}.{f.__name__}-same-tensor",
        }
        for f in SwapReplicateTensorArithmetic.OPERATIONS
    ],
    # Simplify linear operation with replicated input
    {
        "f": lambda x: linear(
            jet.utils.replicate(x, 5, pos=0),
            linspace(-2.0, 10, 12).reshape(3, 4),  # weight
            linspace(-1.0, 2.0, 3),  # bias
        ),
        "f_simple": lambda x: jet.utils.replicate(
            linear(
                x,
                linspace(-2.0, 10, 12).reshape(3, 4),  # weight
                linspace(-1.0, 2.0, 3),  # bias
            ),
            5,
            pos=0,
        ),
        "rules": lambda: [SwapReplicateLinear()],
        "shape": (4,),
        "id": "replicate-linear",
    },
    # Pushing a replicate node through a sum_vmapped node
    *[
        {
            "f": lambda x: jet.utils.sum_vmapped(
                jet.utils.replicate(x, 5, pos=pos1), pos=pos2
            ),
            "f_simple": lambda x: (
                x * 5
                if pos1 == pos2
                else jet.utils.replicate(
                    jet.utils.sum_vmapped(x, pos=pos2 if pos1 > pos2 else pos2 - 1),
                    5,
                    pos=pos1 - 1 if pos1 > pos2 else pos1,
                )
            ),
            "rules": lambda: [SwapReplicateSumVmapped()],
            "shape": (4, 3),
            "id": f"replicate{pos1}-sum_vmapped{pos2}",
        }
        for pos1, pos2 in [(2, 2), (2, 0), (0, 2)]
    ],
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
    f_mod = WrapperModule(f)
    f_simplified = GraphModule(f_mod, JetTracer().trace(f_mod))

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
    f_x = f(x)
    assert f_x.allclose(f_simple(x))
    assert f_x.allclose(f_simplified(x))

    # compare the graphs of f_simplified and f_simple
    f_simple_graph = JetTracer().trace(f_simple)
    compare_graphs(f_simple_graph, f_simplified.graph)
