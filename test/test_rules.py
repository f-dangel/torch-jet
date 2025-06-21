"""Test individual simplification rules."""

from typing import Any

from pytest import mark
from torch import Size, linspace, manual_seed, rand
from torch.fx import Graph, GraphModule, Node
from torch.nn import Module
from torch.nn.functional import linear

import jet.utils
from jet import JetTracer
from jet.rules import (
    PullSumVmappedLinear,
    PullSumVmappedReplicateMultiplication,
    PullSumVmappedScalarMultiplication,
    PullSumVmappedTensorAddition,
    PushReplicateElementwise,
    PushReplicateLinear,
    PushReplicateScalarArithmetic,
    PushReplicateSumVmapped,
    PushReplicateTensorArithmetic,
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


CASES = []


# swapping replicate nodes with elementwise functions
class ReplicateElementwise(Module):
    def __init__(self, op, times=5, pos=0):
        super().__init__()
        self.op = op
        self.times = times
        self.pos = pos

    def forward(self, x):
        return self.op(jet.utils.replicate(x, self.times, pos=self.pos))


class SimpleReplicateElementwise(ReplicateElementwise):
    def forward(self, x):
        return jet.utils.replicate(self.op(x), self.times, pos=self.pos)


CASES.extend(
    [
        {
            "f": ReplicateElementwise(op),
            "f_simple": SimpleReplicateElementwise(op),
            "rules": lambda: [PushReplicateElementwise()],
            "shape": (3,),
            "id": f"replicate-{op.__module__}.{op.__name__}",
        }
        for op in PushReplicateElementwise.OPERATIONS
    ]
)


# swapping replicate nodes with arithmetic operations involving one integer/float
class ReplicateScalarArithmetic(Module):
    def __init__(self, op, times=5, pos=0, scalar=3.0):
        super().__init__()
        self.op = op
        self.times = times
        self.pos = pos
        self.scalar = scalar

    def forward(self, x):
        return self.op(jet.utils.replicate(x, self.times, pos=self.pos), self.scalar)


class SimpleReplicateScalarArithmetic(ReplicateScalarArithmetic):
    def forward(self, x):
        return jet.utils.replicate(self.op(x, self.scalar), self.times, pos=self.pos)


CASES.extend(
    [
        {
            "f": ReplicateScalarArithmetic(op),
            "f_simple": SimpleReplicateScalarArithmetic(op),
            "rules": lambda: [PushReplicateScalarArithmetic()],
            "shape": (4,),
            "id": f"replicate-{op.__module__}.{op.__name__}-scalar",
        }
        for op in PushReplicateScalarArithmetic.OPERATIONS
    ]
)


# swapping arithmetic operations that consume two replicate nodes
class ReplicateTensorArithmetic(Module):
    def __init__(self, op, times=5, pos=0, same: bool = False):
        super().__init__()
        self.op = op
        self.times = times
        self.pos = pos
        self.same = same

    def forward(self, x):
        return self.op(
            jet.utils.replicate(x, self.times, pos=self.pos),
            jet.utils.replicate(x if self.same else x + 1, self.times, pos=self.pos),
        )


class SimpleReplicateTensorArithmetic(ReplicateTensorArithmetic):
    def forward(self, x):
        return jet.utils.replicate(
            self.op(x, x if self.same else x + 1), self.times, pos=self.pos
        )


CASES.extend(
    [
        {
            "f": ReplicateTensorArithmetic(op),
            "f_simple": SimpleReplicateTensorArithmetic(op),
            "rules": lambda: [PushReplicateTensorArithmetic()],
            "shape": (4,),
            "id": f"replicate-{op.__module__}.{op.__name__}-two-tensors",
        }
        for op in PushReplicateTensorArithmetic.OPERATIONS
    ]
)


# swapping arithmetic operations that consume the same replicate node twice
CASES.extend(
    [
        {
            "f": ReplicateTensorArithmetic(op, same=True),
            "f_simple": SimpleReplicateTensorArithmetic(op, same=True),
            "rules": lambda: [PushReplicateTensorArithmetic()],
            "shape": (4,),
            "id": f"replicate-{op.__module__}.{op.__name__}-same-tensor",
        }
        for op in PushReplicateTensorArithmetic.OPERATIONS
    ]
)

# Simplify linear operation with replicated input
CASES.append(
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
        "rules": lambda: [PushReplicateLinear()],
        "shape": (4,),
        "id": "replicate-linear",
    }
)


# Pushing a replicate node through a sum_vmapped node
class ReplicateSumVmapped(Module):
    def __init__(self, pos1, pos2, times=5) -> None:
        super().__init__()
        self.pos1 = pos1
        self.pos2 = pos2
        self.times = times

    def forward(self, x):
        return jet.utils.sum_vmapped(
            jet.utils.replicate(x, self.times, pos=self.pos1), pos=self.pos2
        )


class SimpleReplicateSumVmapped(ReplicateSumVmapped):
    def forward(self, x):
        if self.pos1 == self.pos2:
            return x * self.times
        else:
            return jet.utils.replicate(
                jet.utils.sum_vmapped(
                    x, pos=self.pos2 if self.pos1 > self.pos2 else self.pos2 - 1
                ),
                self.times,
                pos=self.pos1 - 1 if self.pos1 > self.pos2 else self.pos1,
            )


CASES.extend(
    [
        {
            "f": ReplicateSumVmapped(pos1, pos2),
            "f_simple": SimpleReplicateSumVmapped(pos1, pos2),
            "rules": lambda: [PushReplicateSumVmapped()],
            "shape": (4, 3),
            "id": f"replicate{pos1}-sum_vmapped{pos2}",
        }
        for pos1, pos2 in [(2, 2), (2, 0), (0, 2)]
    ]
)


# Pulling a sum_vmapped node through an arithmetic operation with an integer/float
class SumVmappedScalarMultiplication(Module):
    def __init__(self, op, pos=0, scalar=3.0):
        super().__init__()
        self.op = op
        self.pos = pos
        self.scalar = scalar

    def forward(self, x):
        return jet.utils.sum_vmapped(self.op(x, self.scalar), pos=self.pos)


class SimpleSumVmappedScalarMultiplication(SumVmappedScalarMultiplication):
    def forward(self, x):
        return self.op(jet.utils.sum_vmapped(x, pos=self.pos), self.scalar)


CASES.extend(
    [
        {
            "f": SumVmappedScalarMultiplication(op),
            "f_simple": SimpleSumVmappedScalarMultiplication(op),
            "rules": lambda: [PullSumVmappedScalarMultiplication()],
            "shape": (4,),
            "id": f"sum_vmapped-{op.__module__}.{op.__name__}-scalar",
        }
        for op in PullSumVmappedScalarMultiplication.OPERATIONS
    ]
)


# pulling a sum_vmapped node through addition/subtraction of two tensors
class SumVmappedTensorAddition(Module):
    def __init__(self, op, pos=0):
        super().__init__()
        self.op = op
        self.pos = pos

    def forward(self, x):
        return jet.utils.sum_vmapped(self.op(x, x + 1), pos=self.pos)


class SimpleSumVmappedTensorAddition(SumVmappedTensorAddition):
    def forward(self, x):
        return self.op(
            jet.utils.sum_vmapped(x, pos=self.pos),
            jet.utils.sum_vmapped(x + 1, pos=self.pos),
        )


CASES.extend(
    [
        {
            "f": SumVmappedTensorAddition(op),
            "f_simple": SimpleSumVmappedTensorAddition(op),
            "rules": lambda: [PullSumVmappedTensorAddition()],
            "shape": (4,),
            "id": f"sum_vmapped-{op.__module__}.{op.__name__}-two-tensors",
        }
        for op in PullSumVmappedTensorAddition.OPERATIONS
    ]
)

# Pull a sum_vmapped node through a linear layer
CASES.append(
    {
        "f": lambda x: jet.utils.sum_vmapped(
            linear(x, linspace(-2.0, 10, 12).reshape(3, 4)),  # weight
            pos=0,
        ),
        "f_simple": lambda x: linear(
            jet.utils.sum_vmapped(x, pos=0),
            linspace(-2.0, 10, 12).reshape(3, 4),  # weight
        ),
        "rules": lambda: [PullSumVmappedLinear()],
        "shape": (5, 4),
        "id": "sum_vmapped-linear",
    }
)


# Pull a sum_vmapped through a multiplication, one of whose arguments is a replicate
class SumVmappedReplicateMultiplication(Module):
    def __init__(self, times=5, shape=(4,)):
        super().__init__()
        self.times = times
        self.shape = Size(shape)
        self.y = linspace(-2.0, 6.0, self.times * self.shape.numel()).reshape(
            times, *shape
        )

    def forward(self, x):
        return jet.utils.sum_vmapped(
            jet.utils.replicate(x, self.times, pos=0) * self.y, pos=0
        )


class SimpleSumVmappedReplicateMultiplication(SumVmappedReplicateMultiplication):
    def forward(self, x):
        return x * jet.utils.sum_vmapped(self.y, pos=0)


CASES.append(
    {
        "f": SumVmappedReplicateMultiplication(),
        "f_simple": SimpleSumVmappedReplicateMultiplication(),
        "rules": lambda: [PullSumVmappedReplicateMultiplication()],
        "shape": (4,),
        "id": "sum_vmapped-replicate-multiplication",
    }
)


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

    num_matches = 0
    do_simplify = True
    while do_simplify:
        do_simplify = False
        for rule in rules:
            for node in f_simplified.graph.nodes:
                if rule.match(node):
                    num_matches += 1
                    rule.apply(node, f_simplified.graph)
                    do_simplify = True
    print(f"Got {num_matches=}.")
    f_simplified.graph.eliminate_dead_code()

    # make sure all functions yield the same result
    f_x = f(x)
    assert f_x.allclose(f_simple(x))
    assert f_x.allclose(f_simplified(x))

    # compare the graphs of f_simplified and f_simple
    f_simple_graph = JetTracer().trace(f_simple)
    compare_graphs(f_simple_graph, f_simplified.graph)
