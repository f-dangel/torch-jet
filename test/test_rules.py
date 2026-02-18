"""Test individual simplification rules."""

import operator
from itertools import product
from typing import Any, Callable

import torch
from pytest import mark
from torch import Tensor, linspace, manual_seed, rand
from torch.fx import Graph, Node
from torch.nn import Module
from torch.nn.functional import linear

from jet.rules import (
    PullSumLinear,
    PullSumScalarMultiplication,
    PullSumTensorAddition,
)
from jet.simplify import apply_once
from jet.tracing import capture_graph

_aten = torch.ops.aten

CASES = []

_MULTIPLICATION_OPS = [operator.mul]
_ADDITION_OPS = [operator.add, operator.sub]


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
        if not node1.op == node2.op == "get_attr":
            assert node1.target == node2.target
        assert len(node1.args) == len(node2.args)
        for arg1, arg2 in zip(node1.args, node2.args):
            if isinstance(arg1, Node) and isinstance(arg2, Node):
                # node comparison
                assert arg1.op == arg2.op
                if not arg1.op == arg2.op == "get_attr":
                    assert arg1.target == arg2.target
                assert node_mapping[arg1] == arg2
            else:
                assert arg1 == arg2
        # TODO Support comparing kwargs that contain nodes
        assert node1.kwargs == node2.kwargs

        # nodes match, hence add them to the mapping
        node_mapping[node1] = node2


# Pulling a sum node through an arithmetic operation with an integer/float
class SumScalarMultiplication(Module):  # noqa: D101
    def __init__(  # noqa: D107
        self,
        op: Callable[[float | int | Tensor, float | int | Tensor], Tensor],
        pos: int,
        scalar: float | int,
        scalar_first: bool,
    ):
        super().__init__()
        self.op = op
        self.pos = pos
        self.scalar = scalar
        self.scalar_first = scalar_first

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        res = self.op(self.scalar, x) if self.scalar_first else self.op(x, self.scalar)
        return res.sum(self.pos)


class SimpleSumScalarMultiplication(SumScalarMultiplication):  # noqa: D101
    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x_sum = x.sum(self.pos)
        return (
            self.op(self.scalar, x_sum)
            if self.scalar_first
            else self.op(x_sum, self.scalar)
        )


CASES.extend(
    [
        {
            "f": SumScalarMultiplication(op, pos=0, scalar=3.0, scalar_first=first),
            "f_simple": SimpleSumScalarMultiplication(
                op, pos=0, scalar=3.0, scalar_first=first
            ),
            "rules": lambda: [PullSumScalarMultiplication()],
            "shape": (4,),
            "id": f"sum-{op.__module__}.{op.__name__}-scalar-{first=}",
        }
        for op, first in product(_MULTIPLICATION_OPS, [False, True])
    ]
)


# pulling a sum node through addition/subtraction of two tensors
class SumTensorAddition(Module):  # noqa: D101
    def __init__(self, op: Callable[[Tensor, Tensor], Tensor], pos: int):  # noqa: D107
        super().__init__()
        self.op, self.pos = op, pos

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        y = x + 1
        return self.op(x, y).sum(self.pos)


class SimpleSumTensorAddition(SumTensorAddition):  # noqa: D101
    def forward(self, x):  # noqa: D102
        return self.op(
            x.sum(self.pos),
            (x + 1).sum(self.pos),
        )


CASES.extend(
    [
        {
            "f": SumTensorAddition(op, pos=0),
            "f_simple": SimpleSumTensorAddition(op, pos=0),
            "rules": lambda: [PullSumTensorAddition()],
            "shape": (4,),
            "id": f"sum-{op.__module__}.{op.__name__}-two-tensors",
        }
        for op in _ADDITION_OPS
    ]
)


# Pull a sum node through mm (linear without bias).
class SumMM(Module):  # noqa: D101
    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return linear(x, linspace(-2.0, 10, 12).reshape(3, 4)).sum(0)


class SimpleSumMM(SumMM):  # noqa: D101
    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        W = linspace(-2.0, 10, 12).reshape(3, 4)
        Wt = _aten.t.default(W)
        sv = x.sum(0)
        out = _aten.mm.default(_aten.unsqueeze.default(sv, 0), Wt)
        return _aten.squeeze.dim(out, 0)


CASES.append(
    {
        "f": SumMM(),
        "f_simple": SimpleSumMM(),
        "rules": lambda: [PullSumLinear()],
        "shape": (5, 4),
        "id": "sum-mm",
    }
)


# Pull a sum node through addmm (linear with bias).
class SumAddmm(Module):  # noqa: D101
    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return linear(
            x, linspace(-2.0, 10, 12).reshape(3, 4), linspace(-1.0, 2.0, 3)
        ).sum(0)


class SimpleSumAddmm(SumAddmm):  # noqa: D101
    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        W = linspace(-2.0, 10, 12).reshape(3, 4)
        b = linspace(-1.0, 2.0, 3)
        Wt = _aten.t.default(W)
        sv = x.sum(0)
        out = _aten.mm.default(_aten.unsqueeze.default(sv, 0), Wt)
        out = _aten.squeeze.dim(out, 0)
        scaled_b = _aten.mul.Tensor(b, x.shape[0])
        return _aten.add.Tensor(out, scaled_b)


CASES.append(
    {
        "f": SumAddmm(),
        "f_simple": SimpleSumAddmm(),
        "rules": lambda: [PullSumLinear()],
        "shape": (5, 4),
        "id": "sum-addmm",
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
    f_simplified = capture_graph(f, x)

    do_simplify = True
    while do_simplify:
        do_simplify = apply_once(rules, f_simplified, verbose=True)
    f_simplified.graph.eliminate_dead_code()

    # make sure all functions yield the same result
    f_x = f(x)
    assert f_x.allclose(f_simple(x))
    f_simplified.recompile()
    assert f_x.allclose(f_simplified(x))

    # compare the graphs of f_simplified and f_simple
    f_simple_mod = capture_graph(f_simple, x)
    compare_graphs(f_simple_mod.graph, f_simplified.graph)
