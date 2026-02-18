"""Test individual simplification rules."""

import operator
from typing import Callable

import torch
from pytest import mark
from torch import Tensor, linspace, manual_seed, rand
from torch.fx import Graph, Node
from torch.nn import Module
from torch.nn.functional import linear

from jet.rules import (
    PullSumAddMM,
    PullSumBroadcastedMultiplication,
    PullSumMM,
    PullSumScalarMultiplication,
    PullSumSqueeze,
    PullSumTensorAddition,
    PullSumUnsqueeze,
    PullSumView,
)
from jet.simplify import apply_all
from jet.tracing import capture_graph

_aten = torch.ops.aten


class RuleTestCase(Module):
    """A test case for a simplification rule.

    Subclasses define ``forward`` (the original computation) and
    ``forward_simple`` (the expected simplified computation), along with
    ``rules``, ``shape``, and ``id``.
    """

    shape: tuple[int, ...]
    id: str
    rules: list

    def forward_simple(self, x: Tensor) -> Tensor:  # noqa: D102
        raise NotImplementedError


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
        if node1.op not in ("get_attr", "placeholder"):
            assert node1.target == node2.target
        assert len(node1.args) == len(node2.args)
        for arg1, arg2 in zip(node1.args, node2.args):
            if isinstance(arg1, Node) and isinstance(arg2, Node):
                # node comparison
                assert arg1.op == arg2.op
                if arg1.op not in ("get_attr", "placeholder"):
                    assert arg1.target == arg2.target
                assert node_mapping[arg1] == arg2
            else:
                assert arg1 == arg2
        # TODO Support comparing kwargs that contain nodes
        assert node1.kwargs == node2.kwargs

        # nodes match, hence add them to the mapping
        node_mapping[node1] = node2


CASES: list[RuleTestCase] = []


# Pulling a sum node through an arithmetic operation with an integer/float
class SumScalarMultiplication(RuleTestCase):  # noqa: D101
    """Test case for ``sum(5 * x) = 5 * sum(x)``."""

    shape = (4,)

    def __init__(  # noqa: D107
        self, pos: int, scalar: float | int, scalar_first: bool
    ):
        super().__init__()
        self.pos = pos
        self.scalar = scalar
        self.scalar_first = scalar_first
        self.id = f"sum-scalar-{scalar_first=}"
        self.rules = [PullSumScalarMultiplication()]

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        res = self.scalar * x if self.scalar_first else x * self.scalar
        return res.sum(self.pos)

    def forward_simple(self, x: Tensor) -> Tensor:  # noqa: D102
        x_sum = x.sum(self.pos)
        return self.scalar * x_sum if self.scalar_first else x_sum * self.scalar


_POS, _SCALAR = 0, 3.0
CASES.extend(
    SumScalarMultiplication(_POS, _SCALAR, scalar_first)
    for scalar_first in [False, True]
)

_ADDITION_OPS = [operator.add, operator.sub]


# Pulling a sum node through addition/subtraction of two tensors
class SumTensorAddition(RuleTestCase):  # noqa: D101
    shape = (4,)

    def __init__(self, op: Callable[[Tensor, Tensor], Tensor], pos: int):  # noqa: D107
        super().__init__()
        self.op, self.pos = op, pos
        self.id = f"sum-{op.__module__}.{op.__name__}-two-tensors"
        self.rules = [PullSumTensorAddition()]

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        y = x + 1
        return self.op(x, y).sum(self.pos)

    def forward_simple(self, x: Tensor) -> Tensor:  # noqa: D102
        return self.op(
            x.sum(self.pos),
            (x + 1).sum(self.pos),
        )


CASES.extend(SumTensorAddition(op, pos=0) for op in _ADDITION_OPS)


# Pulling a sum node through a broadcasted tensor multiplication
class SumBroadcastedMul(RuleTestCase):  # noqa: D101
    shape = (5, 4)
    id = "sum-broadcasted-mul"
    rules = [PullSumBroadcastedMultiplication()]

    def __init__(self, pos: int):  # noqa: D107
        super().__init__()
        self.pos = pos

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        b = linspace(1.0, 5.0, 4)
        return (x * b).sum(self.pos)

    def forward_simple(self, x: Tensor) -> Tensor:  # noqa: D102
        b = linspace(1.0, 5.0, 4)
        return x.sum(self.pos) * b


CASES.append(SumBroadcastedMul(pos=0))


# Pull a sum node through mm (linear without bias).
class SumMM(RuleTestCase):  # noqa: D101
    shape = (5, 4)
    id = "sum-mm"
    rules = [PullSumMM()]

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return linear(x, linspace(-2.0, 10, 12).reshape(3, 4)).sum(0)

    def forward_simple(self, x: Tensor) -> Tensor:  # noqa: D102
        W = linspace(-2.0, 10, 12).reshape(3, 4)
        Wt = _aten.t.default(W)
        sv = x.sum(0)
        out = _aten.mm.default(_aten.unsqueeze.default(sv, 0), Wt)
        return _aten.squeeze.dim(out, 0)


CASES.append(SumMM())


# Pull a sum node through mm (linear without bias), summing over last dim.
class SumMMLastDim(RuleTestCase):  # noqa: D101
    shape = (5, 4)
    id = "sum-mm-last-dim"
    rules = [PullSumMM()]

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return linear(x, linspace(-2.0, 10, 12).reshape(3, 4)).sum(1)

    def forward_simple(self, x: Tensor) -> Tensor:  # noqa: D102
        W = linspace(-2.0, 10, 12).reshape(3, 4)
        Wt = _aten.t.default(W)
        sv = Wt.sum(1)
        out = _aten.mm.default(x, _aten.unsqueeze.default(sv, 1))
        return _aten.squeeze.dim(out, 1)


CASES.append(SumMMLastDim())


# Pull a sum node through addmm (linear with bias).
class SumAddmm(RuleTestCase):  # noqa: D101
    shape = (5, 4)
    id = "sum-addmm"
    rules = [PullSumAddMM()]

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return linear(
            x, linspace(-2.0, 10, 12).reshape(3, 4), linspace(-1.0, 2.0, 3)
        ).sum(0)

    def forward_simple(self, x: Tensor) -> Tensor:  # noqa: D102
        W = linspace(-2.0, 10, 12).reshape(3, 4)
        b = linspace(-1.0, 2.0, 3)
        Wt = _aten.t.default(W)
        sv = x.sum(0)
        out = _aten.mm.default(_aten.unsqueeze.default(sv, 0), Wt)
        out = _aten.squeeze.dim(out, 0)
        scaled_b = _aten.mul.Tensor(b, x.shape[0])
        return _aten.add.Tensor(out, scaled_b)


CASES.append(SumAddmm())


# Pull a sum node through addmm (linear with bias), summing over last dim.
class SumAddmmLastDim(RuleTestCase):  # noqa: D101
    shape = (5, 4)
    id = "sum-addmm-last-dim"
    rules = [PullSumAddMM()]

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return linear(
            x, linspace(-2.0, 10, 12).reshape(3, 4), linspace(-1.0, 2.0, 3)
        ).sum(1)

    def forward_simple(self, x: Tensor) -> Tensor:  # noqa: D102
        W = linspace(-2.0, 10, 12).reshape(3, 4)
        b = linspace(-1.0, 2.0, 3)
        Wt = _aten.t.default(W)
        sv = Wt.sum(1)
        out = _aten.mm.default(x, _aten.unsqueeze.default(sv, 1))
        out = _aten.squeeze.dim(out, 1)
        sum_b = _aten.sum.default(b)
        return _aten.add.Tensor(out, sum_b)


CASES.append(SumAddmmLastDim())


# Pull a sum node through squeeze.
class SumSqueeze(RuleTestCase):  # noqa: D101
    shape = (5, 1, 4)
    id = "sum-squeeze"
    rules = [PullSumSqueeze()]

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return x.squeeze(1).sum(0)

    def forward_simple(self, x: Tensor) -> Tensor:  # noqa: D102
        return _aten.squeeze.dim(x.sum(0), 0)


CASES.append(SumSqueeze())


# Pull a sum node through unsqueeze (no-op: sum dim == unsqueeze dim).
class SumUnsqueezeNoop(RuleTestCase):  # noqa: D101
    shape = (5, 4)
    id = "sum-unsqueeze-noop"
    rules = [PullSumUnsqueeze()]

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return x.unsqueeze(1).sum(1)

    def forward_simple(self, x: Tensor) -> Tensor:  # noqa: D102
        return x


CASES.append(SumUnsqueezeNoop())


# Pull a sum node through unsqueeze (swap: dims differ).
class SumUnsqueezeSwap(RuleTestCase):  # noqa: D101
    shape = (5, 4)
    id = "sum-unsqueeze-swap"
    rules = [PullSumUnsqueeze()]

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return x.unsqueeze(0).sum(1)

    def forward_simple(self, x: Tensor) -> Tensor:  # noqa: D102
        return _aten.unsqueeze.default(x.sum(0), 0)


CASES.append(SumUnsqueezeSwap())


# Pull a sum node through view.
class SumView(RuleTestCase):  # noqa: D101
    shape = (5, 4)
    id = "sum-view"
    rules = [PullSumView()]

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return x.view(5, 2, 2).sum(0)

    def forward_simple(self, x: Tensor) -> Tensor:  # noqa: D102
        return _aten.view.default(x.sum(0), [2, 2])


CASES.append(SumView())


@mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_simplification_rules(case: RuleTestCase):
    """Test simplification rules.

    Args:
        case: The test case specifying the rule to test.
    """
    manual_seed(0)
    x = rand(*case.shape)
    # simplify the function
    f_simplified = capture_graph(case, x)

    do_simplify = True
    while do_simplify:
        do_simplify = apply_all(case.rules, f_simplified, verbose=True)
    f_simplified.graph.eliminate_dead_code()

    # make sure all functions yield the same result
    f_x = case(x)
    assert f_x.allclose(case.forward_simple(x))
    f_simplified.recompile()
    assert f_x.allclose(f_simplified(x))

    # compare the graphs of f_simplified and f_simple
    f_simple_mod = capture_graph(lambda x: case.forward_simple(x), x)  # noqa: PLW0108
    compare_graphs(f_simple_mod.graph, f_simplified.graph)


# === Negative test cases: rules should NOT trigger ===

NEGATIVE_CASES: list[RuleTestCase] = []


# sum(x + y) where x and y have different shapes (broadcasted addition)
class SumAddBroadcasted(RuleTestCase):  # noqa: D101
    shape = (3, 4)
    id = "neg-sum-add-broadcasted"
    rules = [PullSumTensorAddition()]

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        y = linspace(1.0, 4.0, 4)
        return (x + y).sum(0)


NEGATIVE_CASES.append(SumAddBroadcasted())


# sum(x + scalar) where the second arg is a Python float, not a Node
class SumAddScalar(RuleTestCase):  # noqa: D101
    shape = (3, 4)
    id = "neg-sum-add-scalar"
    rules = [PullSumTensorAddition()]

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return (x + 5.0).sum(0)


NEGATIVE_CASES.append(SumAddScalar())


# sum(x * y) where both args are tensors (no scalar)
class SumMulTwoTensors(RuleTestCase):  # noqa: D101
    shape = (4,)
    id = "neg-sum-mul-two-tensors"
    rules = [PullSumScalarMultiplication()]

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        y = x + 1
        return (x * y).sum(0)


NEGATIVE_CASES.append(SumMulTwoTensors())


# sum(x * y) where both tensors have the same shape (neither broadcasted)
class SumMulSameShape(RuleTestCase):  # noqa: D101
    shape = (5, 4)
    id = "neg-sum-mul-same-shape"
    rules = [PullSumBroadcastedMultiplication()]

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        y = x + 1
        return (x * y).sum(0)


NEGATIVE_CASES.append(SumMulSameShape())


# sum(view(x, ...), 0) where the dim doesn't map cleanly (split dim)
class SumViewSplitDim(RuleTestCase):  # noqa: D101
    shape = (10, 4)
    id = "neg-sum-view-split-dim"
    rules = [PullSumView()]

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return x.view(2, 5, 4).sum(0)


NEGATIVE_CASES.append(SumViewSplitDim())


@mark.parametrize("case", NEGATIVE_CASES, ids=lambda c: c.id)
def test_rules_do_not_trigger(case: RuleTestCase):
    """Test that rules do NOT trigger on non-matching patterns.

    Args:
        case: The test case specifying the rule to test.
    """
    manual_seed(0)
    x = rand(*case.shape)
    f = capture_graph(case, x)
    graph_before = str(f.graph)
    applied = apply_all(case.rules, f, verbose=True)
    assert not applied, f"Rule unexpectedly triggered for {case.id}"
    assert str(f.graph) == graph_before, "Graph was modified despite no rule match"
