"""Test individual simplification rules."""

import operator
from itertools import product
from typing import Any, Callable

import torch
from pytest import mark
from torch import Size, Tensor, linspace, manual_seed, mul, rand
from torch.fx import Graph, Node
from torch.nn import Module

import jet.utils
from jet.rules import (
    PullSumVmappedReplicateMultiplication,
    PullSumVmappedScalarMultiplication,
    PullSumVmappedTensorAddition,
    PushReplicateElementwise,
    PushReplicateScalarArithmetic,
    PushReplicateSumVmapped,
    PushReplicateTensorArithmetic,
)
from jet.simplify import apply_once
from jet.tracing import capture_graph


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


CASES = []

# Python-level ops for constructing test computations. make_fx will trace these
# to their ATen equivalents, which the rules then match.
_ELEMENTWISE_OPS = [torch.cos, torch.sin, torch.tanh, torch.sigmoid, torch.cosh]
_SCALAR_ARITHMETIC_OPS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.pow,
]
_TENSOR_ARITHMETIC_OPS = _SCALAR_ARITHMETIC_OPS
_MULTIPLICATION_OPS = [operator.mul]
_ADDITION_OPS = [operator.add, operator.sub]


# swapping replicate nodes with elementwise functions
class ReplicateElementwise(Module):  # noqa: D101
    def __init__(self, op: Callable[[Tensor], Tensor], times: int, pos: int):  # noqa: D107
        super().__init__()
        self.op, self.times, self.pos = op, times, pos

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return self.op(jet.utils.replicate(x, self.times, pos=self.pos))


class SimpleReplicateElementwise(ReplicateElementwise):  # noqa: D101
    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return jet.utils.replicate(self.op(x), self.times, pos=self.pos)


CASES.extend(
    [
        {
            "f": ReplicateElementwise(op, times=2, pos=pos),
            "f_simple": SimpleReplicateElementwise(op, times=2, pos=pos),
            "rules": lambda: [PushReplicateElementwise()],
            "shape": (3,),
            "id": f"replicate{pos}-{op.__module__}.{op.__name__}",
        }
        for op, pos in product(_ELEMENTWISE_OPS, [0, 1])
    ]
)


# swapping replicate nodes with arithmetic operations involving one integer/float
class ReplicateScalarArithmetic(Module):  # noqa: D101
    def __init__(  # noqa: D107
        self,
        op: Callable[[float | int | Tensor, float | int | Tensor], Tensor],
        times: int,
        pos: int,
        scalar: int | float,
        scalar_first: bool,
    ):
        super().__init__()
        self.op = op
        self.times = times
        self.pos = pos
        self.scalar = scalar
        self.scalar_first = scalar_first

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x_rep = jet.utils.replicate(x, self.times, pos=self.pos)
        return (
            self.op(self.scalar, x_rep)
            if self.scalar_first
            else self.op(x_rep, self.scalar)
        )


class SimpleReplicateScalarArithmetic(ReplicateScalarArithmetic):  # noqa: D101
    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        res = self.op(self.scalar, x) if self.scalar_first else self.op(x, self.scalar)
        return jet.utils.replicate(res, self.times, pos=self.pos)


CASES.extend(
    [
        {
            "f": ReplicateScalarArithmetic(
                op, times=5, pos=0, scalar=3.0, scalar_first=first
            ),
            "f_simple": SimpleReplicateScalarArithmetic(
                op, times=5, pos=0, scalar=3.0, scalar_first=first
            ),
            "rules": lambda: [PushReplicateScalarArithmetic()],
            "shape": (4,),
            "id": f"replicate-{op.__module__}.{op.__name__}-scalar-{first=}",
        }
        for op, first in product(_SCALAR_ARITHMETIC_OPS, [False, True])
        # Skip scalar_first=True for truediv: it decomposes into reciprocal + mul
        if not (first and op is operator.truediv)
    ]
)


# swapping arithmetic operations that consume two replicate nodes
class ReplicateTensorArithmetic(Module):  # noqa: D101
    def __init__(  # noqa: D107
        self, op: Callable[[Tensor, Tensor], Tensor], times: int, pos: int, same: bool
    ):
        super().__init__()
        self.op = op
        self.times = times
        self.pos = pos
        self.same = same

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        y = x if self.same else x + 1
        return self.op(
            jet.utils.replicate(x, self.times, pos=self.pos),
            jet.utils.replicate(y, self.times, pos=self.pos),
        )


class SimpleReplicateTensorArithmetic(ReplicateTensorArithmetic):  # noqa: D101
    def forward(self, x):  # noqa: D102
        y = x if self.same else x + 1
        return jet.utils.replicate(self.op(x, y), self.times, pos=self.pos)


CASES.extend(
    [
        {
            "f": ReplicateTensorArithmetic(op, times=5, pos=0, same=same),
            "f_simple": SimpleReplicateTensorArithmetic(op, times=5, pos=0, same=same),
            "rules": lambda: [PushReplicateTensorArithmetic()],
            "shape": (4,),
            "id": f"replicate-{op.__module__}.{op.__name__}-tensors-{same=}",
        }
        for op, same in product(_TENSOR_ARITHMETIC_OPS, [False, True])
    ]
)


# Simplify linear operation with replicated input
# NOTE: Skipped because make_fx decomposes `linear` into lower-level ops
# (mm, addmm, t, view), so PushReplicateLinear cannot match.
# CASES.append(
#     {
#         "f": lambda x: linear(
#             jet.utils.replicate(x, 5, pos=0),
#             linspace(-2.0, 10, 12).reshape(3, 4),
#             linspace(-1.0, 2.0, 3),
#         ),
#         ...
#     }
# )


# Pushing a replicate node through a sum_vmapped node
class ReplicateSumVmapped(Module):  # noqa: D101
    def __init__(self, pos1: int, pos2: int, times: int) -> None:  # noqa: D107
        super().__init__()
        self.pos1, self.pos2, self.times = pos1, pos2, times

    def forward(self, x):  # noqa: D102
        x_rep = jet.utils.replicate(x, self.times, pos=self.pos1)
        return jet.utils.sum_vmapped(x_rep, self.pos2)


class SimpleReplicateSumVmapped(ReplicateSumVmapped):  # noqa: D101
    def forward(self, x):  # noqa: D102
        if self.pos1 == self.pos2:
            return x * self.times

        new_sum_pos = self.pos2 if self.pos1 > self.pos2 else self.pos2 - 1
        x_sum = jet.utils.sum_vmapped(x, new_sum_pos)

        new_rep_pos = self.pos1 - 1 if self.pos1 > self.pos2 else self.pos1
        return jet.utils.replicate(x_sum, self.times, pos=new_rep_pos)


CASES.extend(
    [
        {
            "f": ReplicateSumVmapped(pos1, pos2, times=5),
            "f_simple": SimpleReplicateSumVmapped(pos1, pos2, times=5),
            "rules": lambda: [PushReplicateSumVmapped()],
            "shape": (4, 3),
            "id": f"replicate{pos1}-sum_vmapped{pos2}",
        }
        for pos1, pos2 in [(2, 2), (2, 0), (0, 2)]
    ]
)


# Pulling a sum_vmapped node through an arithmetic operation with an integer/float
class SumVmappedScalarMultiplication(Module):  # noqa: D101
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
        return jet.utils.sum_vmapped(res, self.pos)


class SimpleSumVmappedScalarMultiplication(SumVmappedScalarMultiplication):  # noqa: D101
    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x_sum = jet.utils.sum_vmapped(x, self.pos)
        return (
            self.op(self.scalar, x_sum)
            if self.scalar_first
            else self.op(x_sum, self.scalar)
        )


CASES.extend(
    [
        {
            "f": SumVmappedScalarMultiplication(
                op, pos=0, scalar=3.0, scalar_first=first
            ),
            "f_simple": SimpleSumVmappedScalarMultiplication(
                op, pos=0, scalar=3.0, scalar_first=first
            ),
            "rules": lambda: [PullSumVmappedScalarMultiplication()],
            "shape": (4,),
            "id": f"sum_vmapped-{op.__module__}.{op.__name__}-scalar-{first=}",
        }
        for op, first in product(_MULTIPLICATION_OPS, [False, True])
    ]
)


# pulling a sum_vmapped node through addition/subtraction of two tensors
class SumVmappedTensorAddition(Module):  # noqa: D101
    def __init__(self, op: Callable[[Tensor, Tensor], Tensor], pos: int):  # noqa: D107
        super().__init__()
        self.op, self.pos = op, pos

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        y = x + 1
        return jet.utils.sum_vmapped(self.op(x, y), self.pos)


class SimpleSumVmappedTensorAddition(SumVmappedTensorAddition):  # noqa: D101
    def forward(self, x):  # noqa: D102
        return self.op(
            jet.utils.sum_vmapped(x, self.pos),
            jet.utils.sum_vmapped(x + 1, self.pos),
        )


CASES.extend(
    [
        {
            "f": SumVmappedTensorAddition(op, pos=0),
            "f_simple": SimpleSumVmappedTensorAddition(op, pos=0),
            "rules": lambda: [PullSumVmappedTensorAddition()],
            "shape": (4,),
            "id": f"sum_vmapped-{op.__module__}.{op.__name__}-two-tensors",
        }
        for op in _ADDITION_OPS
    ]
)

# Pull a sum_vmapped node through a linear layer
# NOTE: Skipped because make_fx decomposes `linear` into lower-level ops
# (mm, addmm, t, view), so PullSumVmappedLinear cannot match.
# CASES.append(
#     {
#         "f": lambda x: jet.utils.sum_vmapped(
#             linear(x, linspace(-2.0, 10, 12).reshape(3, 4)),
#             pos=0,
#         ),
#         ...
#     }
# )


# Pull a sum_vmapped through a multiplication, one of whose arguments is a replicate
class SumVmappedReplicateMultiplication(Module):  # noqa: D101
    def __init__(  # noqa: D107
        self, times: int, shape: tuple[int, ...], pos: int, replicate_first: bool
    ):
        super().__init__()
        self.times = times
        self.shape = Size(shape)
        self.pos = pos
        self.replicate_first = replicate_first
        self.y = linspace(-2.0, 6.0, self.times * self.shape.numel()).reshape(
            times, *shape
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        res = (
            jet.utils.replicate(x, self.times, pos=0) * self.y
            if self.replicate_first
            else mul(self.y, jet.utils.replicate(x, self.times, pos=0))
        )
        return jet.utils.sum_vmapped(res, self.pos)


class SimpleSumVmappedReplicateMultiplication(SumVmappedReplicateMultiplication):  # noqa: D101
    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return (
            x * jet.utils.sum_vmapped(self.y, self.pos)
            if self.replicate_first
            else mul(jet.utils.sum_vmapped(self.y, self.pos), x)
        )


CASES.extend(
    [
        {
            "f": SumVmappedReplicateMultiplication(
                times=5, shape=(4,), pos=0, replicate_first=first
            ),
            "f_simple": SimpleSumVmappedReplicateMultiplication(
                times=5, shape=(4,), pos=0, replicate_first=first
            ),
            "rules": lambda: [
                PullSumVmappedReplicateMultiplication(),
            ],
            "shape": (4,),
            "id": f"sum_vmapped-replicate-multiplication-{first=}",
        }
        for first in [True, False]
    ]
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
    f_simplified = capture_graph(f, example_input=x)

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
    f_simple_mod = capture_graph(f_simple, example_input=x)
    compare_graphs(f_simple_mod.graph, f_simplified.graph)
