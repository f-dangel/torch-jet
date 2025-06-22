"""Test simplification rules."""

import operator
from copy import deepcopy

from pytest import mark
from torch import Tensor, manual_seed, rand
from torch.fx import GraphModule, Node
from torch.nn import Module

import jet.utils
from jet import JetTracer
from jet.simplify import simplify


class SumVmappedReplicate(Module):
    """First replicate along a first axis, then sum a tensor along a second one."""

    def __init__(self, times: int, pos1: int, pos2: int):
        """Initialize the module.

        Args:
            times: Number of times to replicate the input tensor.
            pos1: Position along which to replicate the input tensor.
            pos2: Position along which to sum the replicated tensor.
        """
        super().__init__()
        self.times = times
        self.pos1 = pos1
        self.pos2 = pos2

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the module.

        Args:
            x: Input tensor to be replicated and summed.

        Returns:
            The result of replicating `x` and then summing it along the
            specified axis.
        """
        return jet.utils.sum_vmapped(
            jet.utils.replicate(x, self.times, pos=self.pos1), pos=self.pos2
        )


POS1_POS2 = [
    (2, 2),  # case 1
    (2, 0),  # case 2
    (0, 2),  # case 3
]


@mark.parametrize("pos1, pos2", POS1_POS2)
def test_push_replicate_through_sum_vmapped(pos1: int, pos2: int, times: int = 5):
    """Test the simplification of pushing replicate nodes through sum_vmapped nodes.

    Consider `sum_vmapped(replicate(x, times, pos1), pos2)`.
    This constellation is implemented in the `SumVmappedReplicate` module.
    There are three different scenarios how to simplify this:

    1. `pos1 == pos2`: `times * x`
    2. `pos1 > pos2`: `replicate(sum_vmapped(x, pos2), times, pos1 - 1)`
    3. `pos1 < pos2`: `replicate(sum_vmapped(x, pos2 - 1), times, pos1)`

    Args:
        pos1: Position along which to replicate the input tensor.
        pos2: Position along which to sum the replicated tensor.
        times: Number of times to replicate the input tensor. Default: `5`.
    """
    # generate synthetic input, module representing the computation, and its graph
    manual_seed(0)
    x = rand(8, 7)
    f = SumVmappedReplicate(times, pos1, pos2)
    f_x = f(x)
    f_mod = GraphModule(f, JetTracer().trace(f))
    f_graph = f_mod.graph

    # perform the simplification
    f_mod = simplify(f_mod)
    f_graph = f_mod.graph

    # verify the graph
    if pos1 == pos2:
        (n_x, n_mul, n_out) = f_graph.nodes
        assert (n_x.op, n_mul.op, n_out.op) == (
            "placeholder",
            "call_function",
            "output",
        )
        assert (n_mul.target, n_mul.args, n_mul.kwargs) == (
            operator.mul,
            (n_x, times),
            {},
        )
    else:
        (n_x, n_sum, n_rep, n_out) = f_graph.nodes
        assert (n_x.op, n_sum.op, n_rep.op, n_out.op) == (
            "placeholder",
            "call_function",
            "call_function",
            "output",
        )
        assert (n_sum.target, n_sum.args, n_sum.kwargs) == (
            jet.utils.sum_vmapped,
            (n_x,),
            {"pos": pos2 if pos1 > pos2 else pos2 - 1},
        )
        assert (n_rep.target, n_rep.args, n_rep.kwargs) == (
            jet.utils.replicate,
            (n_sum, times),
            {"pos": pos1 - 1 if pos1 > pos2 else pos1},
        )

    f_mod_x = f_mod(x)
    assert f_x.shape == f_mod_x.shape
    assert f_x.allclose(f_mod_x)


@mark.parametrize("pos1, pos2", POS1_POS2)
def test_simplify_sum_of_replicates(pos1: int, pos2: int):
    """Test simplification when summing two replicated tensors.

    This test ensures that the simplification logic is agnostic to the `pos` argument
    of the replicate function.

    Args:
        pos1: Position along which to replicate the first tensor.
        pos2: Position along which to replicate the second tensor.
    """
    # generate synthetic input and graph
    manual_seed(0)
    d = 5
    x = rand(d, d)

    def f(x: Tensor) -> Tensor:
        """Replicate the input tensor along two different axes, then sum.

        Args:
            x: Input tensor to be replicated and summed.

        Returns:
            The result of replicating `x` along two axes and summing them.
        """
        y = jet.utils.replicate(x + 1, d, pos=pos1)
        z = jet.utils.replicate(x, d, pos=pos2)
        return y + z

    f_mod = GraphModule({}, JetTracer().trace(f))
    f_graph = deepcopy(f_mod.graph)

    # perform the simplification
    f_mod_simplified = simplify(f_mod)
    f_graph_simplified = f_mod_simplified.graph

    if pos1 != pos2:
        # verify the graph remains unchanged
        assert len(f_graph.nodes) == len(f_graph_simplified.nodes)
        for n1, n2 in zip(f_graph.nodes, f_graph_simplified.nodes):
            assert (n1.op, n1.target, len(n1.args)) == (n2.op, n2.target, len(n2.args))
            for a1, a2 in zip(n1.args, n2.args):
                if isinstance(a1, Node) and isinstance(a2, Node):
                    assert (a1.op, a1.target, a1.name) == (a2.op, a2.target, a2.name)
                else:
                    assert a1 == a2
            assert n1.kwargs == n2.kwargs

    else:
        # verify that the graph was simplified
        (n_x, n_inc, n_add, n_rep, n_out) = f_graph_simplified.nodes
        assert (n_x.op, n_inc.op, n_add.op, n_rep.op, n_out.op) == (
            "placeholder",
            "call_function",
            "call_function",
            "call_function",
            "output",
        )
        assert (n_inc.target, n_add.target, n_rep.target) == (
            operator.add,
            operator.add,
            jet.utils.replicate,
        )
        assert (n_inc.args, n_inc.kwargs) == ((n_x, 1), {})
        assert (n_add.args, n_add.kwargs) == ((n_inc, n_x), {})
        assert (n_rep.args, n_rep.kwargs) == ((n_add, d), {"pos": pos1})

    # verify that simplification does not change the result
    assert f(x).allclose(f_mod_simplified(x))
