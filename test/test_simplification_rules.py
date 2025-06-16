"""Test simplification rules."""

import operator

from pytest import mark
from torch import Tensor, manual_seed, rand
from torch.fx import Graph, GraphModule
from torch.nn import Module

import jet.utils
from jet import JetTracer


def push_replicate_through_sum_vmapped(graph: Graph):
    """Push replicate nodes through sum_vmapped nodes in the graph (in-place).

    Args:
        graph: The graph to be modified
    """
    for n_rep in list(graph.nodes):
        if n_rep.op != "call_function" or n_rep.target != jet.utils.replicate:
            continue

        children = [n for n in graph.nodes if n_rep in n.all_input_nodes]
        if len(children) != 1:
            continue

        (n_sum,) = children
        if n_sum.op != "call_function" or n_sum.target != jet.utils.sum_vmapped:
            continue

        pos_sum: int = n_sum.kwargs["pos"]
        pos_rep: int = n_rep.kwargs["pos"]

        if pos_sum == pos_rep:
            # insert a multiplication node before the replicate node
            with graph.inserting_before(n_rep):
                n_mul = graph.call_function(operator.mul, args=n_rep.args)
            n_sum.replace_all_uses_with(n_mul)

        else:
            # insert a replication node after the sum node
            with graph.inserting_after(n_sum):
                times = n_rep.args[1]
                n_rep_new = graph.call_function(
                    jet.utils.replicate,
                    args=(n_sum, times),
                    kwargs={"pos": pos_rep - 1 if pos_rep > pos_sum else pos_rep},
                )
            n_sum.replace_all_uses_with(n_rep_new)
            n_rep_new.args = (n_sum, times)
            n_sum.args = (n_rep.args[0],)
            n_sum.kwargs = {"pos": pos_sum if pos_rep > pos_sum else pos_sum - 1}

        graph.eliminate_dead_code()


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
    push_replicate_through_sum_vmapped(f_graph)
    f_graph.lint()
    f_mod.recompile()

    # verify the graph
    if pos1 == pos2:
        (n_x, n_mul, n_out) = f_graph.nodes
        assert (n_x.op, n_mul.op, n_out.op) == (
            "placeholder",
            "call_function",
            "output",
        )
        assert n_mul.target == operator.mul
        assert n_mul.args == (n_x, times)
        assert n_mul.kwargs == {}
    else:
        (n_x, n_sum, n_rep, n_out) = f_graph.nodes
        assert (n_x.op, n_sum.op, n_rep.op, n_out.op) == (
            "placeholder",
            "call_function",
            "call_function",
            "output",
        )
        assert n_sum.target == jet.utils.sum_vmapped
        assert n_rep.target == jet.utils.replicate
        assert n_sum.args == (n_x,)
        assert n_sum.kwargs == {"pos": pos2 if pos1 > pos2 else pos2 - 1}
        assert n_rep.args == (n_sum, times)
        assert n_rep.kwargs == {"pos": pos1 - 1 if pos1 > pos2 else pos1}

    f_mod_x = f_mod(x)
    assert f_x.shape == f_mod_x.shape
    assert f_x.allclose(f_mod_x)
