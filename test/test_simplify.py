"""Test simplification mechanism of compute graphs captured with `torch.fx`.

There are two kinds of tests:

1. [Replicating functions] Take a function f: x -> f(x).
   i. Construct the replicated function f_rep: x -> f(replicate(x))
   ii. Simplify the compute graph of f_rep. This should yield the compute graph of
       x -> replicate(f(x)).

2. [Replicating jets] Take a function f: x -> f(x).
    i. Construct the jet of f, jet_f: x, v1, v2, ... -> jet_f(x, v1, v2, ...)
    ii. Construct the replicated jet of f, jet_f_rep:
        x, v1, v2, ... -> jet_f(replicate(x), replicate(v1), replicate(v2), ...)
    iii. Simplify the compute graph of jet_f_rep. This should yield the compute graph of
        x, v1, v2, ... -> replicate(jet_f(x, v1, v2, ...)).
"""

from test.test___init__ import CASES, compare_jet_results
from typing import Callable, Dict

import pytest
from torch import Tensor, manual_seed
from torch.fx import Graph, GraphModule, symbolic_trace, wrap
from torch.nn import Module

from jet import JetTracer, jet, rev_jet
from jet.simplify import RewriteReplicate, simplify
from jet.utils import (
    PrimalAndCoefficients,
    ValueAndCoefficients,
    WrapperModule,
    replicate,
)

# tell `torch.fx` to trace `replicate` as one node
wrap(replicate)


class Replicate(Module):
    """Layer that replicates the forward pass of a function.

    This module is trace-able and the trace will correspond to the
    graph that the `jet` function transforms, which means it is close to
    the forward pass in the compute graph of a `jet`.
    """

    def __init__(self, f: Callable[[Tensor], Tensor], num_replica: int) -> None:
        """Initialize the `Replicate` module.

        Args:
            f: The function to replicate. Must process its input like `vmap` if
                the input is a batch of tensors.
            num_replica: The number of replicas to create.
        """
        super().__init__()
        # Wrap the function in a module if it is not already a module.
        # We want to always produce an executable `torch.fx.GraphModule`.
        if not isinstance(f, Module):
            f = WrapperModule(f)
        # Trace the function so we get the same representation as inside `jet`
        graph = JetTracer().trace(f)
        self.traced_f = GraphModule(f, graph)
        self.num_replica = num_replica

    def forward(self, x: Tensor) -> Tensor:
        """Replicate the input tensor, then compute a forward pass through the function.

        Args:
            x: The input tensor.

        Returns:
            The replicated output tensor.
        """
        X = replicate(x, self.num_replica)
        return self.traced_f(X)


def ensure_outputs_only_replicate(graph: Graph):
    """Make sure the compute graph outputs only `replicate` nodes.

    Args:
        graph: The compute graph to check.
    """
    output = list(graph.nodes)[-1]  # -1 is the output node
    parents = [n for n in graph.nodes if n in output.all_input_nodes]
    for parent in parents:
        assert RewriteReplicate.is_replicate(parent)


def ensure_num_replicates(graph: Graph, num_replicates: int):
    """Make sure the compute graph has the specified number of `replicate` nodes.

    Args:
        graph: The compute graph to check.
        num_replicates: The number of `replicate` nodes to check for.
    """
    replicates = [n for n in graph.nodes if RewriteReplicate.is_replicate(n)]
    assert len(replicates) == num_replicates


@pytest.mark.parametrize("config", CASES, ids=lambda c: c["id"])
def test_propagate_replication(config: Dict[str, Callable], num_replicas: int = 3):
    """Test the propagation of replication node through a compute graph.

    It is always better to compute then replicate, rather than carry out
    redundant computations on a replicated tensor.

    Args:
        config: The configuration of the test case.
        num_replicas: The number of replicas to create. Default: `3`.
    """
    manual_seed(0)
    f = config["f"]
    x = config["primal"]().double()
    f_rep = Replicate(f, num_replicas)

    # check that the `Replicate` module works as expected
    ref = replicate(f(x), num_replicas)
    assert ref.allclose(f_rep(x), atol=1e-7)
    print("Replicate module works as expected.")

    # check that the `Replicate` module can be traced and simplified
    fast = symbolic_trace(f_rep)
    fast = simplify(fast, verbose=True)
    assert ref.allclose(fast(x))
    print("After simplifying, Replicate module still behaves the same.")

    # make sure the `replicate` node made it to the end
    ensure_outputs_only_replicate(fast.graph)
    # make sure there are no other `replicate` nodes in the graph
    ensure_num_replicates(fast.graph, 1)


class ReplicateJet(Module):
    """Layer that replicates the jet of a given function and is trace-able."""

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        num_replica: int,
        k: int,
        verbose: bool = False,
    ) -> None:
        """Initialize the `ReplicateJet` module.

        Args:
            f: The function whose jet to replicate. Must process its input like `vmap`
                if the input is a batch of tensors.
            num_replica: The number of replicas to create.
            k: The order of the Taylor expansion.
            verbose: Whether to print debug information when creating the jet.
                Default: `False`.
        """
        super().__init__()
        self.jet_f = jet(f, k, vmap=True, verbose=verbose)
        self.k = k
        self.num_replica = num_replica

    def forward(self, s: PrimalAndCoefficients) -> ValueAndCoefficients:
        """Replicate the input tensor and coefficients, then compute the jet.

        Args:
            s: The input tensor and Taylor coefficients.

        Returns:
            The replicated output tensor.
        """
        x, vs = s[0], s[1:]
        X = replicate(x, self.num_replica)
        VS = [replicate(vs[k], self.num_replica) for k in range(self.k)]
        return self.jet_f(X, *VS)


@pytest.mark.parametrize("config", CASES, ids=lambda c: c["id"])
def test_propagate_replication_jet(config: Dict[str, Callable], num_replicas: int = 3):
    """Test the propagation of replication nodes through a compute graph of a jet.

    It is always better to compute then replicate, rather than carry out
    redundant computations on a replicated tensor.

    Args:
        config: The configuration of the test case.
        num_replicas: The number of replicas to create. Default: `3`.
    """
    manual_seed(0)
    f = config["f"]
    x = config["primal"]().double()
    vs = [v.double() for v in config["coefficients"]()]
    k = len(vs)

    # use a single jet, then replicate
    jet_f = rev_jet(f, order=k)
    jet_f_result = jet_f(x, *vs)
    jet_f_result = tuple(replicate(j, num_replicas) for j in jet_f_result)

    # # use a replicated jet
    mod = ReplicateJet(f, num_replicas, k)
    mod_result = mod((x, *vs))

    compare_jet_results(jet_f_result, mod_result)
    print("ReplicateJet module works as expected.")

    # simplify the traced module
    fast = symbolic_trace(mod)
    fast = simplify(fast, verbose=True)
    fast_result = fast((x, *vs))
    compare_jet_results(jet_f_result, fast_result)

    # make sure the `replicate` nodes made it to the end
    ensure_outputs_only_replicate(fast.graph)
    ensure_num_replicates(fast.graph, k + 1)
