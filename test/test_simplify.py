"""Test simplification mechanism of compute graphs captured with `torch.fx`."""

from test.test___init__ import CASES, compare_jet_results
from test.test_laplacian import Laplacian, laplacian
from test.utils import Sin, measure_time
from typing import Callable, Dict, Tuple

import pytest
from torch import Tensor, manual_seed, rand, rand_like
from torch.fx import GraphModule, symbolic_trace, wrap
from torch.nn import Linear, Module, Sequential, Sigmoid, Tanh

from jet import JetTracer, jet, rev_jet
from jet.simplify import simplify
from jet.utils import replicate

# tell `torch.fx` to trace `replicate` as one node
wrap(replicate)


class Replicate(Module):
    """Layer that replicates the forward pass of a given net.

    Also, this module is trace-able and the trace will correspond to the
    graph that the `jet` function transforms, which means it is close to
    the forward pass in the compute graph of a `jet`.
    """

    def __init__(self, net: Module, num_replica: int, x_ndim: int) -> None:
        """Initialize the `Replicate` module.

        Args:
            net: The net to replicate. Must process its input like `vmap` if
                the input is a batch of tensors.
            num_replica: The number of replicas to create. This number only has
                to be passed because `torch.fx`'s tracing mechanism cannot infer it.
            x_ndim: The number of dimensions of the input tensor. This number only
                has to be passed because `torch.fx`'s tracing mechanism cannot infer it.
        """
        super().__init__()
        # Trace the net so we get the same representation as inside the `jet` function
        graph = JetTracer().trace(net)
        self.traced_net = GraphModule(net, graph)
        self.num_replica = num_replica
        self.x_ndim = x_ndim

    def forward(self, x: Tensor) -> Tensor:
        """Replicate the input tensor, then compute a forward pass through the net.

        Args:
            x: The input tensor.

        Returns:
            The replicated output tensor.
        """
        X = replicate(x, self.num_replica, self.x_ndim)
        return self.traced_net(X)


def test_propagate_replication():
    """Test the propagation of replication node through a compute graph.

    It is always better to compute then replicate, rather than carry out
    redundant computations on a replicated tensor.
    """
    manual_seed(0)
    D = 512
    x = rand(D).double()
    mlp = Sequential(
        Linear(D, D, bias=False),
        # Tanh(),
        # Linear(D, D, bias=True),
        # Sin(),
        # Linear(D, D, bias=True),
        # Sigmoid(),
    ).double()
    num_replicas = 128
    mod = Replicate(mlp, num_replicas, x.ndim)

    # check that the `Replicate` module works as expected
    ref = mlp(x).unsqueeze(0).expand(num_replicas, -1)
    assert ref.allclose(mod(x), atol=1e-7)
    print("Replicate module works as expected.")

    # check that the `Replicate` module can be traced and simplified
    mod_fast = symbolic_trace(mod)
    mod_fast = simplify(mod_fast, verbose=True)
    assert ref.allclose(mod_fast(x))
    print("After simplifying, Replicate module still behaves the same.")

    # make sure the `replicate` node made it to the end
    last_node = list(mod_fast.graph.nodes)[-2]  # -1 is the output node
    assert last_node.op == "call_function"
    assert last_node.target == replicate

    # compare time and verify speed-ups empirically
    mean_slow, _ = measure_time(lambda: mod(x), "Replicate module")
    mean_fast, _ = measure_time(lambda: mod_fast(x), "Replicate module (simplified)")
    mean_ref, _ = measure_time(
        lambda: mlp(x).unsqueeze(0).expand(num_replicas, -1), "Manually simplified"
    )
    # make sure that manually simplified and automatically simplified differ by
    # at most a factor of 2 in mean run time
    assert 2 * min(mean_fast, mean_ref) > max(mean_fast, mean_ref)
    # make sure that the automatically simplified version is faster than the original
    # achieving at least 10% of the expected speed-up of `num_replicas`
    expected_speedup = num_replicas
    measured_speedup = mean_slow / mean_fast
    assert measured_speedup > 0.1 * expected_speedup


class ReplicateJet(Module):
    """Layer that replicates the jet of a given net and is trace-able."""

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        num_replica: int,
        x_ndim: int,
        k: int,
        verbose: bool = False,
    ) -> None:
        """Initialize the `Replicate` module.

        Args:
            net: The net to replicate. Must process its input like `vmap` if
                the input is a batch of tensors.
            num_replica: The number of replicas to create. This number only has
                to be passed because `torch.fx`'s tracing mechanism cannot infer it.
            x_ndim: The number of dimensions of the input tensor. This number only
                has to be passed because `torch.fx`'s tracing mechanism cannot infer it.
        """
        super().__init__()
        self.jet_f = jet(f, k, vmap=True, verbose=verbose)
        self.k = k
        self.num_replica = num_replica
        self.x_ndim = x_ndim

    def forward(self, s: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """Replicate the input tensor, then compute a forward pass through the net.

        Args:
            x: The input tensor.

        Returns:
            The replicated output tensor.
        """
        x, vs = s[0], s[1:]
        X = replicate(x, self.num_replica, self.x_ndim)
        VS = [replicate(vs[k], self.num_replica, self.x_ndim) for k in range(self.k)]
        return self.jet_f(X, *VS)


@pytest.mark.parametrize("config", CASES[:5], ids=lambda c: c["id"])
def test_propagate_replication_jet(
    config: Dict[str, Callable], num_replicas: int = 128
):
    """Test the propagation of replication node through a compute graph of a jet."""
    manual_seed(0)
    f = config["f"]
    x = config["primal"]().double()
    vs = [v.double() for v in config["coefficients"]()]
    k = len(vs)

    # use a single jet, then replicate
    jet_f = rev_jet(f, order=k)
    jet_f_result = jet_f(x, *vs)
    jet_f_result = tuple(replicate(j, num_replicas, x.ndim) for j in jet_f_result)

    # # use a replicated jet
    mod = ReplicateJet(f, num_replicas, x.ndim, k)
    mod_result = mod((x, *vs))

    compare_jet_results(jet_f_result, mod_result)
    print("ReplicateJet module works as expected.")

    # simplify the traced module
    mod_fast = symbolic_trace(mod)
    mod_fast = simplify(mod_fast, verbose=True)
    mod_result = mod_fast((x, *vs))
    compare_jet_results(jet_f_result, mod_result)

    # make sure the `replicate` nodes made it to the end
    last_node = list(mod_fast.graph.nodes)[-1]
    outputs = last_node.args[0]
    for out in outputs:
        assert out.op == "call_function"
        assert out.target == replicate
    num_replicates = [
        n
        for n in mod_fast.graph.nodes
        if n.op == "call_function" and n.target == replicate
    ]
    assert len(num_replicates) == k + 1


# def test_propagate_replication_laplacian():
#     """Test the propagation of replication node through a compute graph of a Laplacian.

#     It is always better to compute then replicate, rather than carry out
#     redundant computations on a replicated tensor.
#     """
#     manual_seed(0)
#     D = 1
#     x = rand(D).double()
#     mlp = Sequential(
#         # Linear(D, 1, bias=False),
#         Tanh(),
#         # Sin(),
#         # Sin(),
#         # Linear(D, D, bias=True),
#         # Sin(),
#         # Linear(D, D, bias=True),
#         # Sin(),
#         # Sigmoid(),
#         # Linear(D, 1, bias=False),
#     ).double()

#     lap = Laplacian(mlp, x)
#     ref = laplacian(mlp, x)
#     _, _, lap_jet = lap(x)
#     assert ref.allclose(lap_jet)
#     print("Laplacian via jet matches Laplacian via functorch.")

#     # trace the Laplacian module
#     mod = symbolic_trace(lap)
#     # make sure the traced module still works
#     _, _, lap_jet = mod(x)
#     assert ref.allclose(lap_jet)
#     print("Laplacian via functorch matches Laplacian via traced module.")

#     # try simplifying the traced module
#     mod_fast = simplify(mod, verbose=True)
#     _, _, lap_fast = mod_fast(x)
#     assert ref.allclose(lap_fast)
#     print("Laplacian via functorch matches Laplacian via simplified module.")

#     # make sure the `replicate` node made it to the end of the forward pass
#     output_node = list(mod_fast.graph.nodes)[-1]  # -1 is the output node
#     ((forward_node, _, _),) = output_node.args
#     assert forward_node.op == "call_function"
#     assert forward_node.target == replicate
#     assert False

#     # print(last_node.args)
#     # last_node = list(mod_fast.graph.nodes)[-2]  # -1 is the output node
#     #
#     # print(mod_fast.graph)
#     # assert False

#     # # compare time and verify speed-ups empirically
#     # mean_slow, _ = measure_time(lambda: mod(x), "Replicate module")
#     # mean_fast, _ = measure_time(lambda: mod_fast(x), "Replicate module (simplified)")
#     # mean_ref, _ = measure_time(
#     #     lambda: mlp(x).unsqueeze(0).expand(num_replicas, -1), "Manually simplified"
#     # )
#     # # make sure that manually simplified and automatically simplified differ by
#     # # at most a factor of 2 in mean run time
#     # assert 2 * min(mean_fast, mean_ref) > max(mean_fast, mean_ref)
#     # # make sure that the automatically simplified version is faster than the original
#     # # achieving at least 10% of the expected speed-up of `num_replicas`
#     # expected_speedup = num_replicas
#     # measured_speedup = mean_slow / mean_fast
#     # assert measured_speedup > 0.1 * expected
