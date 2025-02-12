"""Test simplification mechanism of compute graphs captured with `torch.fx`."""

from test.utils import Sin, measure_time

from torch import Tensor, manual_seed, rand
from torch.fx import GraphModule, symbolic_trace, wrap
from torch.nn import Linear, Module, Sequential, Sigmoid, Tanh

from jet import JetTracer
from jet.simplify import simplify
from jet.utils import replicate

# tell `torch.fx` to trace `replicate` as one node
wrap("replicate")


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
        Tanh(),
        Linear(D, D, bias=True),
        Sin(),
        Linear(D, D, bias=True),
        Sigmoid(),
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
    assert last_node.op == "call_function" and last_node.target == replicate

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
