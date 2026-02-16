"""Script for profiling the impact of torch.compile."""

from functools import partial

from torch import compile, cuda, device, manual_seed, randn, vmap, zeros
from torch.nn import Linear, Sequential, Tanh

from jet.bilaplacian import Bilaplacian
from jet.exp.utils import measure_peak_memory, measure_time
from jet.laplacian import Laplacian
from jet.simplify import simplify
from jet.tracing import capture_graph

if __name__ == "__main__":
    is_cuda = cuda.is_available()
    dev = device("cuda" if is_cuda else "cpu")
    print(f"Running on {dev=}")

    manual_seed(0)
    N = 2048
    D = 5

    model = Sequential(
        Linear(D, 768),
        Tanh(),
        Linear(768, 768),
        Tanh(),
        Linear(768, 512),
        Tanh(),
        Linear(512, 512),
        Tanh(),
        Linear(512, 1),
    ).to(dev)
    X = randn(N, D).to(dev)

    for op in ["laplacian", "bilaplacian"]:
        cls = {"laplacian": Laplacian, "bilaplacian": Bilaplacian}[op]

        for stochastic in [False, True]:
            randomization = ("normal", 30) if stochastic else None
            dummy_x = zeros(D, device=dev)
            lap = cls(model, dummy_x, randomization=randomization)
            print(f"\n{20 * '-'} {op=}, {randomization=} {20 * '-'}")

            # print number of computation graph nodes
            f_before = capture_graph(lap)  # noqa: B023
            print("Before simplification:", len(list(f_before.graph.nodes)))
            f_simple1 = simplify(lap, pull_sum=False)
            print("Naive after simplification:", len(list(f_simple1.graph.nodes)))
            f_simple2 = simplify(lap, pull_sum=True)
            print("Collapsed after simplification:", len(list(f_simple2.graph.nodes)))

            print("\n--")

            # Vmap over data points
            randomness = "error" if randomization is None else "different"
            f_simple1 = vmap(f_simple1, randomness=randomness)
            f_simple2 = vmap(f_simple2, randomness=randomness)

            # [NO COMPILATION] Benchmark memory and time
            measure_peak_memory(partial(f_simple1, X), "naive", is_cuda)
            measure_peak_memory(partial(f_simple2, X), "collapsed", is_cuda)
            measure_time(partial(f_simple1, X), "naive", is_cuda)
            measure_time(partial(f_simple2, X), "collapsed", is_cuda)

            print("--")

            # [COMPILATION] Now use compilation
            f_simple1, f_simple2 = compile(f_simple1), compile(f_simple2)
            measure_peak_memory(partial(f_simple1, X), "naive+compile", is_cuda)
            measure_peak_memory(partial(f_simple2, X), "collapsed+compile", is_cuda)
            measure_time(lambda: f_simple1(X), "naive+compile", is_cuda)  # noqa: B023
            measure_time(partial(f_simple2, X), "collapsed+compile", is_cuda)
