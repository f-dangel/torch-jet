"""Script for profiling the impact of torch.compile."""

from functools import partial

from torch import compile, cuda, device, manual_seed, randn, vmap, zeros
from torch.nn import Linear, Sequential, Tanh

from jet.bilaplacian import bilaplacian as jet_bilaplacian
from jet.exp.utils import measure_peak_memory, measure_time
from jet.laplacian import laplacian as jet_laplacian
from jet.simplify import common_subexpression_elimination

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
        factory = {"laplacian": jet_laplacian, "bilaplacian": jet_bilaplacian}[op]

        for stochastic in [False, True]:
            randomization = ("normal", 30) if stochastic else None
            dummy_x = zeros(D, device=dev)
            lap = factory(model, dummy_x, randomization=randomization)
            print(f"\n{20 * '-'} {op=}, {randomization=} {20 * '-'}")

            # Simplify with CSE + DCE
            common_subexpression_elimination(lap.graph)
            lap.recompile()
            f_simple = lap
            print("After simplification:", len(list(f_simple.graph.nodes)))

            # Vmap over data points
            randomness = "error" if randomization is None else "different"
            f_simple = vmap(f_simple, randomness=randomness)

            # [NO COMPILATION] Benchmark memory and time
            measure_peak_memory(partial(f_simple, X), "collapsed", is_cuda)
            measure_time(partial(f_simple, X), "collapsed", is_cuda)

            print("--")

            # [COMPILATION] Now use compilation
            f_compiled = compile(f_simple)
            measure_peak_memory(partial(f_compiled, X), "collapsed+compile", is_cuda)
            measure_time(partial(f_compiled, X), "collapsed+compile", is_cuda)
