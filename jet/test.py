from torch import compile, cuda, device, manual_seed, randn
from torch.fx import symbolic_trace
from torch.nn import Linear, Sequential, Tanh

from jet.bilaplacian import Bilaplacian, RandomizedBilaplacian
from jet.exp.utils import measure_peak_memory, measure_time
from jet.laplacian import Laplacian, RandomizedLaplacian
from jet.simplify import simplify

if __name__ == "__main__":
    is_cuda = cuda.is_available()
    dev = device("cuda" if is_cuda else "cpu")
    print(f"Running on device {str(dev)}")

    manual_seed(0)
    model = Sequential(
        Linear(5, 768),
        Tanh(),
        Linear(768, 768),
        Tanh(),
        Linear(768, 512),
        Tanh(),
        Linear(512, 512),
        Tanh(),
        Linear(512, 1),
    ).to(dev)
    X = randn(2048, 5).to(dev)

    for op in ["laplacian", "bilaplacian"]:
        for stochastic in [True, False]:
            if stochastic:
                cls = {
                    "laplacian": RandomizedLaplacian,
                    "bilaplacian": RandomizedBilaplacian,
                }[op]
                if op == "laplacian":
                    print("\nRandomized Laplacian")
                else:
                    print("\nRandomized Bilaplacian")
                lap = cls(
                    model, X, is_batched=True, num_samples=30, distribution="normal"
                )
            else:
                cls = {"laplacian": Laplacian, "bilaplacian": Bilaplacian}[op]
                if op == "laplacian":
                    print("\nExact Laplacian")
                else:
                    print("\nExact Bilaplacian")
                lap = cls(model, X, is_batched=True)

            # print number of computation graph nodes
            f_before = symbolic_trace(lap)
            print("Before simplification:", len(list(f_before.graph.nodes)))
            f_simple1 = simplify(symbolic_trace(lap), pull_sum_vmapped=False)
            print("Naive after simplification:", len(list(f_simple1.graph.nodes)))
            f_simple2 = simplify(symbolic_trace(lap), pull_sum_vmapped=True)
            print("Collapsed after simplification:", len(list(f_simple2.graph.nodes)))

            # Benchmark memory and time
            peakmem = measure_peak_memory(lambda: f_simple1(X), "naive", is_cuda)
            peakmem = measure_peak_memory(lambda: f_simple2(X), "collapsed", is_cuda)
            measure_time(lambda: f_simple1(X), "naive", is_cuda)
            measure_time(lambda: f_simple2(X), "collapsed", is_cuda)
            # Now use compilation
            f_simple1, f_simple2 = compile(f_simple1), compile(f_simple2)
            peakmem = measure_peak_memory(
                lambda: f_simple1(X), "naive+compile", is_cuda
            )
            peakmem = measure_peak_memory(
                lambda: f_simple2(X), "collapsed+compile", is_cuda
            )
            measure_time(lambda: f_simple1(X), "naive+compile", is_cuda)
            measure_time(lambda: f_simple2(X), "collapsed+compile", is_cuda)
