from torch import cuda, manual_seed, randn
from torch.fx import symbolic_trace
from torch.nn import Linear, Sequential, Tanh

from jet.bilaplacian import RandomizedBilaplacian
from jet.exp.utils import measure_peak_memory
from jet.simplify import simplify

dev = device("cuda" if cuda.is_available() else "cpu")


manual_seed(0)
model = Sequential(
    Linear(50, 768),
    Tanh(),
    Linear(768, 768),
    Tanh(),
    Linear(768, 512),
    Tanh(),
    Linear(512, 512),
    Tanh(),
    Linear(512, 1),
).to(dev)
X = randn(2048, 50).to(dev)
num_samples = 30
distribution = "normal"

rand_bilap = RandomizedBilaplacian(
    model, X, is_batched=True, num_samples=num_samples, distribution="normal"
)
# print(rand_bilap(X).shape)

f_traced = symbolic_trace(rand_bilap)
print("Before simplification:", len(list(f_traced.graph.nodes)))

f_simple1 = simplify(f_traced, pull_sum_vmapped=False)
print("After simplification:", len(list(f_simple1.graph.nodes)))
# print(f_simple1.graph)

f_simple2 = simplify(f_traced, pull_sum_vmapped=True)
print("After simplification:", len(list(f_simple2.graph.nodes)))
# print(f_simple2.graph)

if __name__ == "__main__":
    peakmem = measure_peak_memory(lambda: f_simple1(X), "naive", False)
    peakmem = measure_peak_memory(lambda: f_simple2(X), "collapsed", False)
