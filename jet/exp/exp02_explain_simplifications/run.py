"""Print the compute graph and its simplification for the Laplacian of a small net."""

from os import makedirs, path
from typing import Tuple

from torch import Tensor, eye, manual_seed, rand, sin, zeros, zeros_like
from torch.fx import GraphModule, passes, wrap
from torch.nn import Linear, Module, Sequential

from jet import JetTracer, jet
from jet.exp.exp01_benchmark_laplacian.execute import laplacian_function
from jet.simplify import simplify
from jet.utils import replicate, sum_vmapped

HEREDIR = path.dirname(path.abspath(__file__))
FIGDIR = path.join(HEREDIR, "figures")
makedirs(FIGDIR, exist_ok=True)

# tell `torch.fx` to trace `replicate` as one node (required for simplification)
wrap(replicate)
# tell `torch.fx` to trace `sum_vmapped` as one node (required for simplification)
wrap(sum_vmapped)

manual_seed(0)  # make deterministic
batch_size = 3
D = 5
X = rand(batch_size, D).double()


class Sin(Module):
    """Sine activation layer."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the sine activation layer.

        Args:
            x: Input tensor.

        Returns:
            The output of the layer.
        """
        return sin(x)


def visualize_graph(mod: GraphModule, name: str):
    """Visualize the compute graph of a module.

    Args:
        mod: Module to visualize.
        name: Name of the module and filename of the generated svg file.
    """
    g = passes.graph_drawer.FxGraphDrawer(mod, name)
    filename = path.join(FIGDIR, f"{name}.svg")
    print(f"Saving compute graph to {filename}")
    g.get_dot_graph().write_svg(filename)


# A simple MLP: One fully-connected layer followed by a sin activation
mlp = Sequential(Linear(D, 1), Sin()).double()

# 1) Obtain and print the compute graph of the MLP
mlp_mod = GraphModule(mlp, JetTracer().trace(mlp))
print("Compute graph of the neural net:")
print(mlp_mod.graph)
visualize_graph(mlp_mod, "forward")

# Compute the Laplacian via nested first-order AD for reference
rev_lap = laplacian_function(mlp, X, is_batched=True, strategy="hessian_trace")()

# 2) Obtain and print the compute graph for computing the 2-jet of the MLP
mlp_2jet = jet(mlp, 2)
print("\nCompute graph of the 2-jet of the neural net (before expansion):")
print(mlp_2jet.graph)
visualize_graph(mlp_2jet, "2jet")

# The above graph contains the new overloaded primitives. We can expand them by
# tracing again
mlp_2jet = GraphModule(mlp_2jet, JetTracer().trace(mlp_2jet))
print("\nCompute graph of the 2-jet of the neural net (after expansion):")
print(mlp_2jet.graph)

# TODO This currently does not work because the graph contains einsums
# (https://github.com/pytorch/pytorch/issues/147884)
# visualize_graph(mlp_2jet, "2jet_expanded")

# Compute the Laplacian by looping over 2-jets (one diagonal element per iteration),
# then summing
diag_elements = []
for d in range(D):
    V1 = zeros_like(X)
    V1[:, d] = 1.0  # for each data point, set the d-th entry to one
    V2 = zeros_like(X)
    _, _, d2i = mlp_2jet(X, V1, V2)
    diag_elements.append(d2i)
lap = sum(diag_elements)
assert rev_lap.allclose(lap)

# 3) Obtain and print the graph for computing multiple 2-jets of the MLP in parallel


# NOTE one annoyance with PyTorch's tracing mechanism is that we have to wrap this
# inside a module to be trace-able
class VmapLaplacian(Module):
    """Trace-able module that computes the Laplacian by parallel independent 2-jets."""

    def __init__(self):
        """Initialize the module."""
        super().__init__()
        self.mlp = mlp
        self.mlp_vmap_2jet = jet(mlp, 2, vmap=True)

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the Laplacian of the function at the input tensor.

        Args:
            X: Input tensor.

        Returns:
            Tuple containing the replicated function value, the Jacobian, and the
            Laplacian.
        """
        X_copy = replicate(X, D)  # shape (D, batch_size, D) containing D copies of X
        V1 = eye(D).unsqueeze(1).expand(D, batch_size, D).double()
        V2 = zeros(D, batch_size, D).double()

        F0, F1, F2 = self.mlp_vmap_2jet(X_copy, V1, V2)
        return F0, F1, sum_vmapped(F2)


vmap_lap = VmapLaplacian()
vmap_lap = GraphModule(vmap_lap, JetTracer().trace(vmap_lap))
print("\nCompute graph of the Laplacian via independent vmapped 2-jets:")
print(vmap_lap.graph)

# TODO This currently does not work because the graph contains einsums
# (https://github.com/pytorch/pytorch/issues/147884)
# visualize_graph(vmap_lap, "vmap_laplacian")

_, _, lap = VmapLaplacian()(X)
assert rev_lap.allclose(lap)

# 4) Note that approach 3) to compute the Laplacian repeats the forward pass 5 times
# because we pass a copy of X to conform to the API of vmap. The first graph
# simplification, we can leverage is to push forward the replication, i.e. if we have
# f(replicate(X)), we can rewrite into replicate(f(X)).
# Obtain and print the graph after simplifying replicated computations.
no_replicates_lap = simplify(vmap_lap, push_replicate=True, pull_sum_vmapped=False)
print("\nCompute graph of the Laplacian after removing replicated computations:")
# Notice how there are no replicate nodes anymore except for the output node
print(no_replicates_lap.graph)

# TODO This currently does not work because the graph contains einsums
# (https://github.com/pytorch/pytorch/issues/147884)
# visualize_graph(no_replicates_lap, "no_replicates_laplacian")
_, _, lap = no_replicates_lap(X)
assert rev_lap.allclose(lap)

# 5) Note that approach 4) computes the Laplacian's diagonal elements in parallel,
# then sums them. But we can propagate the sum up the graph!
fully_simplified_lap = simplify(
    no_replicates_lap,
    push_replicate=False,  # we already did that in step 4
    pull_sum_vmapped=True,
)
print("\nCompute graph of the Laplacian after propagating up the sum:")
# Note how the `sum_vmapped` node disappeared from the graph from step 4
# If you print the module's _tensor_constant0 and _tensor_constant1 tensor shapes
# you will see that the _tensor_constant1 (second-order coefficients )tensor reduced
# in size because it was summed over.
print(fully_simplified_lap.graph)

# TODO This currently does not work because the graph contains einsums
# (https://github.com/pytorch/pytorch/issues/147884)
# visualize_graph(fully_simplified_lap, "fully_simplified_laplacian")

_, _, lap = fully_simplified_lap(X)
assert rev_lap.allclose(lap)
