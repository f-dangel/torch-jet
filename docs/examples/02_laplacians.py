"""# Computing Laplacians.

This example demonstrates how to use Taylor mode to compute the Laplacian, a popular
differential operator that shows up in various applications. Our goal is to go from
most pedagogical to most efficient implementation and highlight (i) how to use Taylor
mode and (ii) how to collapse it to get better performance.

Let's get the imports out of our way.
"""

from os import path
from shutil import which
from time import perf_counter
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch import (
    Tensor,
    arange,
    eye,
    manual_seed,
    rand,
    stack,
    vmap,
    zeros_like,
)
from torch import (
    compile as torch_compile,
)
from torch.func import hessian
from torch.nn import Linear, Sequential, Tanh
from tueplots import bundles

import jet
from jet.simplify import simplify
from jet.tracing import capture_graph
from jet.utils import visualize_graph

HEREDIR = path.dirname(path.abspath(__name__))
# We need to store figures here so they will be picked up in the built doc
GALLERYDIR = path.join(path.dirname(HEREDIR), "generated", "gallery")

_ = manual_seed(0)  # make deterministic

# %%
#
### Definition
#
# Throughout this example, we will consider a vector-to-scalar function $f: \mathbb{R}^D
# \to \mathbb{R}, \mathbf{x} \mapsto f(\mathbf{x})$, e.g. a neural network that maps a
# $D$-dimensional input to a single output.
# The Laplacian $\Delta f(\mathbf{x})$ of $f$ at $\mathbf{x}$ is the sum of pure second-
# order partial derivatives, i.e. the Hessian trace
# $$
# \Delta f(\mathbf{x})
# := \sum_{d=1}^D
# \frac{\partial^2 f(\mathbf{x})}{\partial [\mathbf{x}]\_d^2}
# = \sum_{d=1}^D
# \left[ \frac{\partial^2 f(\mathbf{x})}{\partial \mathbf{x}^2} \right]\_{d,d}
# = \mathrm{Tr} \left( \frac{\partial^2 f(\mathbf{x})}{\partial \mathbf{x}^2} \right)\,,
# $$
# with $\frac{\partial^2 f(\mathbf{x})}{\partial \mathbf{x}^2} \in
# \mathbb{R}^{D \times D}$ the Hessian of $f$ at $\mathbf{x}$.
#
# In the following we compute the Laplacian of a neural network. Here is the setup:

D = 3
f = Sequential(Linear(D, 128), Tanh(), Linear(128, 64), Tanh(), Linear(64, 1))
x = rand(D)

f_x = f(x)
print(f_x.shape)

# %%
#
### Via `torch.func`
#
# To make sure all approaches we develop yield the correct result, let's compute
# the Laplacian with `torch.func` as ground truth.

hess_func = hessian(f)  # x ↦ ∂²f/∂x²


def compute_hessian_trace_laplacian(x: Tensor) -> Tensor:
    """Compute the Laplacian by taking the trace of the Hessian.

    The Hessian is computed with `torch.func`, which uses forward-over-reverse mode
    (nested) automatic differentiation under the hood.

    Args:
        x: Input tensor of shape [D].

    Returns:
        The Laplacian of shape [1].
    """
    hess = hess_func(x)  # has shape [1, D, D]
    return hess.squeeze(0).trace().unsqueeze(0)  # has shape [1]


hessian_trace_laplacian = compute_hessian_trace_laplacian(x)
print(hessian_trace_laplacian)


# %%
#
### Via Taylor Mode
#
# Now, we will look at different variants to employ Taylor mode to compute the
# Laplacian. We will go from most pedagogical to most efficient.
#
# First, note that we can compute the $d$-th Hessian diagonal element with a
# vector-Hessian-vector product
# $$
# \frac{\partial^2 f(\mathbf{x})}{\partial [\mathbf{x}]\_d^2}
# = \mathbf{e}\_d^\top
# \left( \frac{\partial^2 f(\mathbf{x})}{\partial \mathbf{x}^2} \right)
# \mathbf{e}\_d
# $$
# using $f_{2\text{-jet}}$ with Taylor coefficients $\mathbf{x}_0 = \mathbf{x}$,
# $\mathbf{x}_1 = \mathbf{e}_d$, and $\mathbf{x}_2 = \mathbf{0}$. Then, the
# second output Taylor coefficient will be $f_2 = \mathbf{e}_d^\top
# \left( \frac{\partial^2 f(\mathbf{x})}{\partial \mathbf{x}^2} \right) \mathbf{e}_d$.
#
# Let's set up the jet function:

k = 2
f_jet = jet.jet(f, k, (x,))

# %%
#
#### Pedagogical Way
#
# The easiest way to compute the Laplacian is to loop over the input dimensions and
# compute one element of the Hessian diagonal at a time, then sum the result. Here
# is a function that does that:


def compute_loop_laplacian(x: Tensor) -> Tensor:
    """Compute the Laplacian using Taylor mode and a for loop.

    Args:
        x: Input tensor of shape [D].

    Returns:
        The Laplacian of shape [1].
    """
    x0, x2 = x, zeros_like(x)  # fixed Taylor coefficients

    lap = zeros_like(f_x)  # Laplacian accumulator
    for d in range(D):  # compute the d-th Hessian diagonal element
        x1 = zeros_like(x)
        x1[d] = 1.0
        _, (_, f2) = f_jet((x0,), ((x1,), (x2,)))
        lap += f2

    return lap


loop_laplacian = compute_loop_laplacian(x)
print(loop_laplacian)

# make sure the loop Laplacian matches the torch.func Laplacian
if loop_laplacian.allclose(hessian_trace_laplacian):
    print("Taylor mode Laplacian via loop matches Hessian trace!")
else:
    raise ValueError("Taylor mode Laplacian via loop does not match Hessian trace!")


# %%
#
#### Without `for` Loop
#
# To get rid of the `for` loop, we can use `torch.vmap`, which is composable with out `jet`
# implementation, and compute the $D$ jets in parallel:


def compute_loop_free_laplacian(x: Tensor) -> Tensor:
    """Compute the Laplacian using multiple 2-jets in parallel.

    Args:
        x: Input tensor of shape [D].

    Returns:
        The Laplacian of shape [1].
    """
    x0, x2 = x, zeros_like(x)  # fixed Taylor coefficients
    eval_f2 = lambda x1: f_jet((x0,), ((x1,), (x2,)))[1][1]  # noqa: E731
    vmap_eval_f2 = vmap(eval_f2)

    # generate all basis vectors at once and compute their Hessian diagonal elements
    X1 = eye(D)
    F2 = vmap_eval_f2(X1)

    return F2.sum(dim=0)  # sum the diagonal to obtain the Laplacian


loop_free_laplacian = compute_loop_free_laplacian(x)
print(loop_free_laplacian)

# make sure the loop-free Laplacian matches the torch.func Laplacian
if loop_free_laplacian.allclose(hessian_trace_laplacian):
    print("Taylor mode vmap Laplacian matches Hessian trace!")
else:
    raise ValueError("Taylor mode vmap Laplacian does not match Hessian trace!")

# %%
#
#### Collapsing Taylor Mode
#
# We are already quite close to a high performance Laplacian implementation.
# Now comes the more complicated part, which is hard to understand without reading our
# paper. The idea is that instead of computing 2-jets along the $D$ directions, then
# summing their result, we can rewrite the computational graph to directly propagate
# the summed second-order Taylor coefficients. We call this "collapsing" the Taylor
# mode.
#
# To give a high-level intuition how this works, we will look at the computational
# graph for computing a Laplacian. For that, we will write a function factory
# `make_laplacian` that takes a function and a mock input, and returns a new function
# computing the Laplacian. We can then trace this function and look at its graph.
#
# Here is the function factory:


def make_laplacian(
    f: Callable[[Tensor], Tensor], mock_x: Tensor
) -> Callable[[Tensor], Tensor]:
    """Create a function that computes the Laplacian of f using jets.

    Args:
        f: The function whose Laplacian is computed.
        mock_x: A mock input tensor for tracing. Only the shape matters.

    Returns:
        A function that computes the Laplacian of f at a given input.
    """
    in_shape = mock_x.shape
    in_dim = mock_x.numel()
    jet_f = jet.jet(f, 2, (mock_x,))

    def lap_f(x: Tensor) -> Tensor:
        """Compute the Laplacian.

        Args:
            x: Input tensor of shape [D].

        Returns:
            The Laplacian of shape [1].
        """
        in_meta = {"dtype": x.dtype, "device": x.device}
        X1 = eye(in_dim, **in_meta).reshape(in_dim, *in_shape)
        vmapped = vmap(
            lambda x1: jet_f((x,), ((x1,), (zeros_like(x),))),
            randomness="different",
            out_dims=(None, (0, 0)),
        )
        _, (_, F2) = vmapped(X1)
        return F2.sum(0)

    return lap_f


lap_fn = make_laplacian(f, x)

# %%
#
# We can verify that this indeed computes the correct Laplacian:

fn_laplacian = lap_fn(x)
print(fn_laplacian)

if fn_laplacian.allclose(hessian_trace_laplacian):
    print("Taylor mode Laplacian via function factory matches Hessian trace!")
else:
    raise ValueError(
        "Taylor mode Laplacian via function factory does not match Hessian trace!"
    )

# %%
#
# Now, let's look at three different graphs which will become clear in a moment
# (we evaluated approaches 2 and 3 in our paper).

# Graph 1: Simply capture the function that computes the Laplacian
lap_traced = capture_graph(lap_fn, x)
visualize_graph(
    lap_traced, path.join(GALLERYDIR, "02_laplacian_module.png"), use_custom=True
)
assert hessian_trace_laplacian.allclose(lap_traced(x))

# Graph 2: Standard simplifications (dead code elimination, CSE, but no collapsing)
lap_standard = simplify(lap_fn, x, pull_sum=False)
visualize_graph(
    lap_standard, path.join(GALLERYDIR, "02_laplacian_standard.png"), use_custom=True
)
assert hessian_trace_laplacian.allclose(lap_standard(x))

# Graph 3: Collapsing simplifications — pull summations up the graph to directly
# propagate sums of Taylor coefficients
lap_collapsed = simplify(lap_fn, x, pull_sum=True)
visualize_graph(
    lap_collapsed, path.join(GALLERYDIR, "02_laplacian_collapsed.png"), use_custom=True
)
assert hessian_trace_laplacian.allclose(lap_collapsed(x))

# %%
#
# There is quite some stuff going on here. Let's try to break down the essential
# differences between these three graphs.
#
# First, we can look at the graph sizes:

print(f"1) Captured: {len(lap_traced.graph.nodes)} nodes")
print(f"2) Standard simplifications: {len(lap_standard.graph.nodes)} nodes")
print(f"3) Collapsing simplifications: {len(lap_collapsed.graph.nodes)} nodes")

# %%
#
# We can see that the number of nodes decreases, and this is a first performance
# indicator.

# %%
#
# Next, let's have a look at the computation graphs. In the visualizations below,
# `sum` nodes are highlighted in orange-red to make them easy to track across
# the three graphs.
#
# | Captured | Standard simplifications | Collapsing simplifications |
# |:--------:|:------------------------:|:---------------------------|
# | ![](02_laplacian_module.png) | ![](02_laplacian_standard.png) | ![](02_laplacian_collapsed.png) |
#
# - Graph 1 (**Captured**) is the raw traced graph. It has a `sum`
#   node (orange-red) at the end, which sums the Hessian diagonal elements to
#   obtain the Laplacian.
#
# - Graph 2 (**Standard simplifications**) applies dead code elimination and common
#   subexpression elimination (CSE), but does not collapse Taylor mode. Note that
#   the `sum` node remains at the bottom of the graph.
#
# - Graph 3 (**Collapsing simplifications**) goes one step further and
#   performs the 'collapsing' of Taylor mode we present in our paper.
#
#     The input to the `sum` node at the end of Graph 2 is the output of a linear
#     operation, something like
#     ```python
#     laplacian = sum(linear(Z, weight)) # standard: D matvecs
#     ```
#     *The crucial insight from our paper is that the `sum` can be propagated up the
#     graph!* For our example, we can first sum, then apply the linear operation, as
#     this is mathematically equivalent, but cheaper:
#     ```python
#     laplacian = linear(sum(Z), weight) # collapsed: 1 matvec
#     ```
#     In the graph perspective, we have 'pulled' the `sum` node (orange-red) up the
#     graph. We can repeat this procedure until we run out of possible
#     simplifications. Effectively, this 'collapses' the Taylor coefficients we
#     propagate forward; hence the name 'collapsed Taylor mode'.
#
# We can verify successful collapsing by looking at the tensor constants of the graph
# which represent the forward-propagated coefficients:

print("2) Standard simplifications tensor constants:")
for name, buf in lap_standard.named_buffers():
    print(f"\t{name}: {buf.shape}")

print("3) Collapsing simplifications tensor constants:")
for name, buf in lap_collapsed.named_buffers():
    print(f"\t{name}: {buf.shape}")

# %%
#
# We see that the collapsed Taylor mode graph has a tensor constant whose shape
# is smaller than the one of the standard simplifications graph. This reflects that,
# instead of propagating $D$ second-order Taylor coefficients (shape `[D, D]`),
# collapsed Taylor mode directly propagates their sum (shape `[D]`).
#
### Batching
#
# Before we confirm that collapsing is beneficial for performance, let's add
# one last ingredient. So far, we computed the Laplacian for a single datum $\mathbf{x}$.
# In practise, we often want to compute the Laplacian for a batch of data in parallel.
# *We can trivially achieve this by calling `vmap` on all Laplacian functions!*
#
# In the following, we will compare three implementations, like in the paper:
#
# 1. **Nested first-order AD:** Computes the Hessian with `torch.func` (forward-
#    over-reverse mode AD), then traces it.
#
# 2. **Standard Taylor mode:** Computes each Hessian diagonal element with a 2-jet,
#    then sums the results.
#
# 3. **Collapsed Taylor mode:** Same as 2, but collapses the 2-jets.

compute_batched_nested_laplacian = vmap(compute_hessian_trace_laplacian)
compute_batched_standard_laplacian = vmap(lap_standard)
compute_batched_collapsed_laplacian = vmap(lap_collapsed)

# %%
#
# Let's check if this yields the correct result. First, a sanity check that `vmap`
# worked as expected:

batch_size = 2_048
X = rand(batch_size, D)  # batched input

# ground truth: Loop over data points and compute the Laplacian for each, then
# concatenate the results
reference = stack([compute_hessian_trace_laplacian(X[b]) for b in range(batch_size)])
print(reference.shape)

# %%
#
# Let's check that all implementations yield the same Laplacian:

# NOTE Since we are computing in single precision, we need to slightly increase the
# tolerances to make Taylor mode and nested first-order AD match.
tols = {"atol": 1e-7, "rtol": 1e-4}

nested = compute_batched_nested_laplacian(X)
assert reference.allclose(nested, **tols)

standard = compute_batched_standard_laplacian(X)
assert reference.allclose(standard, **tols)

collapsed = compute_batched_collapsed_laplacian(X)
assert reference.allclose(collapsed, **tols)

# %%
#
### Performance
#
# Now that we have verified correctness, let's compare the performance in terms of run
# time. As measuring protocol, let's define the following function which repeats the
# measurements multiple times and reports the minimum run time as proxy for the actual run
# time.


def measure_runtime(f: Callable, num_repeats: int = 50) -> float:
    """Measure the run time of a function.

    Args:
        f: The function to measure.
        num_repeats: How many times to repeat the measurement.

    Returns:
        The minimum run time of the function in seconds.
    """
    runtimes = []
    for _ in range(num_repeats):
        start = perf_counter()
        f()
        end = perf_counter()
        runtimes.append(end - start)

    return min(runtimes)


ms_nested = 10**3 * measure_runtime(lambda: compute_batched_nested_laplacian(X))
ms_standard = 10**3 * measure_runtime(lambda: compute_batched_standard_laplacian(X))
ms_collapsed = 10**3 * measure_runtime(lambda: compute_batched_collapsed_laplacian(X))

print(f"Nested 1st-order AD: {ms_nested:.2f}ms ({ms_nested / ms_nested:.2f}x)")
print(f"Standard Taylor: {ms_standard:.2f}ms ({ms_standard / ms_nested:.2f}x)")
print(f"Collapsed Taylor: {ms_collapsed:.2f}ms ({ms_collapsed / ms_nested:.2f}x)")

# %%
#
# Let's also measure how much `torch.compile` can speed things up:

compute_batched_nested_compiled = torch_compile(compute_batched_nested_laplacian)
compute_batched_standard_compiled = torch_compile(compute_batched_standard_laplacian)
compute_batched_collapsed_compiled = torch_compile(compute_batched_collapsed_laplacian)

ms_nested_c = 10**3 * measure_runtime(lambda: compute_batched_nested_compiled(X))
ms_standard_c = 10**3 * measure_runtime(lambda: compute_batched_standard_compiled(X))
ms_collapsed_c = 10**3 * measure_runtime(lambda: compute_batched_collapsed_compiled(X))

print(
    f"Nested 1st-order AD (compiled): {ms_nested_c:.2f}ms ({ms_nested_c / ms_nested_c:.2f}x)"
)
print(
    f"Standard Taylor (compiled): {ms_standard_c:.2f}ms ({ms_standard_c / ms_nested_c:.2f}x)"
)
print(
    f"Collapsed Taylor (compiled): {ms_collapsed_c:.2f}ms ({ms_collapsed_c / ms_nested_c:.2f}x)"
)

# %%
#
# We see that collapsed Taylor mode is faster than standard Taylor mode.
# Of course, we use a relatively small neural net and a CPU in this example, but our
# paper also confirms this performance benefits on bigger nets and on GPU (also in
# terms of memory consumption). Intuitively, this makes sense, as the collapsed
# propagation uses less operations and smaller tensors.
#
# Here is a quick summary of the performance results in a single diagram:

methods = ["Nested 1st-order", "Standard Taylor", "Collapsed Taylor"]
times_eager = [ms_nested, ms_standard, ms_collapsed]
times_compiled = [ms_nested_c, ms_standard_c, ms_collapsed_c]
colors = [
    (117 / 255, 112 / 255, 179 / 255),
    (217 / 255, 95 / 255, 2 / 255),
    (27 / 255, 158 / 255, 119 / 255),
]

# Use LaTeX if available
USETEX = which("latex") is not None

with plt.rc_context(bundles.neurips2024(usetex=USETEX)):
    plt.figure(dpi=150)

    x_pos = arange(len(methods))
    bar_width = 0.32
    gap = 0.02

    # Eager bars (solid)
    bars_eager = plt.bar(
        x_pos - bar_width / 2 - gap / 2,
        times_eager,
        bar_width,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )
    # Compiled bars (same color, hatched)
    bars_compiled = plt.bar(
        x_pos + bar_width / 2 + gap / 2,
        times_compiled,
        bar_width,
        color=colors,
        hatch="//",
        edgecolor="black",
        linewidth=0.8,
    )

    plt.xticks(x_pos, methods)
    plt.ylabel("Time [ms]")
    plt.title(f"Computing Batched Laplacians ($N = {batch_size}$)")

    # Legend with only eager/compiled distinction (no color)
    plt.legend(
        handles=[
            Patch(facecolor="white", edgecolor="black", linewidth=0.8, label="eager"),
            Patch(
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                hatch="//",
                label="compiled",
            ),
        ]
    )

    # Add values on top of bars and relative speed-up as second label
    for bars, baseline in [
        (bars_eager, times_eager[0]),
        (bars_compiled, times_compiled[0]),
    ]:
        for bar in bars:
            height = bar.get_height()
            speedup = height / baseline
            x_mid = bar.get_x() + bar.get_width() / 2.0
            plt.text(x_mid, height, f"{height:.2f}ms", ha="center", va="bottom")
            plt.text(
                x_mid,
                height / 2,
                f"{speedup:.2f}x",
                ha="center",
                va="center",
                color="white",
            )

# %%
#
# That's all for now.
