"""# Introduction to Taylor Mode.

This example provides an introduction to Taylor Mode, specifically the `jet` function
transformation, and how to use it to compute higher-order derivatives.
We will focus on second-order derivatives.

First, the imports.
"""

from os import path

from pytest import raises
from torch import Tensor, cos, manual_seed, ones_like, rand, sin, zeros_like
from torch.func import hessian
from torch.fx import GraphModule
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.nn import Linear, Sequential, Tanh
from torch.nn.functional import relu

from jet import jet
from jet.tracing import capture_graph

HEREDIR = path.dirname(path.abspath(__name__))
# We need to store figures here so they will be picked up in the built doc
GALLERYDIR = path.join(path.dirname(HEREDIR), "generated", "gallery")

_ = manual_seed(0)  # make deterministic

# %%
#
### What Is Taylor Mode?
#
# Taylor mode is an autodiff technique for efficiently computing higher-order
# derivatives of functions. The basic idea is described by the following diagram
# (taken from the paper):
#
# ---
# ![Taylor mode concept](01_taylor_mode.png)
# ---
#
# Let's walk through this diagram for the example of a scalar function
# $f : \mathbb{R} \to \mathbb{R}, x \mapsto f(x)$ and assume we want to evaluate the
# function or its derivatives at a point $x_0 \in \mathbb{R}$.
#
# - **Top left:** Instead of considering a point $x_0$ in input space, let us instead
#   consider a curve $x(t)$, where $t \in \mathbb{R}$ is time. Importantly, this curve
#   has to intersect the anchor when $t=0$, or $x(0) = x_0$.
#
# - **Top left $\to$ right:** Clearly, the curve $x(t)$ in the input space gives rise
#   to a curve $f(x(t))$ in the output space. Our goal is to extract information about
#   the output curve given information about the input curve.
#
# - **Left top $\to$ bottom:** So, how do derivatives come into play? The answer is that
#   derivatives naturally allow us to control properties of the curve $x(t)$. Let's say
#   we want to control the curve's velocity, acceleration, etc. at the anchor point.
#   We can do this by writing out the curve's Taylor expansion
#   $$
#   x(t) = x_0 + x_1 t + \frac{1}{2} x_2 t^2 + \ldots
#   $$
#   where $x_0$ is the anchor point, $x_1$ is the velocity at the anchor, $x_2$ is the
#   acceleration, and so on. We call $(x_0, x_1, x_2)$ the 2-jet of $x(t)$, and
#   $x_i = \left.\frac{\mathrm{d}^i x(t)}{\mathrm{d} t^i}\right|_{t=0}$ the $i$th Taylor
#   coefficient.
#
# - **Right top $\to$ bottom:** Just like for the input curve, we can also write
#   the Taylor expansion of the output curve $f(x(t))$ at the anchor point:
#   $$
#   f(x(t)) = f_0 + f_1 t + \frac{1}{2} f_2 t^2 + \ldots
#   $$
#   where $(f_0, f_1, f_2)$ is the 2-jet of $f(x(t))$ and the $i$th Taylor coefficient
#   is $f_i = \left.\frac{\mathrm{d}^i f(x(t))}{\mathrm{d} t^i}\right|_{t=0}$.
#
# - **Bottom left $\to$ right:** The question is now how we can compute the
#   2-jet of the output curve $f(x(t))$ given the 2-jet of the input curve $x(t)$.
#   This is exactly what Taylor mode does!
#
# **The propagation rules** are relatively easy to derive by hand using the chain rule:
#
# $$
# \begin{matrix}
# f_0 =& \left.\frac{\mathrm{d}^0 f(x(t))}{\mathrm{d} t^0}\right|\_{t=0}
#     =& f(x_0)
# \\\\
# f_1 =& \left.\frac{\mathrm{d} f(x(t))}{\mathrm{d} t}\right|\_{t=0}
#     =& f'(x_0) x_1
# \\\\
# f_2 =& \left.\frac{\mathrm{d}^2 f(x(t))}{\mathrm{d} t^2}\right|\_{t=0}
#     =& f''(x_0) x_2 + f'(x_0) x_1^2
# \\\\
# \vdots
# \end{matrix}
# $$
# See the paper's appendix for a cheat sheet that contains even higher orders.
# The important insight is that, by specifying the Taylor coefficients
# $(x_0, x_1, \dots)$, we can compute various derivatives!
#
# **In code,** the `jet` library offers a function transformation `jet(f, k)` that
# takes a function $f: x_0 \mapsto f(x_0) = f_0$ and a degree $k$ and returns a new
# function $f_{k\text{-jet}}: (x_0, x_1, x_2, \dots, f_k)$ that computes the $k$-jet
# $(f_0, f_1, f_2, \dots, f_k)$ of $f$.

# %%
#
### Scalar-to-scalar Function
#
# Let's make computing higher-order derivatives with Taylor mode concrete, sticking to
# the scalar case from a above with a function $f : \mathbb{R} \to \mathbb{R}$.
# We will illustrate how to compute the second-order derivative $f''(x)$, and hence
# use the 2-jet of $f$, whose propagation is (re-stated from above)
# $$
# f_{2\text{-jet}}:
# \begin{pmatrix}
# x_0 \\\\ x_1 \\\\ x_2
# \end{pmatrix}
# \mapsto
# \begin{pmatrix}
# f_0 = & f(x_0) \\\\
# f_1 = & f'(x_0) x_1 \\\\
# f_2 = & f''(x_0) x_1^2 + f'(x_0) x_2
# \end{pmatrix}\,.
# $$
# To achieve our goal, note that we can compute the second-order derivative $f''(x)$,
# by setting $x_0 = x$, $x_1 = 1$, and $x_2 = 0$, which yields $f_2 = f''(x)$:

# Define a function and obtain its jet function
f = sin  # propagates x₀ ↦ f(x₀)
k = 2  # jet degree
x = rand(1)
f_jet = jet(f, k, example_input=x)  # propagates (x₀, x₁, x₂) ↦ (f₀, f₁, f₂)

# Set up the Taylor coefficients to compute the second derivative

x0 = x
x1 = ones_like(x)
x2 = zeros_like(x)

# Evaluate the second derivative
f0, f1, f2 = f_jet(x0, x1, x2)

# %%
#
# Let's verify that this indeed yields the correct result:

# Compare to the second derivative computed with first-order autodiff
d2f = hessian(f)(x)

if f2.allclose(d2f):
    print("Taylor mode Hessian matches functorch Hessian!")
else:
    raise ValueError(f"{f2} does not match {d2f}!")

# We know the sine function's second derivative, so let's also compare with that
d2f_manual = -sin(x)
if f2.allclose(d2f_manual):
    print("Taylor mode Hessian matches manual Hessian!")
else:
    raise ValueError(f"{f2} does not match {d2f_manual}!")

# %%
#
### Vector-to-scalar Function
#
# Next, let's consider a vector to-scalar-function $f : \mathbb{R}^D \to \mathbb{R}$,
# $\mathbf{x} \mapsto f(\mathbf{x})$ (for the most general, please see the paper).
# We can do the exact derivation as above to obtain the output jets
# $$
# f_{2\text{-jet}}:
# \begin{pmatrix}
# \mathbf{x}_0 \\\\ \mathbf{x}_1 \\\\ \mathbf{x}_2
# \end{pmatrix}
# \mapsto
# \begin{pmatrix}
# f_0 = & f(\mathbf{x}_0) \\\\
# f_1 = & (\nabla f(\mathbf{x}_0))^\top \mathbf{x}_1 \\\\
# f_2 = & \mathbf{x}_1^\top (\nabla^2 f(\mathbf{x}_0)) \mathbf{x}_1
#         + (\nabla f(\mathbf{x}_0))^\top \mathbf{x}_2
# \end{pmatrix}\,,
# $$
# where $\nabla f(\mathbf{x}_0) \in \mathbb{R}^D$ is the gradient, and
# $\nabla^2 f(\mathbf{x}_0) \in \mathbb{R}^{D\times D}$ the Hessian, of $f$ at
# $\mathbf{x}_0$, while $\mathbf{x}_i$ is the $i$th input space Taylor coefficient. If
# we set $\mathbf{x}_0 = \mathbf{x}$, $\mathbf{x}_2 = \mathbf{0}$, then we can compute
# vector-Hessian-vector products (VHVPs) of the form
# $$
# f_2 = \mathbf{v}^\top (\nabla^2 f(\mathbf{x})) \mathbf{v}
# $$
# by setting $\mathbf{x}_1 = \mathbf{v}$.
# One interesting example is setting $\mathbf{x}_1 = \mathbf{e}_i$ to the $i$th
# canonical basis vector, which yields the $i$th diagonal entry of the Hessian, i.e.,
# $$
# [\nabla^2 f(\mathbf{x})]\_{i,i}
# =
# \mathbf{e}_i^\top (\nabla^2 f(\mathbf{x})) \mathbf{e}_i\,.
# $$
#
# Let's try this out and compute the Hessian diagonal with Taylor mode.
# This time, we will use a neural network with $\mathrm{tanh}$ activations:

D = 3
f = Sequential(Linear(D, 1), Tanh())
x = rand(D)
f_jet = jet(f, 2, example_input=x)

# constant Taylor coefficients
x0 = x
x2 = zeros_like(x)

d2_diag = zeros_like(x)

# Compute the d-th diagonal element of the Hessian
for d in range(D):
    x1 = zeros_like(x)
    x1[d] = 1.0  # d-th canonical basis vector
    f0, f1, f2 = f_jet(x0, x1, x2)
    d2_diag[d] = f2

# %%
#
# Let's compare this to computing the Hessian with `functorch` and then taking its
# diagonal:

d2f = hessian(f)(x)  # has shape [1, D, D]
hessian_diag = d2f.squeeze(0).diag()

if d2_diag.allclose(hessian_diag):
    print("Taylor mode Hessian diagonal matches functorch Hessian diagonal!")
else:
    raise ValueError(f"{d2_diag} does not match {hessian_diag}!")

# %%
#
### Conclusion
#
# If your goal was to learn how to use the `jet` function, you can stop reading at this point.
# But if you are interested in how `jet` works under the hood, and what its limitations
# are, keep reading!

# %%
#
### How It Works
#
# Let's have a look at the return type of `jet`:

tp = type(f_jet)
print(f"{tp.__module__}.{tp.__name__}")

# %%
#
# We can see that `jet` produces a `torch.fx.GraphModule`, which is an object that
# represents a compute graph. We can inspect this compute graph by `print`ing it:

print(f_jet.graph)

# %%
#
# While reading this string description already provides some insights into the graph's
# structure, it is often more convenient to visualize it. For this, we define the
# following helper:


def visualize_graph(mod: GraphModule, savefile: str, name: str = ""):
    """Visualize the compute graph of a module and store it as .png.

    Args:
        mod: The module whose compute graph to visualize.
        savefile: The path to the file where the graph should be saved.
        name: A name for the graph, used in the visualization.
    """
    drawer = FxGraphDrawer(mod, name)
    dot_graph = drawer.get_dot_graph()
    with open(savefile, "wb") as f:
        f.write(dot_graph.create_png())


# %%
#
# Let's visualize two compute graphs: The original function $f$, and its 2-jet function
# $f_{2\text{-jet}}$:

visualize_graph(capture_graph(f, example_input=x), path.join(GALLERYDIR, "01_f.png"))
visualize_graph(f_jet, path.join(GALLERYDIR, "01_f_jet.png"))

# %%
#
# | Original function $f$ | 2-jet function $f_{2\text{-jet}}$ |
# |:---------------------:|:---------------------------------:|
# | ![f graph](01_f.png)  | ![f-jet graph](01_f_jet.png)      |
#
# The way `jet` works is that is overwrites the original function's compute graph
# that leads to the following differences:
#
# 1. The original function $f$ takes a single tensor argument (one node with
#    `op_code="placeholder"`), while the 2-jet function $f_{2\text{-jet}}$ takes three
#    arguments, the Taylor coefficients $\mathbf{x}_0$, $\mathbf{x}_1$, $\mathbf{x}_2$.
#
# 2. `jet` replaces each function call (nodes with `op_code="call_function"`) with a new
#    function that propagates the 2-jet rather than the function value. These functions
#    are defined in the `jet` library, and they take additional arguments (e.g. the jet
#    degree) to take care of special situations we won't discuss here.
#
#     You can see that the following substitutions were performed by `jet`:
#
#     - `torch._C.nn.linear` $\leftrightarrow$ `jet.operations.jet_linear`
#
#     - `torch.tanh` $\leftrightarrow$ `jet.operations.jet_tanh`
#
# We can take an even 'deeper' look into this graph by tracing it again, which will
# 'unroll' the operations of the `jet.operations.*` functions:

visualize_graph(capture_graph(f_jet), path.join(GALLERYDIR, "01_f_jet_unrolled.png"))

# %%
#
# The unrolled graph is, unsurprisingly, much larger. However, you should be able to
# recognize all functions that are being called. We can regard this process as a
# cycle that starts with a function $f$ that uses operations from PyTorch, and ends
# with a function $f_{k\text{-jet}}$ that also uses PyTorch operations.
# This is a desirable property as it enables composability (e.g. taking the `jet` of
# a `jet`).
#
# | Unrolled 2-jet function $f_{2\text{-jet}}$ |
# |:------------------------------------------:|
# | ![f-jet traced graph](01_f_jet_unrolled.png) |


# %%
#
### Limitations
#
# Our `jet` implementation for PyTorch that this library provides has various
# limitations. Here, we want to describe them and comment on the potential to fix them.
#
# **Some limitations are a consequence of our still evolving know-how
# how to properly implement `jet` in PyTorch. So if you have suggestions how to fix
# them, please reach out to us, open an issue, or submit a pull request :wink:.**
#
#### Supported Function Signatures
#
# **At the moment, `jet` only works on functions that process a single tensor.**
#
# **Why?** This is mostly to keep the `jet` function simple and already covers a lot of
# use cases. We believe it is feasible to generalize our `jet` implementation and mostly
# a non-trivial engineering challenge.
# JAX's Taylor mode (`jax.experimental.jet.jet`) can handle such scenarios.
#
# Let's look at an example to make this clear. Coming back to our original example from
# above, let's imagine a function with two arguments, $f: (x, y) \mapsto f(x, y)$.
# Clearly, we can go through the same steps and define Taylor coefficients of the output
# space curve $f(x(t), y(t))$ that can be computed given the Taylor coefficients of the
# input curves $x(t), y(t)$. Hence, the $k$-jet's signature should be
# $$
# f_{k\text{-jet}}:
# \begin{pmatrix}
# \begin{pmatrix}
# x_0 \\\\ x_1 \\\\ \ldots \\\\ x_k
# \end{pmatrix},
# &
# \begin{pmatrix}
# y_0 \\\\ y_1 \\\\ \ldots \\\\ y_k
# \end{pmatrix}
# \end{pmatrix}
# \mapsto
# \begin{pmatrix}
# f_0 \\\\ f_1 \\\\ \ldots \\\\ f_k
# \end{pmatrix}\,.
# $$
# This is currently not implemented:


def f(x: Tensor, y: Tensor) -> Tensor:
    """A function with two tensor arguments (currently not supported).

    Args:
        x: First tensor argument.
        y: Second tensor argument.

    Returns:
        The sum of the two tensors.
    """
    return x + y


with raises(RuntimeError):
    jet(f, 2, example_input=rand(3))

# %%
#
#### Unsupported Operations
#
# **`jet` supports only a small number of operations.**
#
# As described above, `jet` replaces the original function with its Taylor arithmetic.
# This overloading must be specified and correctly implemented for each operation.
# Typically, if a function is not supported, you will encounter an error. For instance,
# the ReLU function is currently not supported:


def f(x):  # noqa: D103
    return relu(x)


with raises(NotImplementedError):
    jet(f, 2, example_input=rand(3))

# %%
#
# ---
#
# **With ``make_fx`` tracing, method calls are automatically handled.**
#
# Previously, ``jet`` only supported ``call_function`` nodes and would crash on
# method calls like ``x.sin()``. With ``make_fx`` tracing, all operations are
# decomposed into ATen-level function calls, so this limitation no longer exists.
#
# For example, the following works

f = sin
_ = jet(f, 2, example_input=rand(3))  # works because sin traces to aten.sin

# %%
#
# and so does calling sin as a method, since ``make_fx`` decomposes both to the
# same ATen operation:


def f(x: Tensor) -> Tensor:
    """Function that calls sin as a method.

    Args:
        x: Input tensor.

    Returns:
        The sine of x.
    """
    return x.sin()


_ = jet(f, 2, example_input=rand(3))  # also works with make_fx tracing

# %%
#
# This limitation is straightforward to address, but we currently off-load the burden
# of calling functions in the supported way to the user.
#
#### Untraceable Functions
#
# **`jet` inherits all limitations of `make_fx` tracing.**
#
# We need to capture the function's compute graph to overload it to obtain a jet.
# We use ``make_fx`` to achieve this. It has certain limitations
# (please see the documentation) that our ``jet`` implementation inherits.
#
# For instance, data-dependent control flow cannot be traced:


def f(x: Tensor):
    """Function with data-dependent control flow (if statement).

    Args:
        x: Input tensor.

    Returns:
        The sine of x if the sum of x is positive, otherwise the cosine of x.
    """
    return sin(x) if x.sum() > 0 else cos(x)


with raises(RuntimeError):
    jet(f, 2, example_input=rand(3))  # crashes because f cannot be traced

# %%
#
# This is a fundamental limitation of ``make_fx`` tracing and cannot be fixed at the
# moment. It may be possible to support in the future if control flow operators are
# added to PyTorch's tracing mechanism.
