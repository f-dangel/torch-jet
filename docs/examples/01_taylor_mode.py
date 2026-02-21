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
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import Linear, Sequential, Tanh
from torch.nn.functional import relu

from jet import jet
from jet.tracing import capture_graph
from jet.utils import visualize_graph

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
# **In code,** the `jet` library offers a function transformation `jet(f, k, mock_args)`
# that takes a function $f$, a degree $k$, and mock arguments, and returns a new function
# `jet_f(primals, series)` that returns `(primals_out, series_out)` — the function value
# and its Taylor coefficients up to order $k$.

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
f_jet = jet(f, k, (x,))  # propagates (x₀, (x₁, x₂)) ↦ (f₀, (f₁, f₂))

# Set up the Taylor coefficients to compute the second derivative

x0 = x
x1 = ones_like(x)
x2 = zeros_like(x)

# Evaluate the second derivative
f0, (f1, f2) = f_jet((x0,), ((x1,), (x2,)))

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
f_jet = jet(f, 2, (x,))

# constant Taylor coefficients
x0 = x
x2 = zeros_like(x)

d2_diag = zeros_like(x)

# Compute the d-th diagonal element of the Hessian
for d in range(D):
    x1 = zeros_like(x)
    x1[d] = 1.0  # d-th canonical basis vector
    f0, (f1, f2) = f_jet((x0,), ((x1,), (x2,)))
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
# `jet` uses `make_fx` to capture the function's ATen-level compute graph, then wraps
# it in an interpreter that dispatches jet operations at runtime.
#
# When `f_jet` is called, the interpreter walks through the captured graph node by
# node. For each `call_function` node, it checks whether any argument is a
# Taylor-expanded value (a `JetTuple`). If so, it dispatches to the corresponding jet
# operation (e.g. `jet_linear`, `jet_tanh`) instead of the original ATen op.
#
# Let's visualize both the original function's compute graph and the unrolled jet
# function (obtained by tracing `f_jet` with `make_fx`):

mod = capture_graph(f, x)
visualize_graph(mod, path.join(GALLERYDIR, "01_f.png"))
visualize_graph(
    make_fx(f_jet)((x0,), ((x1,), (x2,))),
    path.join(GALLERYDIR, "01_f_jet_unrolled.png"),
)

# %%
#
# | Original function $f$ | Unrolled 2-jet function $f_{2\text{-jet}}$  |
# |:---------------------:|:-------------------------------------------:|
# | ![f graph](01_f.png)  | ![f-jet graph](01_f_jet_unrolled.png)       |
#
# The unrolled graph is, unsurprisingly, much larger. However, you should be able to
# recognize all functions that are being called. We can regard this process as a
# cycle that starts with a function $f$ that uses operations from PyTorch, and ends
# with a function $f_{k\text{-jet}}$ that also uses PyTorch operations.
# This is a desirable property as it enables composability (e.g. taking the `jet` of
# a `jet`).


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
# `jet` supports functions with multiple inputs and pytree I/O.  The calling
# convention mirrors `torch.func.jvp`:
#
# ```python
# jet_f = jet(f, k, mock_args)          # mock_args: tuple matching f's args
# primals_out, series_out = jet_f(primals, series)
# ```
#
# For a function with two arguments $f: (x, y) \mapsto f(x, y)$, the $k$-jet is
# called as:
#
# ```python
# f0, (f1, ..., fk) = jet_f((x0, y0), ((x1, y1), ..., (xk, yk)))
# ```
#
# Here is a quick example:


def f(x: Tensor, y: Tensor) -> Tensor:
    """A function with two tensor arguments.

    Args:
        x: First tensor argument.
        y: Second tensor argument.

    Returns:
        The sum of the two tensors.
    """
    return x + y


x, y = rand(3), rand(3)
f_jet = jet(f, 2, (x, y))
f0, (f1, f2) = f_jet(
    (x, y), ((ones_like(x), zeros_like(y)), (zeros_like(x), zeros_like(y)))
)
print(f"f0 = {f0}, f1 = {f1}, f2 = {f2}")

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


x_relu = rand(3)
f_relu = jet(lambda x: relu(x), 2, (x_relu,))  # noqa: PLW0108
with raises(NotImplementedError):
    f_relu((x_relu,), ((rand(3),), (rand(3),)))  # error is raised at call time

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
_ = jet(f, 2, (rand(3),))  # works because sin traces to aten.sin

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


_ = jet(f, 2, (rand(3),))  # also works with make_fx tracing

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
    jet(f, 2, (rand(3),))  # crashes because f cannot be traced

# %%
#
# This is a fundamental limitation of ``make_fx`` tracing and cannot be fixed at the
# moment. It may be possible to support in the future if control flow operators are
# added to PyTorch's tracing mechanism.
