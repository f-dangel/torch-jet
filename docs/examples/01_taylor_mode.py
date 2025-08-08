"""# Introduction to Taylor Mode.

This example provides an introduction to Taylor Mode, and how to use it to
compute higher-order derivatives using our `jet` function.  We will focus
on second-order derivatives.


First, the imports.
"""

from os import path

from torch import manual_seed, ones_like, rand, sin, zeros_like
from torch.func import hessian
from torch.nn import Linear, Sequential, Tanh

from jet import jet

HEREDIR = path.dirname(path.abspath(__name__))  # script directory for figure savepaths

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
f_jet = jet(f, k)  # propagates (x₀, x₁, x₂) ↦ (f₀, f₁, f₂)

# Set up the Taylor coefficients to compute the second derivative
x = rand(1)

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

f = Sequential(Linear(3, 1), Tanh())
f_jet = jet(f, 2)

D = 3
x = rand(D)

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
# But if you are interested in how `jet` works under the hood have a look into
# [From Torch to Torch-Jet](../03_from_torch_to_torch_jet) or
# if you want to learn how to compute PDE operators using collapsed Taylor mode see [here](../02_laplacians).
# For a discussion of `jet`'s limitations checkout [this](../04_limitations).
