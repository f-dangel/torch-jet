"""# Introduction to Taylor Mode.

This example provides an introduction to Taylor Mode, specifically the `jet` function
transformation, and how to use it to compute higher-order derivatives.

First, the imports.
"""

from torch import manual_seed, ones_like, rand, sin, zeros_like
from torch.func import hessian

from jet import jet

manual_seed(0)  # make deterministic

# %%
#
### What is Taylor mode?
#
# TODO: Explain the general idea

# %%
#
### Scalar Function
#
# Let's first consider the scalar case, a function $f : \mathbb{R} \to \mathbb{R}$,
# and set the jet degree to 2.
# Then, the 2-jet propagation through $f$ looks as follows:
# $$
# \begin{pmatrix}
# x_0 \\\\ x_1 \\\\ x_2
# \end{pmatrix}
# \mapsto
# \begin{pmatrix}
# f_0 := & f(x_0) \\\\
# f_1 := & f'(x_0) x_1 \\\\
# f_2 := & f''(x_0) x_1^2 + f'(x_0) x_2
# \end{pmatrix}\,.
# $$
# We can use this to compute the second-order derivative $f''(x)$, by setting $x_0 = x$,
# $x_1 = 1$, and $x_2 = 0$, which yields $f_2 = f''(x)$.

# Define a function and obtain its jet function
f = sin  # propagates x₀ ↦ f(x₀)
f_jet = jet(f, k=2)  # propagates (x₀, x₁, x₂) ↦ (f₀, f₁, f₂)

# Set up the Taylor coefficients to compute the second derivative
x = rand(1)

x0 = x
x1 = ones_like(x)
x2 = zeros_like(x)

f0, f1, f2 = f_jet(x0, x1, x2)

# Compare to the second derivative computed with first-order autodiff
d2f = hessian(f)(x)

if f2.allclose(d2f):
    print("Taylor Mode Hessian matches the true Hessian!")
else:
    raise ValueError(f"{f2} does not match {d2f}!")

# %%
#
### Vector Case
#
# TODO Explain the notation
