"""# Current Limitations of `jet`."""

from pytest import raises
from torch import Tensor, cos, sin
from torch.fx.proxy import TraceError
from torch.nn.functional import relu


from jet import jet

# %%
#
### Limitations
#
# Our `jet` implementation for PyTorch has various limitations.
# Here, we want to describe them and comment on the potential to fix them.
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


with raises(TypeError):
    jet(f, 2)

# %%
#
#### Unsupported Operations
#
# **`jet` supports only a small number of operations.**
#
# As described [here](../03_from_torch_to_torch_jet), `jet` replaces the original function with its Taylor arithmetic.
# This overloading must be specified and correctly implemented for each operation.
# Typically, if a function is not supported, you will encounter an error. For instance,
# the ReLU function is currently not supported:

f = relu

with raises(NotImplementedError):
    jet(f, 2)

# %%
#
# ---
#
# **`jet` only knows how to overload `op_code="functional_call"` nodes.**
#
# There is another scenario of unsupported-ness. PyTorch compute graphs consist of
# nodes that have different types. Currently, `jet`'s Taylor arithmetic overloading
# exclusively works for nodes of type `op_code="functional_call"` and crashes if it
# encounters a different node type.
#
# For example, the following works

f = sin
_ = jet(f, 2)  # works because sin is called via a "functional_call" node

# %%
#
# but the following does not, although the function is the same:

f = lambda x: x.sin()

with raises(ValueError):
    jet(f, 2)  # crashes because sin is called via a "call_method" node

# %%
#
# This limitation is straightforward to address, but we currently off-load the burden
# of calling functions in the supported way to the user.
#
#### Untraceable Functions
#
# **`jet` inherits all limitations of `torch.fx.symbolic_trace`.**
#
# We need to capture the function's compute graph to overload it to obtain a jet.
# We use `torch.fx.symbolic_trace` to achieve this. It has certain limitations
# (please see the documentation) that our `jet` implementation inherits.
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


with raises(TraceError):
    jet(f, 2)  # crashes because f cannot be traced

# %%
#
# This is a fundamental limitation of `torch.fx` and cannot be fixed at the moment.
# It may be possible to support in the future if control flow operators are added to
# PyTorch's tracing mechanism.


# %%
### Conclusion
# Now that you know about `jet`'s limitations checkout this [tutorial](../02_laplacians)
# where we explain how to efficiently compute the Laplacian using collapsed Taylor mode.
