"""# From Torch to Torch-Jet.

This tutorial briefly explains how our Taylor mode (i.e., `jet`) is implemented.
For an introduction to Taylor mode see [here](../01_taylor_mode).

First the imports.
"""

from os import path

from torch import manual_seed
from torch.fx import GraphModule
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.nn import Linear, Sequential, Tanh

from jet import jet
from jet.tracing import capture_graph

HEREDIR = path.dirname(path.abspath(__name__))  # script directory for figure savepaths

_ = manual_seed(0)  # make deterministic


# %%
# Lets define a fourth-order `jet` of a linear layer with `tanh` activation.

f = Sequential(Linear(3, 1), Tanh())
f_jet = jet(f, 4)


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
# Let's visualize two compute graphs: The original function $f$, and its 4-jet function
# $f_{4\text{-jet}}$:

visualize_graph(capture_graph(f), path.join(HEREDIR, "01_f.png"))
visualize_graph(f_jet, path.join(HEREDIR, "01_f_jet.png"))

# %%
#
# | Original function $f$ | 4-jet function $f_{4\text{-jet}}$ |
# |:---------------------:|:---------------------------------:|
# | ![f graph](01_f.png)  | ![f-jet graph](01_f_jet.png)      |
#
# The way `jet` works is that is overwrites the original function's compute graph
# that leads to the following differences:
#
# 1. The original function $f$ takes a single tensor argument (the node with
#    `op_code="placeholder"`), while the 4-jet function $f_{4\text{-jet}}$ takes five
#    arguments, the Taylor coefficients $\mathbf{x}_0, \dots, \mathbf{x}_4$.
#
# 2. `jet` replaces each function call (nodes with `op_code="call_function"`) with a new
#    function that propagates the 4-jet rather than the function value. These jet functions
#    are defined in the `jet` library. Note, they take additional arguments (e.g. the jet
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

visualize_graph(capture_graph(f_jet), path.join(HEREDIR, "01_f_jet_unrolled.png"))

# %%
#
# The unrolled graph is, unsurprisingly, much larger. However, you should be able to
# recognize all functions that are being called. We can regard this process as a
# cycle that starts with a function $f$ that uses operations from PyTorch, and ends
# with a function $f_{k\text{-jet}}$ that also uses PyTorch operations.
# This is a desirable property as it enables composability (e.g. taking the `jet` of
# a `jet`).
#
# | Unrolled 4-jet function $f_{4\text{-jet}}$ |
# |:------------------------------------------:|
# | ![f-jet traced graph](01_f_jet_unrolled.png) |


# %%
### Conclusion
# In this tutorial, we explored how our `jet` function is implemented using PyTorch's FX
# system and how it transforms a given function into its Taylor-expanded counterpart.
# We visualized the original and transformed computation graphs and discussed how operations
# are replaced and unrolled to support higher-order derivatives.
#
# However, the library is in early stage and there several **limitations**. You can find more
# information [here](../04_limitations). If you are curious about how the **Taylor mode collapsing**
# technique works under the hood, checkout [this](../05_collapsing_taylor_mode) tutorial.
