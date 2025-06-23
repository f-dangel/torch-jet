"""Utility functions/classes for capturing compute graphs in PyTorch."""

import math
from typing import Callable

from torch.fx import GraphModule, Tracer
from torch.nn import Linear, Module, Sigmoid, Tanh

import jet.utils
from jet.utils import Primal, Value


class JetTracer(Tracer):
    """Custom tracer for overloading functions with Taylor-mode arithmetic."""

    def __init__(
        self, autowrap_modules=(math, jet.utils), autowrap_functions=()
    ) -> None:
        """Initialize the JetTracer.

        Args:
            autowrap_modules: Modules to autowrap. Default: `(math, jet.utils)`.
                The `jet.utils` module is included to autowrap the `replicate` and
                `sum_vmapped` functions, which are used in the simplification logic.
            autowrap_functions: Functions to autowrap. Default: `()`.
        """
        super().__init__(
            autowrap_modules=autowrap_modules, autowrap_functions=autowrap_functions
        )

    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool:
        """Determine whether a module is a leaf module or should be traced through.

        Args:
            m: Module to check.
            module_qualified_name: Qualified name of the module.

        Returns:
            Whether the module is a leaf module.
        """
        # We don't want to maintain additional logic for replacing `call_module` nodes
        # that execute modules who simply wrap `torch.nn.functional`s. Therefore, we
        # explicitly trace through them, which will result in `call_function` nodes for
        # which we maintain the logic to replace them with Taylor-mode arithmetic.
        if isinstance(m, (Linear, Tanh, Sigmoid)):
            return False
        return super().is_leaf_module(m, module_qualified_name)


class WrapperModule(Module):
    """Wraps a function in a module."""

    def __init__(self, f: Callable[[Primal], Value]):
        """Initialize the module.

        Args:
            f: Function to wrap.
        """
        super().__init__()
        self.f = f

    def forward(self, x: Primal) -> Value:
        """Forward pass of the module.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.f(x)


def capture_graph(f: Module | Callable | GraphModule) -> GraphModule:
    """Capture the compute graph of a module using a custom tracer.

    The tracer's granularity is specialized to perform the Taylor-mode arithmetic
    overloading required for creating jets, and to apply simplifications.

    Args:
        f: The (graph) module or callable to trace.

    Returns:
        The traced module with the captured compute graph.
    """
    f_mod = f if isinstance(f, (Module, GraphModule)) else WrapperModule(f)
    tracer = JetTracer()
    return GraphModule(f_mod, tracer.trace(f_mod))
