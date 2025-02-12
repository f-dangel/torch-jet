"""Taylor-mode automatic differentiation (jets) in PyTorch."""

from math import factorial
from typing import Callable, Optional, Tuple

from torch import Tensor, split, stack, tensor, zeros_like
from torch.autograd import grad
from torch.fx import GraphModule, Tracer
from torch.nn import Linear, Module, Sigmoid, Tanh

from jet.operations import MAPPING
from jet.utils import (
    Primal,
    PrimalAndCoefficients,
    Value,
    ValueAndCoefficients,
    WrapperModule,
)


class JetTracer(Tracer):
    """Custom tracer for overloading functions with Taylor-mode arithmetic."""

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


def jet(
    f: Callable[[Primal], Value],
    k: int,
    vmap: bool = False,
    collapse: bool = False,
    verbose: bool = False,
) -> Callable[[Tuple[Primal, ...]], Tuple[Value, ...]]:
    """Overload a function with its Taylor-mode equivalent.

    Args:
        f: Function to overload. Maps a tensor to another tensor.
        k: The order of the Taylor expansion.
        collapse: Whether to `vmap` and collapse the highest coefficient.
            Default: `False`.
        verbose: Whether to print the traced graphs before and after overloading.
            Default: `False`.

    Returns:
        The overloaded function that computes the function and its Taylor coefficients
        from the input tensor and its Taylor coefficients.
    """
    assert not collapse
    # Wrap the function in a module if it is not already a module.
    # We want to always produce an executable `torch.fx.GraphModule`.
    if not isinstance(f, Module):
        f = WrapperModule(f)

    graph = JetTracer().trace(f)
    mod = GraphModule(f, graph)

    if verbose:
        print(f"Traced graph before jet overloading:\n{mod.graph}")

    jet_mod = _replace_operations_with_taylor(mod, k, vmap=vmap, collapse=collapse)

    if verbose:
        print(f"Traced graph after jet overloading:\n{jet_mod.graph}")

    return jet_mod


def _replace_operations_with_taylor(
    mod: GraphModule, k: int, vmap: bool, collapse: bool
) -> GraphModule:
    """Replace operations in the graph with Taylor-mode equivalents.

    Args:
        mod: Traced PyTorch computation graph module.
        k: The order of the Taylor expansion.
        collapse: Whether to `vmap` and collapse the highest coefficient.

    Returns:
        The overloaded computation graph module with Taylor arithmetic.

    Raises:
        NotImplementedError: If an unsupported operation or node is encountered while
            carrying out the overloading.
    """
    assert not collapse
    graph = mod.graph

    # find the input node and insert nodes for the Taylor coefficients
    (x,) = [node for node in graph.nodes if node.op == "placeholder"]
    with graph.inserting_after(x):
        vs = [graph.placeholder(name=f"v{i}") for i in reversed(range(1, k + 1))][::-1]

    # find the node that consumes the original input, replace it with a new node whose
    # argument argument is the tuple of original input and Taylor coefficients
    (child_x,) = [node for node in graph.nodes if x in node.args]
    with graph.inserting_after(child_x):
        where = child_x.args.index(x)
        new_args = list(child_x.args)
        new_args[where] = (x, *vs)
        new_node = graph.call_function(
            child_x.target, args=tuple(new_args), kwargs=child_x.kwargs
        )
    child_x.replace_all_uses_with(new_node)
    graph.erase_node(child_x)

    # replace all operations (including that of new_node) their Taylor mode equivalents
    for node in tuple(graph.nodes):
        if node.op == "call_function":
            f = node.target
            if f not in MAPPING.keys():
                raise NotImplementedError(f"Unsupported node target: {node.target}")
            with graph.inserting_after(node):
                new_node = graph.call_function(
                    MAPPING[f],
                    args=node.args,
                    kwargs={**node.kwargs, "K": k, "vmap": vmap},
                )
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
        elif node.op == "call_module":
            module = graph.get_submodule(node.target)
            raise NotImplementedError(
                f"Unsupported module: {module}. Consider adding it to the"
                " `JetTracer.is_leaf_module` function."
            )
        elif node.op not in ["output", "placeholder", "get_attr"]:
            raise NotImplementedError(f"Unsupported node operation: {node.op}")

    mod.graph.lint()
    mod.recompile()

    return mod


def rev_jet(
    f: Callable[[Primal], Value], order: Optional[int] = None
) -> Callable[[PrimalAndCoefficients], ValueAndCoefficients]:
    """Implement Taylor-mode arithmetic via nested reverse-mode autodiff.

    Args:
        f: Function to overload. Maps a tensor to another tensor.
        order: Order of the Taylor expansion. Default: `None`.

    Returns:
        The overloaded function that computes the function and its Taylor coefficients
        from the input tensor and its Taylor coefficients.
    """

    def jet_f(x, *vs, order=order) -> ValueAndCoefficients:
        """Compute the function and its Taylor coefficients.

        Args:
            arg: Tuple containing the input tensor and its Taylor coefficients.

        Returns:
            Tuple containing the function value and its Taylor coefficients.
        """
        if order is None:
            order = len(vs)
        else:
            assert order == len(vs)

        def path(t: Tensor):
            x_t = x + sum(
                t**n / factorial(n) * v_n for n, v_n in enumerate(vs, start=1)
            )
            return f(x_t)

        t = tensor(0.0, requires_grad=True, dtype=x.dtype, device=x.device)
        f_x = path(t)

        vs_out = [zeros_like(f_x).flatten() for _ in vs]

        for i, dnf_dt in enumerate(f_x.flatten()):
            for n in range(order):
                (dnf_dt,) = grad(dnf_dt, t, create_graph=True)
                vs_out[n][i] = dnf_dt.detach()

        f_x = f_x.detach()
        vs_out = tuple(v.detach().reshape_as(f_x) for v in vs_out)

        return (f_x, *vs_out)

    return jet_f
