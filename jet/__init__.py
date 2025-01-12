"""Taylor-mode automatic differentiation (jets) in PyTorch."""

from math import factorial
from typing import Callable

from torch import Tensor, tensor, zeros_like
from torch.autograd import grad
from torch.fx import GraphModule, symbolic_trace

from jet.operations import (
    MAPPING,
    Primal,
    PrimalAndCoefficients,
    Value,
    ValueAndCoefficients,
)


def jet(
    f: Callable[[Primal], Value], verbose: bool = False
) -> Callable[[PrimalAndCoefficients], ValueAndCoefficients]:
    """Overload a function with its Taylor-mode equivalent.

    Args:
        f: Function to overload. Maps a tensor to another tensor.
        verbose: Whether to print the traced graphs before and after overloading.
            Default: `False`.

    Returns:
        The overloaded function that computes the function and its Taylor coefficients
        from the input tensor and its Taylor coefficients.
    """
    graph = symbolic_trace(f)

    if verbose:
        print("Traced graph before jet overloading:")
        print(graph.graph)

    jet_f = _replace_operations_with_taylor(graph)

    if verbose:
        print("Traced graph after jet overloading:")
        print(jet_f.graph)

    return jet_f


def _replace_operations_with_taylor(graph: GraphModule) -> GraphModule:
    """Replace operations in the graph with Taylor-mode equivalents.

    Args:
        graph: Traced PyTorch computation graph.

    Returns:
        The overloaded computation graph with Taylor arithmetic.

    Raises:
        NotImplementedError: If an unsupported operation or node is encountered while
            carrying out the overloading.
    """
    for node in tuple(graph.graph.nodes):
        if node.op == "call_function":
            f = node.target
            if f not in MAPPING.keys():
                raise NotImplementedError(f"Unsupported node target: {node.target}")
            with graph.graph.inserting_after(node):
                new_node = graph.graph.call_function(
                    MAPPING[f], args=node.args, kwargs=node.kwargs
                )
            node.replace_all_uses_with(new_node)
            graph.graph.erase_node(node)
        elif node.op not in ["output", "placeholder"]:
            raise NotImplementedError(f"Unsupported node operation: {node.op}")

    graph.graph.lint()
    graph.recompile()
    return graph


def rev_jet(
    f: Callable[[Primal], Value]
) -> Callable[[PrimalAndCoefficients], ValueAndCoefficients]:
    """Implement Taylor-mode arithmetic via nested reverse-mode autodiff.

    Args:
        f: Function to overload. Maps a tensor to another tensor.

    Returns:
        The overloaded function that computes the function and its Taylor coefficients
        from the input tensor and its Taylor coefficients.
    """

    def jet_f(arg: PrimalAndCoefficients) -> ValueAndCoefficients:
        """Compute the function and its Taylor coefficients.

        Args:
            arg: Tuple containing the input tensor and its Taylor coefficients.

        Returns:
            Tuple containing the function value and its Taylor coefficients.
        """
        x, vs = arg
        order = len(vs)

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

        return f_x, vs_out

    return jet_f
