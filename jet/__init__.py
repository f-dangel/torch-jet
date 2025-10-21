"""Taylor-mode automatic differentiation (jets) in PyTorch."""

from math import factorial
from typing import Callable
from warnings import warn

from torch import Tensor, tensor, zeros_like
from torch.autograd import grad
from torch.fx import Graph, GraphModule, Node

from jet.operations import MAPPING, JetInfo
from jet.tracing import capture_graph
from jet.utils import Primal, PrimalAndCoefficients, Value, ValueAndCoefficients


def analyze_dependencies(graph: Graph) -> tuple[set[Node], set[Node]]:
    """Determine nodes that depend on placeholders or only on constants.

    Args:
        graph: The graph to analyze.

    Returns:
        A tuple containing two sets:
        - The first set contains nodes that depend on placeholder nodes.
        - The second set contains nodes that depend only on constants.

    Raises:
        RuntimeError: If the dependencies cannot be determined for a node.
    """
    placeholder_nodes = {node for node in graph.nodes if node.op == "placeholder"}
    constant_nodes = {node for node in graph.nodes if node.op == "get_attr"}

    for node in graph.nodes:
        if node.op in ["placeholder", "get_attr"]:
            continue

        if any(n in placeholder_nodes for n in node.all_input_nodes):
            placeholder_nodes.add(node)
        elif all(n in constant_nodes for n in node.all_input_nodes):
            constant_nodes.add(node)
        else:
            raise RuntimeError(f"Could not detect dependencies for {node=}.\n{graph}")

    return placeholder_nodes, constant_nodes


def jet(
    f: Callable[[Primal], Value], derivative_order: int, verbose: bool = False
) -> Callable[[PrimalAndCoefficients], ValueAndCoefficients]:
    """Overload a function with its Taylor-mode equivalent.

    Args:
        f: Function to overload. Maps a tensor to another tensor.
        derivative_order: The order of the Taylor expansion.
        verbose: Whether to print the traced graphs before and after overloading.
            Default: `False`.

    Returns:
        The overloaded function that computes the function and its Taylor coefficients
        from the input tensor and its Taylor coefficients.
    """
    mod = capture_graph(f)

    if verbose:
        print(f"Traced graph before jet overloading:\n{mod.graph}")

    jet_mod = _replace_operations_with_taylor(mod, derivative_order)

    if verbose:
        print(f"Traced graph after jet overloading:\n{jet_mod.graph}")

    return jet_mod


def _replace_operations_with_taylor(  # noqa: C901, PLR0912, PLR0915
    mod: GraphModule, k: int
) -> GraphModule:
    """Replace operations in the graph with Taylor-mode equivalents.

    Args:
        mod: Traced PyTorch computation graph module.
        k: The order of the Taylor expansion.

    Returns:
        The overloaded computation graph module with Taylor arithmetic.

    Raises:
        NotImplementedError: If an unsupported operation or node is encountered while
            carrying out the overloading.
        RuntimeError: If the multiplication type cannot be detected for a node.
    """
    graph = mod.graph

    # find the nodes that depend on the placeholder nodes and those that depend
    # only on constants
    dependent_on_placeholders, dependent_on_constants = analyze_dependencies(mod.graph)

    # If the output only depends on constants, the Taylor coefficients will be zero
    (output_node,) = [node for node in graph.nodes if node.op == "output"]
    if output_node not in dependent_on_placeholders:
        assert output_node in dependent_on_constants
        warn(
            f"The {output_node=} does not depend on the placeholder nodes. "
            f"The resulting jet will be trivially zero. {graph}",
            stacklevel=2,
        )
        # insert a node that generates the trivial Taylor components based on the
        # function value
        (out_tensor,) = output_node.args
        assert isinstance(out_tensor, Node)
        with graph.inserting_before(output_node):
            trivial_node = graph.call_function(
                lambda *args: tuple(
                    args[0] if i == 0 else zeros_like(args[0]) for i in range(k + 1)
                ),
                args=(out_tensor,),
            )
            output_node.replace_input_with(out_tensor, trivial_node)
        dependent_on_placeholders.add(trivial_node)

    # find the input node and insert nodes for the Taylor coefficients
    # currently we assume there is only one independent node (aka placeholder)
    (x,) = [node for node in graph.nodes if node.op == "placeholder"]
    vs = []
    current_node = x
    # after loop we have vs = [x, v1, v2, ...] and similarly in the graph
    for i in range(1, k + 1):
        with graph.inserting_after(current_node):
            v_i = graph.placeholder(name=f"v{i}")
            vs.append(v_i)
            current_node = v_i

    # find the nodes that consume the original input, replace each with a new node whose
    # argument is the tuple of original input and Taylor coefficients
    children_x = [node for node in graph.nodes if x in node.args]
    for child_x in children_x:
        with graph.inserting_after(child_x):
            where = child_x.args.index(x)
            new_args = list(child_x.args)
            new_args[where] = (x, *vs)
            new_node = graph.call_function(
                child_x.target, args=tuple(new_args), kwargs=child_x.kwargs
            )
        child_x.replace_all_uses_with(new_node)
        graph.erase_node(child_x)
        dependent_on_placeholders.add(new_node)

    # replace all ops (including that of new_node) with their Taylor mode equivalents
    for node in tuple(graph.nodes):
        if node.op == "call_function":
            # figure out which arguments are constants, and which depend on placeholders
            is_taylor = []
            for arg in node.args:
                if isinstance(arg, Node):
                    in_placeholders = arg in dependent_on_placeholders
                    in_constants = arg in dependent_on_constants
                    assert int(in_placeholders) + int(in_constants) == 1
                    is_taylor.append(in_placeholders)

                elif isinstance(arg, tuple) and all(isinstance(a, Node) for a in arg):
                    is_taylor.append(True)

                elif isinstance(arg, (int, float)) or arg is None:
                    is_taylor.append(False)

                else:
                    raise RuntimeError(
                        f"Could not detect dependency of {arg} for {node.target=}."
                    )
            is_taylor = tuple(is_taylor)

            f = node.target

            # if all arguments are constants, we don't have to replace
            if not any(is_taylor):
                # add the node to constant dependencies
                dependent_on_constants.add(node)
                continue

            elif f not in MAPPING.keys():
                raise NotImplementedError(f"Unsupported {node.target=}.")

            with graph.inserting_after(node):
                new_node = graph.call_function(
                    MAPPING[f],
                    args=node.args,
                    kwargs={
                        **node.kwargs,
                        "_jet_info": JetInfo(derivative_order=k, is_taylor=is_taylor),
                    },
                )
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
            dependent_on_placeholders.add(new_node)

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
    f: Callable[[Primal], Value],
    derivative_order: int | None = None,
    detach: bool = True,
) -> Callable[[PrimalAndCoefficients], ValueAndCoefficients]:
    """Implement Taylor-mode via nested reverse-mode autodiff.

    Args:
        f: Function to overload. Maps a tensor to another tensor.
        derivative_order: Order of the Taylor expansion. Default: `None`.
        detach: Whether to detach the output of the function and its Taylor coefficients
            from the computation graph. Default: `True`.

    Returns:
        The overloaded function that computes the function and its Taylor coefficients
        from the input tensor and its Taylor coefficients.
    """
    grad_kwargs = {
        "allow_unused": True,
        "materialize_grads": True,
        "create_graph": True,
    }

    def _maybe_grad(f: Tensor, X: Tensor) -> Tensor:
        """Compute the gradient if f requires grad, otherwise return zeros.

        Args:
            f: The function output for which to compute the gradient.
            X: The input tensor at which to compute the gradient.

        Returns:
            The gradient of f w.r.t. X if f requires grad, otherwise a tensor of zeros.
            Has the same shape as X.
        """
        return grad(f, X, **grad_kwargs)[0] if f.requires_grad else zeros_like(X)

    def jet_f(
        x: Primal, *vs: Primal, derivative_order: int | None = derivative_order
    ) -> ValueAndCoefficients:
        """Compute the function and its Taylor coefficients.

        Args:
            x: Input tensor.
            *vs: Taylor coefficients.
            derivative_order: Order of the Taylor expansion. If `None`, the order is the number of
                Taylor coefficients.

        Returns:
            Tuple containing the function value and its Taylor coefficients.
        """
        if derivative_order is None:
            derivative_order = len(vs)
        else:
            assert derivative_order == len(vs)

        def path(t: Tensor):
            x_t = x + sum(
                t**n / factorial(n) * v_n for n, v_n in enumerate(vs, start=1)
            )
            return f(x_t)

        t = tensor(0.0, requires_grad=True, dtype=x.dtype, device=x.device)
        f_x = path(t)

        vs_out = [zeros_like(f_x).flatten() for _ in vs]

        for i, dnf_dt in enumerate(f_x.flatten()):
            for n in range(derivative_order):
                dnf_dt = _maybe_grad(dnf_dt, t)
                vs_out[n][i] = dnf_dt.detach() if detach else dnf_dt

        f_x = f_x.detach() if detach else f_x
        vs_out = tuple((v.detach() if detach else v).reshape_as(f_x) for v in vs_out)

        return (f_x, *vs_out)

    return jet_f
