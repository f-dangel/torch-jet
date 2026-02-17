"""Taylor-mode automatic differentiation (jets) in PyTorch."""

from math import factorial
from typing import Callable
from warnings import warn

from torch import Tensor, tensor, zeros_like
from torch.autograd import grad
from torch.fx import Graph, GraphModule, Node

from jet.jet_transformer import JetTransformer
from jet.tracing import capture_graph
from jet.utils import Primal, PrimalAndCoefficients, Value, ValueAndCoefficients


def analyze_dependencies(graph: Graph) -> tuple[set[str], set[str]]:
    """Determine nodes that depend on placeholders or only on constants.

    Assume that all nodes have unique names.

    Args:
        graph: The graph to analyze.

    Returns:
        A tuple containing two sets:
        - The first set contains names of the nodes that depend on placeholder nodes.
        - The second set contains names of the nodes that depend only on constants.

    Raises:
        RuntimeError: If the dependencies cannot be determined for a node.
    """
    placeholder_nodes = {node.name for node in graph.nodes if node.op == "placeholder"}
    constant_nodes = {node.name for node in graph.nodes if node.op == "get_attr"}

    for node in graph.nodes:
        if node.op in ["placeholder", "get_attr"]:
            continue

        if any(n.name in placeholder_nodes for n in node.all_input_nodes):
            placeholder_nodes.add(node.name)
        elif all(n.name in constant_nodes for n in node.all_input_nodes):
            constant_nodes.add(node.name)
        else:
            raise RuntimeError(f"Could not detect dependencies for {node=}.\n{graph}")

    return placeholder_nodes, constant_nodes


def jet(
    f: Callable[[Primal], Value],
    derivative_order: int,
    example_input: Tensor,
    verbose: bool = False,
) -> Callable[[PrimalAndCoefficients], ValueAndCoefficients]:
    """Overload a function with its Taylor-mode equivalent.

    Args:
        f: Function to overload. Maps a tensor to another tensor.
        derivative_order: The order of the Taylor expansion.
        example_input: A concrete example input tensor for tracing. Does not need to
            be the actual input; only the shape and dtype matter.
        verbose: Whether to print the traced graphs before and after overloading.
            Default: `False`.

    Returns:
        The overloaded function that computes the function and its Taylor coefficients
            from the input tensor and its Taylor coefficients.

    Examples:
        >>> from torch import sin, cos, zeros, Tensor
        >>> from jet import jet
        >>> f = sin
        >>> jet2_f = jet(f, 2, example_input=zeros(1))
        >>> # Set up the Taylor coefficients
        >>> x0, x1, x2 = Tensor([0.123]), Tensor([-0.456]), Tensor([0.789])
        >>> # Compute the function value and its Taylor coefficients
        >>> f0, f1, f2 = jet2_f(x0, x1, x2)
        >>> # Manually verify the Taylor coefficients (Faa di Bruno)
        >>> df, d2f = cos(x0), -sin(x0) # derivatives of the sin function
        >>> assert f0.allclose(sin(x0))
        >>> assert f1.allclose(df * x1)
        >>> assert f2.allclose(df * x2 + d2f * x1 ** 2)
    """
    mod = capture_graph(f, example_input)

    if verbose:
        print(f"Traced graph before jet overloading:\n{mod.graph}")

    jet_mod = _replace_operations_with_taylor(mod, derivative_order)

    if verbose:
        print(f"Traced graph after jet overloading:\n{jet_mod.graph}")

    return jet_mod


def _replace_operations_with_taylor(  # noqa: C901, PLR0912, PLR0915
    mod: GraphModule, derivative_order: int
) -> GraphModule:
    """Replace operations in the graph with Taylor-mode equivalents.

    Args:
        mod: Traced PyTorch computation graph module.
        derivative_order: The order of the Taylor expansion.

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
    if output_node.name not in dependent_on_placeholders:
        assert output_node.name in dependent_on_constants
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
                lambda arg: tuple(
                    arg if i == 0 else zeros_like(arg)
                    for i in range(derivative_order + 1)
                ),
                args=(out_tensor,),
            )
            output_node.replace_input_with(out_tensor, trivial_node)
        dependent_on_placeholders.add(trivial_node.name)

    mod = JetTransformer(
        mod,
        derivative_order,
        dependent_on_placeholders,
        dependent_on_constants,
    ).transform()
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
