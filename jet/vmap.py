"""Trace-able vmap of a function.

Assume we have a PyTorch function f: x ↦ f(x).
We can compute torch.vmap(f): {x} ↦ {f(x)}.

Problem: This vmap is not trace-able by PyTorch's FX tracer.
I.e., if we try torch.fx.symbolic_trace(torch.vmap(f)), we get an error.

Solution: We implement our own vmap that allows the FX tracer to trace it.
To achieve this, we must make some simplifying assumptions.
For instance, operations like addition/subtraction must avoid broadcasting.
"""

import operator
from typing import Callable
from warnings import warn

from torch import Tensor, cos, mul, sigmoid, sin, tanh
from torch.fx import GraphModule, Node
from torch.nn.functional import linear

import jet.utils
from jet import analyze_dependencies
from jet.tracing import capture_graph


def vmap_basic_binary(
    op: Callable[[Tensor | float | int, Tensor | float | int], Tensor],
) -> Callable[
    [Tensor | float | int, Tensor | float | int, tuple[bool, bool], int], Tensor
]:
    """Create a vectorized basic binary operation.

    Used to vmap basic operations like addition, subtraction, and multiplication.

    Args:
        op: The binary operation to be vectorized, e.g., operator.add. Takes two
            arguments, each of which can be a Tensor, float, or int. Produces a tensor.

    Returns:
        The operation that corresponds to the vectorized operation and satisfies the
        signature assumed by the traceable_vmap function.
    """

    def vmap_op(
        a: Tensor | float | int,
        b: Tensor | float | int,
        is_const: tuple[bool, bool],
        vmapsize: int,
    ) -> Tensor:
        """Vectorized operation for basic binary operations.

        Args:
            a: First argument, can be a Tensor, float, or int.
            b: Second argument, can be a Tensor, float, or int.
            is_const: Tuple indicating which arguments are constant (and therefore do
                not have a batch axis).
            vmapsize: Size of the vmapped axis.

        Returns:
            A tensor containing the result of the vmap-ed operation.
        """
        a_new = (
            jet.utils.replicate(a, vmapsize)
            if isinstance(a, (Node, Tensor)) and is_const[0]
            else a
        )
        b_new = (
            jet.utils.replicate(b, vmapsize)
            if isinstance(b, (Node, Tensor)) and is_const[1]
            else b
        )
        return op(a_new, b_new)

    return vmap_op


def vmap_elementwise(
    f: Callable[[Tensor], Tensor],
) -> Callable[[Tensor, tuple[bool], int], Tensor]:
    """Create a vectorized element-wise operation.

    Used to vmap element-wise operations like cos, sin, sigmoid, and tanh.

    Args:
        f: The element-wise operation to be vectorized. Takes a single argument,
            which is a Tensor, and produces a tensor.

    Returns:
        The operation that corresponds to the vectorized operation and satisfies the
        signature assumed by the traceable_vmap function.
    """

    def vmap_f(x: Tensor, is_const: tuple[bool], vmapsize: int) -> Tensor:
        """Vectorized element-wise operation.

        Args:
            x: Input tensor.
            is_const: Tuple indicating which arguments are constant (and therefore do
                not have a batch axis).
            vmapsize: Size of the vmapped axis.

        Returns:
            A tensor containing the result of the element-wise vmap-ed operation.
        """
        return f(x)

    return vmap_f


vmap_add = vmap_basic_binary(operator.add)
vmap_sub = vmap_basic_binary(operator.sub)
vmap_mul = vmap_basic_binary(operator.mul)


def vmap_pow(
    x: Tensor, exponent: float | int, is_const: tuple[bool, bool], vmapsize: int
) -> Tensor:
    """Vectorized power operation.

    Args:
        x: Input tensor.
        exponent: Exponent tensor or scalar.
        is_const: Tuple indicating which arguments are constant (and therefore do not
            have a batch axis).
        vmapsize: Size of the vmapped axis.

    Returns:
        A tensor containing the result of the power operation.

    Raises:
        NotImplementedError: If the input tensor x is constant or if the exponent is not
            constant.
    """
    if is_const != (False, True):
        raise NotImplementedError("x must be non-constant, exponent must be constant.")
    if not isinstance(exponent, (float, int)):
        raise NotImplementedError("Exponent must be a float or int.")

    return x**exponent


vmap_cos = vmap_elementwise(cos)
vmap_sin = vmap_elementwise(sin)
vmap_sigmoid = vmap_elementwise(sigmoid)
vmap_tanh = vmap_elementwise(tanh)


def vmap_linear(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    is_const: tuple[bool, ...] = (True, True, True),
    vmapsize: int = 0,
) -> Tensor:
    """Vectorized linear transformation.

    Args:
        x: Input tensor.
        weight: Weight tensor.
        bias: Optional bias tensor.
        is_const: Tuple indicating which arguments are constant (and therefore do not
            have a batch axis).
        vmapsize: Size of the vmapped axis.

    Returns:
        A tensor containing the result of the vmap-ed linear transformation.

    Raises:
        NotImplementedError: If the input tensor x is constant or if the weight tensor
            W is not constant, or if the bias is not constant.
        ValueError: If vmapsize is not positive.
    """
    if is_const[0] or not is_const[1]:
        raise NotImplementedError("x must be non-constant, W must be constant.")
    if bias is not None and not is_const[2]:
        raise NotImplementedError("bias must be constant.")
    if vmapsize <= 0:
        raise ValueError(f"vmapsize must be positive, got {vmapsize=}.")

    return linear(x, weight, bias)


def vmap_sample(
    x_meta: Tensor,
    distribution: str,
    shape: tuple[int, ...],
    is_const: tuple[bool, ...] = (True, True, True),
    vmapsize: int = 0,
) -> Tensor:
    """Vectorized sampling operation.

    Args:
        x_meta: Metadata tensor for the input.
        distribution: Name of the distribution to sample from.
        shape: Shape of the output tensor.
        is_const: Tuple indicating which arguments are constant (and therefore do not
            have a batch axis).
        vmapsize: Size of the vmapped axis.

    Returns:
        A tensor containing the sampled values.

    Raises:
        NotImplementedError: If the input tensor x_meta is constant or if the
            distribution/shape are not constant.
        ValueError: If vmapsize is not positive.
    """
    if is_const != (False, True, True):
        raise NotImplementedError(
            "x_meta must be non-constant, distribution and shape must be constant."
        )
    if vmapsize <= 0:
        raise ValueError(f"{vmapsize=} must be positive.")

    return jet.utils.sample(x_meta, distribution, (vmapsize, *shape))


def vmap_replicate(
    x: Tensor,
    times: int,
    pos: int = 0,
    is_const: tuple[bool, ...] = (True, True),
    vmapsize: int = 0,
) -> Tensor:
    """Vectorized replicate operation.

    Args:
        x: Input tensor.
        times: Number of times to replicate the tensor.
        pos: Position of the new axis. Default: `0`.
        is_const: Tuple indicating which arguments are constant (and therefore do not
            have a batch axis).
        vmapsize: Size of the vmapped axis.

    Returns:
        A tensor containing the vmap-ed replicated input tensor.

    Raises:
        NotImplementedError: If the input tensor x is constant.
        ValueError: If vmapsize is not positive.
    """
    if is_const[0]:
        raise NotImplementedError("x must be non-constant.")
    if vmapsize <= 0:
        raise ValueError(f"vmapsize must be positive, got {vmapsize=}.")

    return jet.utils.replicate(x, times, pos=pos + 1)


def vmap_sum_vmapped(
    x: Tensor, pos: int = 0, is_const: tuple[bool, ...] = (True,), vmapsize: int = 0
) -> Tensor:
    """Vectorized sum operation for vmapped tensors.

    Args:
        x: Vmap-ed tensor.
        pos: Position of the vmap-ed axis to sum out. Default: `0`.
        is_const: Tuple indicating which arguments are constant (and therefore do not
            have a batch axis).
        vmapsize: Size of the vmapped axis.

    Returns:
        A tensor containing the sum of the vmap-ed tensor.

    Raises:
        NotImplementedError: If the input tensor x is constant.
        ValueError: If vmapsize is not positive.
    """
    if is_const[0]:
        raise NotImplementedError("x must be non-constant.")
    if vmapsize <= 0:
        raise ValueError(f"vmapsize must be positive, got {vmapsize=}.")

    return jet.utils.sum_vmapped(x, pos=pos + 1)


MAPPING = {
    # addition, subtraction, multiplication
    operator.add: vmap_add,
    operator.sub: vmap_sub,
    operator.mul: vmap_mul,
    mul: vmap_mul,
    # power
    operator.pow: vmap_pow,
    # linear layer
    linear: vmap_linear,
    # element-wise functions
    cos: vmap_cos,
    sin: vmap_sin,
    sigmoid: vmap_sigmoid,
    tanh: vmap_tanh,
    # replicate
    jet.utils.replicate: vmap_replicate,
    # sum_vmapped
    jet.utils.sum_vmapped: vmap_sum_vmapped,
    # sampling
    jet.utils.sample: vmap_sample,
}


def traceable_vmap(  # noqa: C901
    f: Callable[[Tensor], Tensor | tuple[Tensor, ...]], vmapsize: int
) -> GraphModule:
    """Create a traceable 'batched' function.

    Args:
        f: Function to be vmapped, which takes a tensor and returns a tensor or tuple of
            tensors.
        vmapsize: Size of the vmapped axis. Must be specified to ensure trace-ability
            with torch.fx.

    Returns:
        The 'batched' function of f as traced graph module.

    Raises:
        NotImplementedError: If the function or its operations are not supported.
        ValueError: If vmapsize is not positive.
    """
    if vmapsize <= 0:
        raise ValueError(f"vmapsize must be positive, got {vmapsize=}.")

    mod = capture_graph(f)
    graph = mod.graph

    # eliminate dead code
    graph.eliminate_dead_code()

    # analyze dependencies
    placeholder_deps, constant_deps = analyze_dependencies(mod.graph)

    # If the output only depends on constants, the vmap-ed result will be simply
    # a copy of these constant
    (output,) = [node for node in graph.nodes if node.op == "output"]
    if output not in placeholder_deps:
        warn(
            f"The {output=} does not depend on the placeholder nodes. "
            f"The resulting vmap will be a replicate. {graph}"
        )
        assert all(isinstance(arg, Node) for arg in output.args)
        out_tensors = set(output.all_input_nodes)
        # replicate the output tensors before returning them
        for t_old in out_tensors:
            with graph.inserting_before(output):
                t_new = graph.call_function(jet.utils.replicate, args=(t_old, vmapsize))
                output.replace_input_with(t_old, t_new)

    # Replace all nodes with their vmap-ed versions
    else:
        for node in tuple(graph.nodes):
            # node is purely generated from constant -> no replacement required
            if node.op == "call_function" and all(
                in_node in constant_deps for in_node in node.all_input_nodes
            ):
                constant_deps.add(node)

            elif node.op == "call_function":
                # FIXME Brittle if kwargs are supplied in random order
                is_const = tuple(
                    isinstance(arg, (float, int, str))
                    or (isinstance(arg, tuple) and all(isinstance(a, int) for a in arg))
                    or arg in constant_deps
                    or arg is None
                    for arg in list(node.args) + list(node.kwargs.values())
                )
                f = node.target
                if f not in MAPPING.keys():
                    raise NotImplementedError(f"Unsupported {node.target=}.")

                with graph.inserting_after(node):
                    new_node = graph.call_function(
                        MAPPING[f],
                        args=node.args,
                        kwargs={
                            **node.kwargs,
                            "is_const": is_const,
                            "vmapsize": vmapsize,
                        },
                    )
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)
                placeholder_deps.add(new_node)

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
