"""Trace-able vmap of a function.

Assume we have a PyTorch function f: x ↦ f(x).
We can compute torch.vmap(f): {x} ↦ {f(x)}.

Problem: This vmap is not trace-able by PyTorch's FX tracer.
I.e., if we try torch.fx.symbolic_trace(torch.vmap(f)), we get an error.

Solution: We implement our own vmap that allows the FX tracer to trace it.
To achieve this, we must make some simplifying assumptions.
"""

import operator
from typing import Callable
from warnings import warn

from torch import Tensor, add
from torch.fx import GraphModule, Node
from torch.nn import Module

from jet import JetTracer, analyze_dependencies
from jet.utils import WrapperModule, replicate


def vmap_add(
    a: Tensor | float | int,
    b: Tensor | float | int,
    is_const: tuple[bool, bool],
    vmapsize: int,
) -> Tensor:
    """Vectorized addition of two tensors or scalars.

    Args:
        a: First operand, can be a tensor or scalar.
        b: Second operand, can be a tensor or scalar.
        is_const: Tuple indicating which arguments are constant (and therefore do not
            have a batch axis).
        vmapsize: Size of the vmapped axis.

    Returns:
        A tensor containing the result of the addition.
    """
    a_new = (
        replicate(a, vmapsize) if isinstance(a, (Node, Tensor)) and is_const[0] else a
    )
    b_new = (
        replicate(b, vmapsize) if isinstance(b, (Node, Tensor)) and is_const[1] else b
    )
    return a_new + b_new


MAPPING = {
    add: vmap_add,
    operator.add: vmap_add,
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

    # Wrap the function in a module if it is not already a module.
    # We want to always produce an executable `torch.fx.GraphModule`.
    if not isinstance(f, Module):
        f = WrapperModule(f)

    mod = GraphModule(f, JetTracer().trace(f))
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
                t_new = graph.call_function(replicate, args=(t_old, vmapsize))
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
                is_const = tuple(
                    isinstance(arg, (float, int)) or arg in constant_deps
                    for arg in node.args
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
