"""Functions to simplify a compute graph captured with `torch.fx`."""

from typing import List

from torch import Tensor
from torch import add as torch_add
from torch import cos, cosh, einsum, mul
from torch import pow as torch_pow
from torch import sigmoid, sin, tanh
from torch.fx import GraphModule, Node
from torch.nn.functional import linear

from jet.utils import replicate

# operations that act repeatedly along the leading dimension of a tensor
# and can therefore be swapped with a `replicate` operation
ACTS_REPEATED = {linear, sin, cos, tanh, sigmoid, cosh, torch_pow}


def swappable_children(mod: GraphModule, verbose: bool = False) -> List[Node]:
    """Find children that can be swapped with their parent.

    Args:
        mod: A computation graph.
        verbose: Whether to print debug information. Default: `False`.

    Returns:
        A list of children nodes that can be swapped with their parent.
    """
    for child in mod.graph.nodes:
        if child.op != "call_function":
            continue
        parents = [
            n for n in mod.graph.nodes if n in child.args and n.op == "call_function"
        ]
        if len(parents) == 1:
            (parent,) = parents
            if parent.target != replicate:
                continue

            if child.target in ACTS_REPEATED:
                if verbose:
                    print(f"Can swap {parent} and {child}.")
                return [child]
            elif child.target == mul and isinstance(child.args[1], (float, int)):
                if verbose:
                    print(f"Can swap {parent} and {child}.")
                return [child]
            elif verbose:
                print(f"Cannot swap {parent} and {child}.")
        elif len(parents) == 2 and child.target == torch_add:
            if all(p.op == "call_function" and p.target == replicate for p in parents):
                if verbose:
                    print(f"Can swap {parents} and {child}.")
                return [child]
        elif len(parents) > 1:
            if (
                all(p.op == "call_function" and p.target == replicate for p in parents)
                and child.target == einsum
            ):
                num_tensors = len(child.args[1:])
                equation = ",".join(num_tensors * ["..."]) + "->..."
                if equation == child.args[0]:
                    if verbose:
                        print(f"Can swap {parents} and {child}.")
                    return [child]

    return []


def simplify(mod: GraphModule, verbose: bool = False) -> GraphModule:
    """Simplify a compute graph.

    At the moment, the following simplifications are implemented:

    - Propagation of `replicate` nodes down the graph as much as possible.
      This avoids redundant computations on replicated tensors.

    Args:
        mod: A computation graph that will be simplified.
        verbose: Whether to print debug information. Default: `False`.

    Returns:
        The simplified computation graph.
    """
    if verbose:
        print(f"Traced graph before simplification:\n{mod.graph}")

    graph = mod.graph
    # Propagate replicate nodes down the graph as much as possible: It is always better
    # to compute then replicate, rather than compute on a replicated object
    while swappable := swappable_children(mod, verbose=verbose):
        (child,) = swappable
        parents = [
            n for n in graph.nodes if n in child.args and n.op == "call_function"
        ]

        if len(parents) == 1:
            (parent,) = parents
            # do the swap:
            # 1. Insert a new replicate node right after the child
            # 2. replace all the child's uses with the new replicate node
            # 3. Set the child's output as the input to the new replicate node
            # 4. Set the child's input to the old replicate node's input
            with graph.inserting_after(child):
                new_parent = graph.call_function(
                    replicate, args=parent.args, kwargs=parent.kwargs
                )
            child.replace_all_uses_with(new_parent)
            new_parent.args = (child, *parent.args[1:])

            new_args = list(child.args)
            where = child.args.index(parent)
            new_args[where] = parent.args[0]
            child.args = tuple(new_args)

            # if parent has no more children, remove it
            if not any(n for n in graph.nodes if parent in n.args):
                if verbose:
                    print(f"Erasing node {parent}.")
                graph.erase_node(parent)
        elif (
            len(parents) == 2
            and child.op == "call_function"
            and child.target == torch_add
        ):

            with graph.inserting_after(child):
                new_parent = graph.call_function(replicate)
            child.replace_all_uses_with(new_parent)

            def find_parent(node):
                for parent in graph.nodes:
                    if parent in node.args:
                        return parent
                raise RuntimeError

            parent_inputs = tuple(find_parent(n) for n in child.args)
            child.args = parent_inputs
            new_parent.args = (child,) + parents[0].args[1:]

            # if parent has no more children, remove it
            for parent in parents:
                if (
                    not any(n for n in graph.nodes if parent in n.args)
                    and parent not in list(graph.nodes)[-1].args[0]
                ):
                    if verbose:
                        print(f"Erasing node {parent}.")
                        # print(graph)
                    graph.erase_node(parent)

        else:
            # raise Exception(f"{child} has parents {parents}")
            with graph.inserting_after(child):
                new_parent = graph.call_function(replicate)
            print(f"Einsum arguments: {child.args[1:]}")
            child.replace_all_uses_with(new_parent)

            def find_parent(node):
                for parent in graph.nodes:
                    if parent in node.args:
                        return parent
                raise RuntimeError

            parent_inputs = tuple(find_parent(n) for n in child.args[1:])
            child.args = (child.args[0],) + parent_inputs

            new_parent.args = (child,) + parents[0].args[1:]

            print(f"New einsum arguments: {child.args[1:]}")

            # if parent has no more children, remove it
            for parent in parents:
                if (
                    not any(n for n in graph.nodes if parent in n.args)
                    and parent not in list(graph.nodes)[-1].args[0]
                ):
                    if verbose:
                        print(f"Erasing node {parent}.")
                        # print(graph)
                    graph.erase_node(parent)

        print("\n\n")
        print(graph)
        print("\n\n")

    mod.graph.lint()
    mod.recompile()

    if verbose:
        print(f"Traced graph after simplification:\n{mod.graph}")

    return mod
