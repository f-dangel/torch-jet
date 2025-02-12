"""Functions to simplify a compute graph captured with `torch.fx`."""

from typing import List

from torch import cos, sigmoid, sin, tanh, zeros_like
from torch.fx import GraphModule, Node
from torch.nn.functional import linear

from jet.utils import replicate

# operations that act repeatedly along the leading dimension of a tensor
# and can therefore be swapped with a `replicate` operation
ACTS_REPEATED = {zeros_like, linear, sin, cos, tanh, sigmoid}


def swaps_with_replicate(node: Node) -> bool:
    """Check if a node can be swapped with a `replicate` node.

    Args:
        node: The node to check that processes the output of a `replicate` node.

    Returns:
        Whether the node can be swapped with the `replicate` node it processes.
    """
    return node.op == "call_function" and node.target in ACTS_REPEATED


def swappable_replicate_nodes(mod: GraphModule, verbose: bool = False) -> List[Node]:
    """Find `replicate` nodes that can be swapped with their children.

    Args:
        mod: A computation graph.
        verbose: Whether to print debug information. Default: `False`.

    Returns:
        A list of `replicate` nodes that can be swapped with their children.
    """
    rep_nodes = [
        n for n in mod.graph.nodes if n.op == "call_function" and n.target == replicate
    ]
    swappable = []
    for rep in rep_nodes:
        children = [n for n in mod.graph.nodes if rep in n.args]
        if not children:
            if verbose:
                print(f"Replicate node {rep} has no children.")
            continue
        elif all(swaps_with_replicate(child) for child in children):
            if verbose:
                print(f"Can swap {rep} with {children}.")
            swappable.append(rep)
        elif verbose:
            print(f"Cannot swap {rep} with {children}.")

    return swappable


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
    while swappable := swappable_replicate_nodes(mod):
        for rep in swappable:
            children = [n for n in graph.nodes if rep in n.args]
            # do the swap:
            # 1. Insert a new replicate node right after the child
            # 2. replace all the child's uses with the new replicate node
            # 3. Set the child's output as the input to the new replicate node
            # 4. Set the child's input to the old replicate node's input
            for child in children:
                with graph.inserting_after(child):
                    new_rep = graph.call_function(
                        replicate, args=rep.args, kwargs=rep.kwargs
                    )
                child.replace_all_uses_with(new_rep)
                new_rep.args = (child, *rep.args[1:])

                new_args = list(child.args)
                where = child.args.index(rep)
                new_args[where] = rep.args[0]
                child.args = tuple(new_args)

            # remove the replicate node
            graph.erase_node(rep)

    mod.graph.lint()
    mod.recompile()

    if verbose:
        print(f"Traced graph after simplification:\n{mod.graph}")

    return mod
