"""Functions to simplify a compute graph captured with `torch.fx`."""

from torch.fx import Graph


def common_subexpression_elimination(graph: Graph) -> bool:
    """Replace duplicate subexpressions with a single node.

    Args:
        graph: The graph to be optimized.

    Returns:
        Whether a subexpression was replaced.
    """
    nodes = {}

    replaced = False

    for node in list(graph.nodes):
        node_hash = (node.op, node.target, node.args, node.kwargs)
        if node_hash in nodes:
            node.replace_all_uses_with(nodes[node_hash])
            replaced = True
        else:
            nodes[node_hash] = node

    if replaced:
        graph.eliminate_dead_code()

    return replaced
