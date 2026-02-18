"""Utility functions for computing jets."""

from math import factorial, prod

from torch import Tensor, device, dtype, empty, randn
from torch.fx import GraphModule, Node
from torch.fx.passes.graph_drawer import FxGraphDrawer

# type annotation for arguments and Taylor coefficients in input and output space
Primal = Tensor
Value = Tensor
# primals and values form a tuple
PrimalAndCoefficients = tuple[Primal, ...]
ValueAndCoefficients = tuple[Value, ...]


def integer_partitions(n: int, I: int = 1):  # noqa: E741
    """Compute the integer partitions of a positive integer.

    Taken from: https://stackoverflow.com/a/44209393.

    Args:
        n: Positive integer.
        I: Minimum value of the partition's first entry. Default: `1`.

    Yields:
        Tuple of integers representing the integer partition.
    """
    yield (n,)
    for i in range(I, n // 2 + 1):
        for p in integer_partitions(n - i, i):
            yield (i,) + p


def multiplicity(sigma: tuple[int, ...]) -> float:
    """Compute the scaling of a summand in Faa di Bruno's formula.

    Args:
        sigma: Tuple of integers representing the integer partitioning.

    Returns:
        Multiplicity of the summand.

    Raises:
        ValueError: If the multiplicity is not an integer.
    """
    # see the scheme above the 'Variations' section here:
    # https://en.wikipedia.org/wiki/Fa%C3%A0_di_Bruno%27s_formula
    k = sum(sigma)
    n_i = {i + 1: sigma.count(i + 1) for i in range(k)}
    multiplicity = (
        factorial(k)
        / prod(factorial(eta) for eta in sigma)
        / prod(factorial(n) for n in n_i.values())
    )
    if not multiplicity.is_integer():
        raise ValueError(f"Multiplicity should be an integer, but got {multiplicity}.")
    return multiplicity


def rademacher(*shape: int, dtype: dtype | None = None, device: device | None = None):
    """Sample from Rademacher distribution.

    Args:
        shape: Shape of the output tensor.
        dtype: Data type of the output tensor. Default: `None`.
        device: Device of the output tensor. Default: `None`.

    Returns:
        Tensor sampled from Rademacher distribution (+1 and -1 entries).
    """
    return (
        empty(*shape, dtype=dtype, device=device).fill_(0.5).bernoulli().mul_(2).sub_(1)
    )


def sample(x_meta: Tensor, distribution: str, shape: tuple[int, ...]) -> Tensor:
    """Sample a random tensor with the same dtype and device as a given tensor.

    Args:
        x_meta: Tensor whose dtype and device are to be matched.
        distribution: Distribution to sample from. Supported: "normal", "rademacher".
        shape: Shape of the output tensor.

    Returns:
        Sampled tensor.
    """
    sample_func = {"normal": randn, "rademacher": rademacher}[distribution]
    return sample_func(*shape, dtype=x_meta.dtype, device=x_meta.device)


class _CustomDrawer(FxGraphDrawer):
    """FxGraphDrawer that highlights sum nodes and de-emphasizes other operations.

    Using this custom drawer to visualize graphs is helpful to troubleshoot collapsing.

    Sum nodes (``aten.sum``) are colored orange-red, other ``call_function`` nodes
    are white. All other node types (placeholders, constants, output) keep their
    default colors.
    """

    _SUM_TARGETS: set[str] = {
        "torch.ops.aten.sum.dim_IntList",
        "torch.ops.aten.sum.default",
    }

    def _get_node_style(self, node: Node) -> dict[str, str]:
        """Return the dot attributes for a graph node.

        Args:
            node: The FX graph node to style.

        Returns:
            Dictionary of dot graph attributes for the node.
        """
        style = super()._get_node_style(node)
        if node.op == "call_function":
            target_name = node._pretty_print_target(node.target)
            style["fillcolor"] = "OrangeRed" if target_name in self._SUM_TARGETS else "white"
        return style


def visualize_graph(
    mod: GraphModule, savefile: str, name: str = "", use_custom: bool = False
):
    """Visualize the compute graph of a module and store it as .png.

    Args:
        mod: The module whose compute graph to visualize.
        savefile: The path to the file where the graph should be saved.
        name: A name for the graph, used in the visualization.
        use_custom: If ``True``, highlight sum nodes in orange-red and use white
            for other operations. Defaults to ``False``.
    """
    cls = _CustomDrawer if use_custom else FxGraphDrawer
    drawer = cls(mod, name)
    dot_graph = drawer.get_dot_graph()
    with open(savefile, "wb") as f:
        f.write(dot_graph.create_png())
