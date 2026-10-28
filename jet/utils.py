"""Utility functions for computing jets."""

from math import factorial, prod
from pathlib import Path
from typing import Any, Callable

from torch import Tensor, device, dtype, empty, manual_seed, randn
from torch.fx import GraphModule, Node
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.random import fork_rng

#: A jet leaf: ``(primal, c_1, ..., c_K)`` bundling a primal with its ``K``
#: Taylor coefficients. Type-checker hint only — not a runtime constructor.
Jet = tuple[Tensor, ...]

#: A pytree of ``Leaf``: arbitrarily nested ``tuple`` / ``list`` / ``dict``
#: whose leaves have type ``Leaf``. Three concrete leaf types appear in this
#: library:
#:
#: - ``PyTree[Tensor]`` — public input side (``mock_args``).
#: - ``PyTree[Jet]`` — public jet-form: arguments to and return value of the
#:   callable returned by :func:`jet.jet` and :func:`jet._rev_jet`, where each
#:   tensor leaf is replaced by a jet ``(primal, c_1, ..., c_K)``.
#: - ``PyTree[JetTuple | Tensor]`` — interpreter-internal
#:   form, returned by ``JetInterpreter.run``.
type PyTree[Leaf] = (
    Leaf | tuple[PyTree[Leaf], ...] | list[PyTree[Leaf]] | dict[str, PyTree[Leaf]]
)


def _is_jet_leaf(x: Any) -> bool:
    """``True`` iff ``x`` is a jet tuple: a ``tuple`` of one or more tensors."""
    return (
        isinstance(x, tuple) and len(x) >= 1 and all(isinstance(e, Tensor) for e in x)
    )


def run_seeded(f: Callable, seed: int, *args, **kwargs):
    """Run a callable with a specific random seed, restoring the RNG state afterwards.

    Args:
        f: The callable to execute.
        seed: The random seed to use.
        *args: Positional arguments forwarded to ``f``.
        **kwargs: Keyword arguments forwarded to ``f``.

    Returns:
        The return value of ``f(*args, **kwargs)``.
    """
    with fork_rng():
        manual_seed(seed)
        return f(*args, **kwargs)


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


def validate_randomization(
    randomization: tuple[str, int] | None, supported_distributions: list[str]
):
    """Validate the randomization arguments.

    Does nothing if ``randomization`` is ``None``.

    Args:
        randomization: Tuple of (distribution name, number of samples), or ``None``.
        supported_distributions: List of supported distribution names.

    Raises:
        ValueError: If the distribution is not supported or the number of samples
            is not positive.
    """
    if randomization is None:
        return
    distribution, num_samples = randomization
    if distribution not in supported_distributions:
        raise ValueError(f"Unsupported {distribution=} ({supported_distributions=}).")
    if num_samples <= 0:
        raise ValueError(f"{num_samples=} must be positive.")


def require_single_tensor_input(
    mock_args: tuple[PyTree[Tensor], ...], transform: str
) -> Tensor:
    """Return the sole input tensor, or raise if ``f`` is not single-tensor.

    The Laplacian / Bi-Laplacian transforms take ``mock_args`` as a tuple
    matching ``f``'s positional arguments (mirroring :func:`jet.jet`), but
    only single-tensor inputs are supported so far.

    Args:
        mock_args: Mock positional arguments for ``f``.
        transform: Name of the calling transform, used in the error message.

    Returns:
        The single input tensor ``mock_args[0]``.

    Raises:
        NotImplementedError: If ``mock_args`` is not a one-tuple of a tensor.
    """
    if len(mock_args) != 1 or not isinstance(mock_args[0], Tensor):
        raise NotImplementedError(
            f"{transform} currently supports a single-tensor input only; got "
            f"{len(mock_args)} positional argument(s). Pytree inputs (multiple "
            "arguments or non-tensor leaves) are not yet supported."
        )
    return mock_args[0]


def require_single_tensor_output(result: PyTree[Jet], transform: str) -> Jet:
    """Return the sole output jet, or raise if ``f`` is not single-tensor.

    Mirrors :func:`require_single_tensor_input` on the output side: the
    Laplacian / Bi-Laplacian transforms unpack a single jet tuple
    ``(f_0, ..., f_K)``, which only exists when ``f`` returns one tensor. A
    pytree-valued output yields a nested structure instead, so guard it here
    to surface a clear error rather than an opaque unpacking failure.

    Args:
        result: The pytree of jets returned by the jet-transformed ``f``.
        transform: Name of the calling transform, used in the error message.

    Returns:
        The single output jet ``result`` (a tuple of coefficient tensors).

    Raises:
        NotImplementedError: If ``f`` returns a pytree of tensors rather than
            a single tensor.
    """
    if not _is_jet_leaf(result):
        raise NotImplementedError(
            f"{transform} currently supports a single-tensor output only; ``f`` "
            "returned a pytree of tensors, which is not yet supported."
        )
    return result


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
            style["fillcolor"] = (
                "OrangeRed" if target_name in self._SUM_TARGETS else "white"
            )
        return style


def visualize_graph(
    mod: GraphModule, savefile: str, name: str = "", use_custom: bool = False
):
    """Visualize the compute graph of a module.

    Requires the optional ``viz`` dependency (``pydot``); install it with
    ``pip install jet-for-pytorch[viz]`` (plus the ``graphviz`` system package).

    Supported formats: ``.png``, ``.pdf``, ``.svg`` (inferred from *savefile*).

    Args:
        mod: The module whose compute graph to visualize.
        savefile: The path to the file where the graph should be saved.
        name: A name for the graph, used in the visualization.
        use_custom: If ``True``, highlight sum nodes in orange-red and use white
            for other operations. Defaults to ``False``.

    Raises:
        ValueError: If *savefile* has an unsupported extension.

    Examples:
        >>> from os.path import exists, join
        >>> from tempfile import TemporaryDirectory
        >>> from torch import zeros
        >>> from jet import capture_graph, visualize_graph
        >>> mod, _ = capture_graph(lambda x: x + x, (zeros(3),))
        >>> with TemporaryDirectory() as tmp:
        ...     path = join(tmp, "graph.svg")
        ...     visualize_graph(mod, path)
        ...     written = exists(path)
        >>> written
        True
    """
    cls = _CustomDrawer if use_custom else FxGraphDrawer
    drawer = cls(mod, name)
    dot_graph = drawer.get_dot_graph()

    creators = {
        ".png": dot_graph.create_png,
        ".pdf": dot_graph.create_pdf,
        ".svg": dot_graph.create_svg,
    }

    suffix = Path(savefile).suffix.lower()
    creator = creators.get(suffix)
    if creator is None:
        supported = ", ".join(sorted(creators))
        raise ValueError(f"Unsupported file format {suffix!r}. Use one of: {supported}")

    # Render before opening the file so a failure leaves no empty artifact.
    data = creator()
    with open(savefile, "wb") as f:
        f.write(data)
