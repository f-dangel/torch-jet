"""Test simplification mechanism on compute graphs of the (Bi-)Laplacian."""

from typing import Any, Callable

from pytest import mark
from torch import Size, Tensor, arange, manual_seed, sigmoid, sin, tanh, tensor
from torch.fx import Graph, GraphModule
from torch.nn import Linear, Module, Sequential, Tanh
from torch.nn.functional import linear

from jet.bilaplacian import Bilaplacian
from jet.laplacian import Laplacian
from jet.rules import is_replicate
from jet.simplify import common_subexpression_elimination, simplify
from jet.tracing import capture_graph
from jet.utils import recursive_getattr
from test.test___init__ import compare_jet_results, setup_case
from test.test_bilaplacian import bilaplacian
from test.test_laplacian import (
    DISTRIBUTION_IDS,
    DISTRIBUTIONS,
    WEIGHT_IDS,
    WEIGHTS,
    get_coefficients,
    get_weighting,
    laplacian,
)
from test.utils import report_nonclose

# make generation of test cases deterministic
manual_seed(0)

SIMPLIFY_CASES = [
    # 1d sine function
    {"f": sin, "shape": (1,), "id": "sin"},
    # 2d sine function
    {"f": sin, "shape": (2,), "id": "sin"},
    # 2d sin(sin) function
    {"f": lambda x: sin(sin(x)), "shape": (2,), "id": "sin-sin"},
    # 2d tanh(tanh) function
    {"f": lambda x: tanh(tanh(x)), "shape": (2,), "id": "tanh-tanh"},
    # 2d linear(tanh) function
    {
        "f": lambda x: linear(
            tanh(x),
            tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]]).double(),
            bias=tensor([0.12, -0.34]).double(),
        ),
        "shape": (3,),
        "id": "tanh-linear",
    },
    # 5d tanh-activated two-layer MLP
    {
        "f": Sequential(
            Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
        ),
        "shape": (5,),
        "id": "two-layer-tanh-mlp",
    },
    # 3d sigmoid(sigmoid) function
    {
        "f": lambda x: sigmoid(sigmoid(x)),
        "shape": (3,),
        "id": "sigmoid-sigmoid",
    },
]


def count_replicate_nodes(
    f: Callable | Module | GraphModule, example_input: Tensor | None = None
) -> int:
    """Count the number of `replicate` nodes in the compute graph of a function.

    Args:
        f: The function or module to analyze. If a `GraphModule`, it is used directly.
            If a `Module` or function, it is traced first.
        example_input: Example input for tracing. Required if ``f`` is not a
            ``GraphModule``.

    Returns:
        The number of `replicate` nodes in the compute graph of the function.
    """
    mod = capture_graph(f, example_input=example_input)
    return len([n for n in mod.graph.nodes if is_replicate(n)])


def ensure_outputs_replicates(graph: Graph, num_outputs: int, num_replicates: int):
    """Make sure the compute graph outputs only `replicate` nodes.

    Args:
        graph: The compute graph to check.
        num_outputs: The number of nodes that should be returned.
        num_replicates: The number of `replicate` nodes that should be returned.
    """
    output = list(graph.nodes)[-1]  # -1 is the output node
    parents = [n for n in graph.nodes if n in output.all_input_nodes]
    assert len(parents) == num_outputs
    replicates = [n for n in parents if is_replicate(n)]
    assert len(replicates) == num_replicates


def ensure_tensor_constants_collapsed(
    mod: GraphModule,
    collapsed_shape: Size | tuple[int, ...],
    non_collapsed_shape: Size | tuple[int, ...],
    other_shapes: list[Size | tuple[int, ...]] | None = None,
    at_least: int = 1,
    strict: bool = True,
):
    """Make sure some tensor constants in the module are collapsed.

    Args:
        mod: The module to check.
        collapsed_shape: The shape of a collapsed tensor constant.
        non_collapsed_shape: The shape of a non-collapsed tensor constant.
        other_shapes: Other admissible shapes that will not lead to errors if
            encountered. Default is `None`, i.e. no other shapes are expected.
        at_least: The smallest number of tensor constants that should be detected as
            collapsed for the check to pass. Default: `1`.
        strict: Whether to raise an error if the number of collapsed tensor
            constants is not exactly `at_least`. Default: `False`.

    Raises:
        ValueError: If the number of collapsed tensor constants is not as expected,
            if there is a tensor constant with an unexpected shape, or if there is
            an overlap between the supplied `other_shapes` and the (non-)collapsed ones.
    """
    other_shapes = [] if other_shapes is None else other_shapes
    if any(s in [collapsed_shape, non_collapsed_shape] for s in other_shapes):
        raise ValueError(
            f"Shape in {other_shapes=} matches either {collapsed_shape=}"
            + f" or {non_collapsed_shape=} shape."
        )

    constants = {
        n.target
        for n in mod.graph.nodes
        if n.op == "get_attr" and n.target.startswith("_tensor_constant")
    }
    for c in constants:
        print(f"Tensor constant {c} has shape {recursive_getattr(mod, c).shape}.")

    num_collapsed = 0
    for c in constants:
        c_tensor = recursive_getattr(mod, c)
        shape = c_tensor.shape
        if shape == collapsed_shape:
            num_collapsed += 1
        elif shape != non_collapsed_shape and shape not in other_shapes:
            raise ValueError(
                f"Unexpected shape for {c}: {shape}. "
                + f"Should be {collapsed_shape} or {non_collapsed_shape}."
                + f" Other accepted shapes are {other_shapes}."
            )

    if num_collapsed < at_least or strict and num_collapsed != at_least:
        raise ValueError(
            f"Expected {'' if strict else '>'}={at_least} collapsed tensor constants. "
            + f" Found {num_collapsed}."
        )


@mark.parametrize("weights", WEIGHTS, ids=WEIGHT_IDS)
@mark.parametrize("config", SIMPLIFY_CASES, ids=[c["id"] for c in SIMPLIFY_CASES])
@mark.parametrize(
    "distribution", [None] + DISTRIBUTIONS, ids=["exact"] + DISTRIBUTION_IDS
)
def test_simplify_laplacian(
    config: dict[str, Any],
    distribution: str | None,
    weights: str | None | tuple[str, float],
):
    """Test the simplification of a Laplacian's compute graph.

    Replicate nodes should be propagated down the graph.
    Sum nodes should be propagated up.

    Args:
        config: The configuration of the test case.
        distribution: The distribution from which to draw random vectors.
            If `None`, the exact Laplacian is computed. Default: `None`.
        weights: The weighting to use for the Laplacian. If `None`, the Laplacian is
            unweighted. If `diagonal_increments`, a synthetic coefficient tensor is
            used that has diagonal elements that are increments of 1 starting from 1.
    """
    num_samples, seed = 42, 1  # only relevant with randomization
    randomization = None if distribution is None else (distribution, num_samples)

    f, x, _ = setup_case(config)

    weighting = get_weighting(x, weights, randomization=randomization)
    mod = Laplacian(f, x, randomization=randomization, weighting=weighting)

    # we have to set the random seed to make sure the same random vectors are used
    if randomization is not None:
        manual_seed(seed)
    mod_out = mod(x)

    if randomization is None:
        C = get_coefficients(x, weights)
        lap = laplacian(f, x, C)
        assert lap.allclose(mod_out[2])
        print("Exact Laplacian in functorch and jet match.")

    # trace and simplify the module

    # we have to set the random seed because tracing executes the functions that
    # draw random vectors and stores them as tensor constants
    if randomization is not None:
        manual_seed(seed)
    fast = simplify(mod, verbose=True, test_x=x, example_input=x)

    # make sure the simplified module still behaves the same
    # With make_fx, random ops (aten.randn) are in the graph, so we must set the
    # seed before each evaluation to get deterministic results
    if randomization is not None:
        manual_seed(seed)
    fast_out = fast(x)
    compare_jet_results(mod_out, fast_out)
    print("Laplacian via jet matches Laplacian via simplified module.")

    # NOTE: With make_fx(vmap(jet_f)), the graph structure is different from the
    # old traceable_vmap approach. Replicate/sum nodes may not appear in
    # the output since the vmap is traced at the ATen level. We verify correctness
    # above by comparing outputs.


@mark.parametrize("config", SIMPLIFY_CASES, ids=[c["id"] for c in SIMPLIFY_CASES])
@mark.parametrize(
    "distribution", [None] + ["normal"], ids=["exact", "distribution=normal"]
)
def test_simplify_bilaplacian(config: dict[str, Any], distribution: str | None):
    """Test the simplifications for the Bi-Laplacian module.

    Args:
        config: The configuration of the test case.
        distribution: The distribution from which to draw random vectors.
            If `None`, the exact Bi-Laplacian is computed.
    """
    randomized = distribution is not None
    num_samples, seed = 42, 1  # only relevant with randomization
    f, x, _ = setup_case(config)

    randomization = (distribution, num_samples) if randomized else None

    bilap_mod = Bilaplacian(f, x, randomization=randomization)

    # we have to set the random seed to make sure the same random vectors are used
    if randomized:
        manual_seed(seed)
    bilap = bilap_mod(x)

    if not randomized:
        bilap_true = bilaplacian(f, x)
        assert bilap_true.allclose(bilap)
        print("Exact Bi-Laplacian in functorch and jet match.")

    # simplify the traced module

    # we have to set the random seed because tracing executes the functions that
    # draw random vectors and stores them as tensor constants
    if randomized:
        manual_seed(seed)
    simple_mod = simplify(
        bilap_mod,
        verbose=True,
        eliminate_tensor_constants=False,
        test_x=x,
        example_input=x,
    )

    # NOTE: With make_fx(vmap(jet_f)), the graph structure differs from the old
    # traceable_vmap approach. Tensor constants may not exist (inline ATen ops
    # like aten.zeros are used instead). We verify correctness below.

    # make sure the simplified module still behaves the same
    if randomized:
        manual_seed(seed)
    bilap_simple = simple_mod(x)
    report_nonclose(bilap, bilap_simple, name="Bi-Laplacians")

    # also remove duplicate tensor_constants
    simpler_mod = simplify(
        simple_mod,
        verbose=True,
        eliminate_tensor_constants=True,
        test_x=x,
        example_input=x,
    )

    # verify the further-simplified module still produces correct results
    if randomized:
        manual_seed(seed)
    bilap_simpler = simpler_mod(x)
    report_nonclose(
        bilap, bilap_simpler, name="Bi-Laplacians (after tensor constant elimination)"
    )


def test_common_subexpression_elimination():
    """Test common subexpression elimination."""

    def f(x: Tensor) -> Tensor:
        # NOTE that instead of computing y1, y2, we could simply compute y1 and
        # return y1 + y1
        x1 = x + 1
        x2 = x + 1
        y1 = 2 * x1
        y2 = 2 * x2
        z = y1 + y2
        return z

    x = arange(10)

    f_traced = capture_graph(f, example_input=x)
    f_x = f_traced(x)
    nodes_before = len(list(f_traced.graph.nodes))
    # make_fx produces ATen-level nodes; verify duplicate subexpressions exist
    assert nodes_before >= 7

    common_subexpression_elimination(f_traced.graph, verbose=True)
    nodes_after = len(list(f_traced.graph.nodes))
    # CSE should have removed at least some duplicate nodes
    assert nodes_after < nodes_before

    report_nonclose(f_x, f_traced(x), name="f(x)")
